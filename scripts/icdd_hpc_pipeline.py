"""Portable ICDD selection/generation/H-only relaxation, four disjoint shards."""
import os, sys, json, argparse, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ['PROJECT_ROOT']=str(ROOT)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)

def prepare_one(path,index):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from pymatgen.core import Composition, Lattice
    from pymatgen.io.cif import CifParser,CifFile
    from pymatgen.core.operations import SymmOp
    from diffcsp.common.wyckoff_reassignment import catalog_from_reference,key_op
    parser=CifParser(path,frac_tolerance=0,site_tolerance=1e-5)
    block=next(iter(parser._cif.data.values()));raw=block.data
    composition=Composition(raw['_chemical_formula_sum'])
    lattice=Lattice.from_parameters(*[float(raw['_cell_length_'+k]) for k in 'abc'],*[float(raw['_cell_angle_'+k]) for k in ['alpha','beta','gamma']])
    operations=raw.get('_space_group_symop_operation_xyz',raw.get('_symmetry_equiv_pos_as_xyz'))
    if not operations:raise ValueError('Missing explicit CIF symmetry operations')
    ops=[SymmOp.from_xyz_str(x).affine_matrix for x in operations]
    # Generic labelled orbits encode the declared operations without missing-H artifacts.
    coords=[];zs=[]
    for z,point in zip([6,7,8],[[.123147,.237193,.341287],[.319137,.417193,.193271],[.291321,.173219,.431137]]):
        orbit=[]
        for op in ops:
            x=(op@np.r_[point,1])[:3]%1
            if not any(lattice.get_distance_and_image(x,y)[0]<1e-6 for y in orbit):orbit.append(x)
        coords.extend(orbit);zs.extend([z]*len(orbit))
    def data(x,z):return Data(frac_coords=torch.tensor(np.array(x),dtype=torch.float64),atom_types=torch.tensor(z,dtype=torch.long),lengths=torch.tensor([lattice.abc],dtype=torch.float64),angles=torch.tensor([lattice.angles],dtype=torch.float64),num_atoms=len(z),num_nodes=len(z),source_row=torch.tensor([index]))
    catalog=catalog_from_reference(data(coords,zs),tolerance=.001)
    if {key_op(x) for x in ops}!={key_op(x) for x in catalog['symmetry_ops']}:
        # Rounding differences below tolerance can occur in rational translations.
        for op in ops:
            matches=[q for q in catalog['symmetry_ops'] if np.allclose(q[:3,:3],op[:3,:3],atol=1e-6) and lattice.get_distance_and_image(q[:3,3],op[:3,3])[0]<1e-5]
            if not matches:raise ValueError('Declared symmetry operations do not match catalog')
        if len(ops)!=len(catalog['symmetry_ops']):raise ValueError('Symmetry operation count mismatch')
    keep=[i for i,s in enumerate(raw['_atom_site_type_symbol']) if s not in ('H','D','T')]
    for loop in block.loops:
        if '_atom_site_type_symbol' in loop:
            for k in loop:raw[k]=[raw[k][i] for i in keep]
    heavy=CifParser.from_str(str(CifFile({'framework':block})),frac_tolerance=0,site_tolerance=1e-5).parse_structures(primitive=False)[0]
    if not heavy.is_ordered:raise ValueError('Non-H framework not ordered')
    ratios=[float(heavy.composition[e]/v) for e,v in composition.items() if str(e)!='H']
    z=float(raw['_cell_formula_units_Z']) if '_cell_formula_units_Z' in raw else float(np.median(ratios))
    if not np.allclose(ratios,z,atol=1e-5):raise ValueError('Formula/Z and framework count disagree')
    nh=float(composition.get('H',0)*z)
    if nh<1 or abs(nh-round(nh))>1e-5:raise ValueError(f'Noninteger or missing hydrogen count: {nh}')
    nh=int(round(nh))
    if len(heavy)+nh>=30:raise ValueError('Full stoichiometric cell no longer <30 atoms')
    rng=np.random.default_rng(20261022+index)
    full=data(np.concatenate([heavy.frac_coords,rng.random((nh,3))]),list(heavy.atomic_numbers)+[1]*nh)
    for field in ['frac_coords','lengths','angles']:setattr(full,field,getattr(full,field).float())
    return full,catalog,dict(card=path.stem,formula=str(composition),nonH_atoms=len(heavy),H_atoms=nh,num_atoms=len(heavy)+nh,Z=z,spacegroup=catalog['spacegroup'],hall_number=catalog['hall_number'],H_coordinate_policy='formula-based count; independent diffusion; source H positions not inherited')

def select(args,rows):
    import torch
    import run_wyckoff_hierarchical20 as hierarchy
    hierarchy.OUT=ROOT/'results';hierarchy.SEED=20261022
    # PXRD ranking uses only the source heavy framework, never random H placeholders.
    from diffcsp.common.pxrd_wyckoff import FixedCellPXRD
    class HeavyPXRD(FixedCellPXRD):
        def peaks(self,zs,xyz):
            mask=zs!=1
            return super().peaks(zs[mask],xyz[mask])
    hierarchy.FixedCellPXRD=HeavyPXRD
    for row in rows:
        mid=row['card'];directory=ROOT/'results/selection'/mid
        if (directory/'selected_inputs.pt').exists():continue
        try:
            data,catalog,meta=prepare_one(ROOT/row['input'],row['index'])
            hierarchy.catalog_from_reference=lambda unused:catalog
            summary=hierarchy.process((row['index'],mid,data))
            save(directory/'input_metadata.json',meta)
            print('SELECT',mid,summary,flush=True)
        except Exception:
            save(directory/'error.json',dict(card=mid,error=traceback.format_exc()))
            print('SELECT_ERROR',mid,flush=True)

def generate(args,rows):
    import torch,numpy as np
    from torch_geometric.data import Batch
    from pymatgen.core import Structure,Lattice
    from pymatgen.io.cif import CifWriter
    from sample_wyckoff_reassigned import load_sampling_helpers
    from diffcsp.common.wyckoff_reassignment import orbit_residual
    model=None;pending=[]
    for row in rows:
        prepared=ROOT/'results/selection'/row['card']/'selected_inputs.pt'
        if not prepared.exists():continue
        prep=torch.load(prepared,map_location='cpu',weights_only=False)
        for number,(item,record) in enumerate(zip(prep['inputs'],prep['records'])):
            if args.scheme_limit and number>=args.scheme_limit:break
            key=f"{row['card']}/scheme_{record['scheme_index']:03d}"
            cif=ROOT/'results/generated_cifs'/f'{key}.cif';meta=ROOT/'results/generation_records'/f'{key}.json'
            if cif.exists() and meta.exists():continue
            pending.append((item,record,key,cif,meta))
    if not pending:return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    load_model,_=load_sampling_helpers();model,_,_=load_model(ROOT/'hydra_jobs/fixlattice_alexmp20',load_data=False)
    model=model.cuda().eval()
    import hashlib
    for start in range(0,len(pending),args.batch_size):
        chunk=pending[start:start+args.batch_size]
        keys=[x[2] for x in chunk]
        seed=(20261022+int(hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:8],16))%(2**32)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        print('GENERATING_BATCH',start,len(chunk),len(pending),flush=True)
        # Fail visibly on OOM rather than silently changing requested batch size.
        with torch.inference_mode():output,_=model.sample_spacegroup_hydrogen(Batch.from_data_list([x[0] for x in chunk]).cuda(),step_lr=1e-5)
        output={k:v.detach().cpu() for k,v in output.items() if isinstance(v,torch.Tensor)}
        counts=output['num_atoms'].tolist()
        if counts!=[int(x[0].num_atoms) for x in chunk]:raise ValueError('Batch atom count mismatch')
        offset=0
        for j,((item,record,key,cif,meta),count) in enumerate(zip(chunk,counts)):
            xyz=output['frac_coords'][offset:offset+count].numpy()
            zs=output['atom_types'][offset:offset+count].numpy();offset+=count
            if not np.array_equal(zs,item.atom_types.numpy()):raise ValueError('Batch species mismatch')
            residual=orbit_residual(xyz,item)
            if residual>2e-4:raise ValueError(f'{key}: Wyckoff residual {residual}')
            structure=Structure(Lattice(output['lattices'][j].numpy()),zs,xyz)
            cif.parent.mkdir(parents=True,exist_ok=True)
            temp=cif.with_suffix('.tmp.cif');CifWriter(structure,symprec=None,significant_figures=12).write_file(temp);temp.replace(cif)
            save(meta,dict(**record,key=key,diffusion_seed=seed,diffusion_batch_size=len(chunk),diffusion_batch_index=j,diffusion_batch_keys=keys,orbit_residual=residual,status='generated'))
            print('GENERATED',key,flush=True)

def relax(args,rows):
    import numpy as np,torch
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from pymatgen.io.cif import CifParser,CifWriter
    from pymatgen.io.ase import AseAtomsAdaptor
    from relax_generated_cifs_with_equiformer import import_equiformer
    torch.set_num_threads(args.threads);calc=None
    for row in rows:
        for cif in sorted((ROOT/'results/generated_cifs'/row['card']).glob('*.cif')):
            key=f"{row['card']}/{cif.stem}";dest=ROOT/'results/relaxed_cifs'/f'{key}.cif';record=ROOT/'results/relax_records'/f'{key}.json'
            if dest.exists() and record.exists() and json.loads(record.read_text()).get('status')=='ok':continue
            try:
                if calc is None:calc=import_equiformer(ROOT/'equiformer_v3')(checkpoint_path=str(ROOT/'models/omat24-mptrj-salex_gradient.pt'),cpu=False,seed=0)
                s=CifParser(cif,frac_tolerance=0,site_tolerance=1e-10).parse_structures(primitive=False)[0]
                a=AseAtomsAdaptor.get_atoms(s);fixed=a.numbers!=1;pos=a.positions.copy();cell=a.cell.array.copy();a.calc=calc
                d=a.get_all_distances(mic=True);np.fill_diagonal(d,np.inf)
                # Prevent graph-neighbor exclusion for near-coincident distinct atoms.
                if d.min()<.01:raise ValueError(f'Atomic overlap below 0.01 A: {d.min()}')
                initial=float(a.get_potential_energy());a.set_constraint(FixAtoms(mask=fixed))
                log=ROOT/'results/relax_logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
                opt=BFGS(a,logfile=str(log));converged=bool(opt.run(fmax=.03,steps=500))
                final=float(a.get_potential_energy());force=float(np.linalg.norm(a.get_forces()[~fixed],axis=1).max())
                delta=float(np.abs(a.positions[fixed]-pos[fixed]).max());cell_delta=float(np.abs(a.cell.array-cell).max())
                assert delta==0 and cell_delta==0
                dest.parent.mkdir(parents=True,exist_ok=True);CifWriter(AseAtomsAdaptor.get_structure(a),symprec=None,significant_figures=12).write_file(dest)
                save(record,dict(key=key,status='ok',converged=converged,steps=opt.nsteps,num_atoms=len(a),initial_energy_eV=initial,relaxed_energy_eV=final,initial_eV_per_atom=initial/len(a),relaxed_eV_per_atom=final/len(a),delta_relax_eV_per_atom=(final-initial)/len(a),max_H_force_eV_A=force,framework_displacement_A=delta,lattice_change_A=cell_delta))
                print('RELAXED',key,converged,final/len(a),flush=True)
            except Exception:save(record,dict(key=key,status='error',error=traceback.format_exc()))

def collect():
    rows=json.loads((ROOT/'manifest.json').read_text());summary={'input_materials':len(rows)}
    for folder in ['selection','generation_records','relax_records']:
        files=list((ROOT/'results'/folder).rglob('*.json'))
        if folder=='relax_records':
            records=[json.loads(p.read_text()) for p in files];save(ROOT/'results/all_relax_results.json',records)
            summary.update(relaxed=sum(r['status']=='ok' for r in records),converged=sum(r.get('converged',False) for r in records),relax_errors=sum(r['status']=='error' for r in records))
    summary['selected_materials']=len(list((ROOT/'results/selection').glob('*/selected_inputs.pt')))
    summary['selected_schemes']=sum(json.loads(p.read_text())['summary']['selected'] for p in (ROOT/'results/selection').glob('*/ranking.json'))
    summary['generated']=len(list((ROOT/'results/generation_records').rglob('*.json')))
    save(ROOT/'results/summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','generate','relax','collect']);p.add_argument('--shard',type=int,choices=range(4),default=0);p.add_argument('--threads',type=int,default=4);p.add_argument('--limit',type=int)
    p.add_argument('--scheme-limit',type=int,help='Smoke test only; omitted in production')
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    if args.batch_size<1:p.error('--batch-size must be positive')
    if args.stage=='collect':collect()
    else:
        rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['shard']==args.shard]
        if args.limit:rows=rows[:args.limit]
        globals()[args.stage](args,rows)
