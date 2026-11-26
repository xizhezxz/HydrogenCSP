"""Generate exactly once for each screened scheme of structures with <20 schemes."""
import os
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from pymatgen.core import Structure,Lattice
from pymatgen.io.cif import CifWriter
from torch_geometric.data import Batch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['PROJECT_ROOT']=str(ROOT)
from diffcsp.common.wyckoff_reassignment import catalog_from_reference,orbit_residual
from diffcsp.common.wyckoff_enumeration import enumerate_allocations,instantiate_allocation
from diffcsp.common.pxrd_wyckoff import unavoidable_overlap
from sample_wyckoff_reassigned import load_sampling_helpers


def write_json(path,data):
    temp=path.with_suffix('.tmp.json')
    for attempt in range(20):
        try:
            temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.5)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/framework_wyckoff_exhaustive_lt20')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    prepared_path=out/'enumerated_inputs.pt'
    refs=torch.load(ROOT/'results/framework_fixed/samples_attempts1_to10.pt',map_location='cpu',weights_only=False)['input_data_batch'].to_data_list()
    if prepared_path.exists():
        prep=torch.load(prepared_path,map_location='cpu',weights_only=False)
        if prep['seed']!=args.seed:raise ValueError('Prepared seeds differ')
        inputs,records,selected=prep['inputs'],prep['records'],prep['selected']
    else:
        screening=json.loads((ROOT/'results/framework_wyckoff_reassigned/wyckoff_screening/screening_counts.json').read_text(encoding='utf-8'))
        selected=[row for row in screening['structures'] if 0<row['after']<20]
        inputs=[];records=[];failures=[]
        for n,row in enumerate(selected,1):
            j=row['structure_index'];data=refs[j];catalog=catalog_from_reference(data)
            lattice=Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),*data.angles.reshape(-1).tolist())
            counts=Counter(int(z) for z in data.atom_types.tolist() if z!=1)
            allocations=list(enumerate_allocations(counts,catalog['positions']))
            if len(allocations)!=row['before']:raise ValueError(f'Enumeration count mismatch: {row["material_id"]}')
            signatures=set();accepted=0
            for allocation in allocations:
                probe,metadata=instantiate_allocation(data,catalog,allocation,seed=0,minimum_distance=0,retries=1)
                if unavoidable_overlap(probe,metadata,lattice,scale=.6):continue
                signature=tuple(sorted((int(z),catalog['positions'][i]['letter']) for z,i in allocation))
                if signature in signatures:raise ValueError('Duplicate enumeration')
                signatures.add(signature);accepted+=1
                seed=args.seed+100003*j+accepted
                record={'job_index':len(records),'structure_index':j,'material_id':row['material_id'],
                        'scheme_index':accepted,'schemes_for_structure':row['after'],'seed':seed,
                        'spacegroup':catalog['spacegroup'],'hall_number':catalog['hall_number'],
                        'initialization_succeeded':False,'allocation':metadata}
                try:
                    item,metadata=instantiate_allocation(data,catalog,allocation,seed=seed)
                    residual=orbit_residual(item.frac_coords.numpy(),item)
                    if residual>2e-4:raise ValueError(f'Initial orbit mismatch: {residual}')
                    record.update(initialization_succeeded=True,allocation=metadata)
                    inputs.append(item)
                except Exception as exc:
                    failures.append({'material_id':row['material_id'],'scheme_index':accepted,'error':str(exc)})
                    inputs.append(None)
                records.append(record)
            if accepted!=row['after']:
                raise ValueError(f'Screened enumeration count mismatch for {row["material_id"]}: {accepted} != {row["after"]}')
            if n%50==0:print(f'ENUMERATED {n}/{len(selected)} structures, {len(records)} schemes',flush=True)
        write_json(out/'preparation_failures.json',failures)
        prep={'seed':args.seed,'inputs':inputs,'records':records,'selected':selected}
        torch.save(prep,prepared_path)
    # The sampler initializes x_T independently, so a packing failure of its
    # unused preprocessing coordinates must not suppress an enumerated scheme.
    # Keep the exact allocation and mark this warning, without substituting it.
    warnings=[]
    for i,item in enumerate(inputs):
        record=records[i]
        if item is None:
            data=refs[record['structure_index']];catalog=catalog_from_reference(data)
            by_letter={p['letter']:k for k,p in enumerate(catalog['positions'])}
            allocation=[(p['atomic_number'],by_letter[p['wyckoff_letter']]) for p in record['allocation']]
            item,metadata=instantiate_allocation(data,catalog,allocation,seed=record['seed'],minimum_distance=0,retries=1)
            inputs[i]=item
            record.update(initialization_succeeded=True,allocation=metadata,
                          initialization_warning='Preprocessing .25 A packing check failed; same scheme retained. Sampler independently draws x_T.')
        if 'initialization_warning' in record:
            warnings.append({'material_id':record['material_id'],'scheme_index':record['scheme_index'],
                             'warning':record['initialization_warning']})
    prep.update(inputs=inputs,records=records)
    torch.save(prep,prepared_path)
    write_json(out/'preparation_failures.json',[])
    write_json(out/'preparation_warnings.json',warnings)
    write_json(out/'generation_plan.json',records)
    summary={'selected_structures':len(selected),'enumerated_schemes':len(records),
             'selection_rule':prep.get('selection_rule','0 < screened scheme count < 20, after conservative unavoidable-overlap filtering; before PXRD selection'),
             'draws_per_scheme':1,'seed':args.seed,
             'initialization_failures':sum(item is None for item in inputs),
             'initialization_warnings':len(warnings),
             'scheme_counts_histogram':dict(Counter(str(sum(r['structure_index']==row['structure_index'] for r in records)) for row in selected))}
    write_json(out/'enumeration_summary.json',summary)
    print('ALL_SCHEMES_ENUMERATED',json.dumps(summary),flush=True)
    if args.prepare_only:return
    settings={'batch_size':args.batch_size,'seed':args.seed,'step_lr':1e-5,
              'model_path':str(ROOT/'hydra_jobs/fixlattice_alexmp20'),'sampler':'sample_spacegroup_hydrogen'}
    setting_path=out/'generation_settings.json'
    if setting_path.exists() and json.loads(setting_path.read_text(encoding='utf-8'))!=settings:
        raise ValueError('Generation settings changed; would duplicate or reorder draws')
    write_json(setting_path,settings)
    write_json(out/'run_status.json',{'status':'loading_model','total_schemes':len(records),
               'selected_structures':len(selected),'draws_per_scheme':1})
    load_model,lattices_to_params_shape=load_sampling_helpers()
    model,_,_=load_model(ROOT/'hydra_jobs/fixlattice_alexmp20',load_data=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=model.to(device).eval()
    if model.keep_coords:raise ValueError('Model coordinate sampling disabled')
    completed={};started=time.time();batches=out/'batches';batches.mkdir(exist_ok=True)
    if any(item is None for item in inputs):
        raise RuntimeError('Some schemes could not initialize; no replacement schemes sampled. See preparation_failures.json')
    for start in range(0,len(inputs),args.batch_size):
        end=min(start+args.batch_size,len(inputs));path=batches/f'jobs_{start:05d}_{end-1:05d}.pt'
        write_json(out/'run_status.json',{'status':'sampling','current_job_start':start+1,'current_job_end':end,
                   'completed':len(completed),'total_schemes':len(records),'selected_structures':len(selected),
                   'elapsed_seconds':time.time()-started})
        if path.exists():
            output=torch.load(path,map_location='cpu',weights_only=False)
        else:
            print(f'GENERATING jobs {start+1}-{end}/{len(inputs)}',flush=True)
            batch=Batch.from_data_list(inputs[start:end]).to(device)
            torch.manual_seed(args.seed+start)
            with torch.inference_mode():output,_=model.sample_spacegroup_hydrogen(batch,step_lr=1e-5)
            output={k:v.detach().cpu() for k,v in output.items() if isinstance(v,torch.Tensor)}
            offset=0;maximum=0.0
            for j,count in enumerate(output['num_atoms'].tolist(),start):
                maximum=max(maximum,orbit_residual(output['frac_coords'][offset:offset+count].numpy(),inputs[j]))
                offset+=count
            if maximum>2e-4:raise ValueError(f'Generated orbit residual: {maximum}')
            output['orbit_max_fractional_error']=maximum
            temp=path.with_suffix('.tmp.pt');torch.save(output,temp);temp.replace(path)
        offset=0
        for j,count in enumerate(output['num_atoms'].tolist(),start):
            record=records[j].copy();directory=out/'generated_cifs'/record['material_id'];directory.mkdir(parents=True,exist_ok=True)
            cif=directory/f'scheme_{record["scheme_index"]:03d}.cif'
            structure=Structure(Lattice(output['lattices'][j-start].numpy()),output['atom_types'][offset:offset+count].numpy(),
                                output['frac_coords'][offset:offset+count].numpy())
            CifWriter(structure,symprec=None,significant_figures=10).write_file(cif)
            record.update(cif_path=str(cif),num_atoms=count,status='generated_once',
                          initialization_seed=record['seed'],diffusion_batch_seed=args.seed+start)
            from diffcsp.common.pxrd_wyckoff import distances_valid
            mask=output['atom_types'][offset:offset+count].numpy()!=1
            valid,ratio=distances_valid(structure.lattice,
                                       output['atom_types'][offset:offset+count].numpy()[mask],
                                       structure.frac_coords[mask],scale=.6)
            record.update(generated_non_h_distance_passed=valid,generated_min_distance_ratio=ratio)
            completed[j]=record;offset+=count
        write_json(out/'generated_cifs_manifest.json',{'summary':summary,'records':[completed[k] for k in sorted(completed)]})
        print(f'SAVED {len(completed)}/{len(records)} unique schemes',flush=True)
    # One flattened sample per enumerated scheme, with a matching untouched reference.
    outputs=[torch.load(path,map_location='cpu',weights_only=False) for path in sorted(batches.glob('jobs_*.pt'))]
    lattices=torch.cat([x['lattices'] for x in outputs])[None]
    lengths,angles=lattices_to_params_shape(lattices)
    payload={'eval_setting':settings,'scheme_records':records,
             'input_data_batch':Batch.from_data_list([refs[r['structure_index']] for r in records]),
             'frac_coords':torch.cat([x['frac_coords'] for x in outputs])[None],
             'lattices':lattices,'lengths':lengths,'angles':angles,
             'num_atoms':torch.cat([x['num_atoms'] for x in outputs])[None],
             'atom_types':torch.cat([x['atom_types'] for x in outputs])[None]}
    torch.save(payload,out/'samples_all_schemes.pt')
    if len(completed)!=len(records):raise RuntimeError('Not all schemes generated')
    write_json(out/'run_status.json',{'status':'complete','completed':len(completed),**summary,
               'elapsed_seconds':time.time()-started})
    print('ALL_SCHEMES_GENERATED_ONCE',len(completed),flush=True)


if __name__=='__main__':main()
