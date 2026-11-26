"""Exhaustive <=20, staged PXRD ranking >20, then one diffusion per selected scheme."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import csv,json,sys,time,subprocess
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from pymatgen.core import Lattice
from ase.data import covalent_radii
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffcsp.common.wyckoff_reassignment import catalog_from_reference
from diffcsp.common.wyckoff_enumeration import enumerate_allocations,instantiate_allocation
from diffcsp.common.pxrd_wyckoff import FixedCellPXRD,fit_scheme,unavoidable_overlap
from screen_wyckoff_allocations import screen_count
from count_wyckoff_allocations import count_allocations

OUT=ROOT/'results/framework_wyckoff_hierarchical_top20'
SEED=20261019
def save(path,obj):
    text=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)
    temp=path.with_suffix('.tmp.json')
    for attempt in range(20):
        try:temp.write_text(text,encoding='utf-8');temp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.5)

def constraints(counts,positions,lattice):
    shortest=min(lattice.get_niggli_reduced_lattice().abc);bad=set();conflicts=set()
    for i,p in enumerate(positions):
        best=float('inf');ops=p['ops']
        for a in range(len(ops)):
            for b in range(a):
                if np.allclose(ops[a,:3,:3],ops[b,:3,:3],atol=1e-7):
                    best=min(best,lattice.get_distance_and_image(ops[a,:3,3],ops[b,:3,3])[0])
        for z in counts:
            if min(shortest,best)+1e-8<1.2*covalent_radii[z]:bad.add((z,i))
    fixed=[i for i,p in enumerate(positions) if p['dof']==0]
    for a,i in enumerate(fixed):
        for j in fixed[:a]:
            d=lattice.get_all_distances(positions[i]['ops'][:,:3,3],positions[j]['ops'][:,:3,3]).min()
            for z in counts:
                for w in counts:
                    if d+1e-8<.6*(covalent_radii[z]+covalent_radii[w]):conflicts.add(frozenset(((z,i),(w,j))))
    def valid(allocation):
        if any(x in bad for x in allocation):return False
        f=[x for x in allocation if positions[x[1]]['dof']==0]
        return not any(frozenset((x,y)) in conflicts for k,x in enumerate(f) for y in f[:k])
    return valid

def process(task):
    index,mid,data=task;directory=OUT/'selection'/mid;directory.mkdir(parents=True,exist_ok=True)
    cached=directory/'selected_inputs.pt'
    if cached.exists():return torch.load(cached,map_location='cpu',weights_only=False)['summary']
    catalog=catalog_from_reference(data);positions=catalog['positions']
    lattice=Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),*data.angles.reshape(-1).tolist())
    counts=Counter(int(z) for z in data.atom_types if z!=1)
    before=count_allocations(list(counts.values()),positions);after,details=screen_count(counts,positions,lattice,.6)
    valid=constraints(counts,positions,lattice);rng=np.random.default_rng(SEED+index)
    pool=[];seen=0;raw=0
    # Reservoir sampling over the entire screened stream avoids backtracking bias.
    for allocation in enumerate_allocations(counts,positions):
        raw+=1
        if not valid(allocation):continue
        seen+=1
        candidate=(seen,allocation)
        if len(pool)<256:pool.append(candidate)
        else:
            k=int(rng.integers(seen))
            if k<256:pool[k]=candidate
    assert raw==before and seen==after,(mid,raw,before,seen,after)
    pool.sort();engine=FixedCellPXRD(lattice) if after>20 else None
    target=engine.peaks(data.atom_types.numpy(),data.frac_coords.numpy()) if engine else None
    candidates=[];items={}
    for number,allocation in pool:
        seed=SEED+1000003*index+number
        item,meta=instantiate_allocation(data,catalog,allocation,seed,minimum_distance=0,retries=1)
        assert not unavoidable_overlap(item,meta,lattice,.6)
        entry={'enumerated_scheme_index':number,'allocation':meta,'seed':seed}
        if engine:
            fit=fit_scheme(engine,item,meta,target,starts=1,maxiter=20,seed=seed)
            entry['quick']=fit['best']
        candidates.append(entry);items[number]=item
    def rank(e,stage):return (not e[stage]['geometry_passed'],-e[stage]['cosine'],e['enumerated_scheme_index'])
    if engine:
        shortlist=sorted(candidates,key=lambda e:rank(e,'quick'))[:64]
        for entry in shortlist:
            fit=fit_scheme(engine,items[entry['enumerated_scheme_index']],entry['allocation'],target,starts=3,maxiter=60,seed=entry['seed'])
            entry['refined']=fit['best'];entry['refined_starts']=fit['starts']
        selected=sorted(shortlist,key=lambda e:rank(e,'refined'))[:20]
    else:selected=candidates
    assert len(selected)==min(after,20)
    records=[];inputs=[]
    for scheme,entry in enumerate(selected,1):
        inputs.append(items[entry['enumerated_scheme_index']])
        records.append({'structure_index':index,'material_id':mid,'scheme_index':scheme,'schemes_for_structure':len(selected),
            'screened_scheme_count':after,'enumerated_scheme_index':entry['enumerated_scheme_index'],
            'seed':entry['seed'],'spacegroup':catalog['spacegroup'],'hall_number':catalog['hall_number'],
            'initialization_succeeded':True,'allocation':entry['allocation'],
            'selection_method':'all_schemes' if after<=20 else 'hierarchical_pxrd',
            'selection_pxrd_cosine':entry.get('refined',{}).get('cosine'),
            'selection_geometry_passed':entry.get('refined',{}).get('geometry_passed')})
    summary={'structure_index':index,'material_id':mid,'before':before,'after':after,'selected':len(selected),
        'quick_evaluated':len(candidates) if engine else 0,'refined_evaluated':min(64,len(candidates)) if engine else 0,
        'selected_geometry_failed':sum(r['selection_geometry_passed'] is False for r in records)}
    save(directory/'ranking.json',{'summary':summary,'candidates':candidates,'selected':records})
    temp=cached.with_suffix('.tmp.pt');torch.save({'inputs':inputs,'records':records,'summary':summary},temp);temp.replace(cached)
    return summary

def main():
    OUT.mkdir(exist_ok=True);(OUT/'selection').mkdir(exist_ok=True)
    settings={'seed':SEED,'all_if_count_le':20,'selected_if_more':20,'uniform_reservoir_candidates':256,
        'quick_starts':1,'quick_iterations':20,'refined_candidates':64,'refined_starts':3,'refined_iterations':60,
        'ranking':'geometry pass first, then PXRD cosine descending','draws_per_selected_scheme':1,
        'note':'Top20 among evaluated candidates, not guaranteed global best; failed geometry candidates flagged if fewer than20 pass. No PXRD .95 cutoff.',
        'sampling':'sample_spacegroup_hydrogen, new coordinates; fitted coordinates used only for ranking',
        'pxrd':{'wavelength_A':1.5406,'two_theta_deg':[5,80],'step_deg':.04,'fwhm_deg':.15,'target':'full reference including H; fit non-H only'}}
    if (OUT/'selection_settings.json').exists():assert json.loads((OUT/'selection_settings.json').read_text(encoding='utf-8'))==settings
    save(OUT/'selection_settings.json',settings)
    refs=torch.load(ROOT/'results/framework_fixed/samples_attempts1_to10.pt',map_location='cpu',weights_only=False)['input_data_batch'].to_data_list()
    with (ROOT/'data/mc3d_H_cifs_le20/mc3d_H_cifs_le20.csv').open(encoding='utf-8',newline='') as f:source=list(csv.DictReader(f))
    tasks=[(i,source[int(d.source_row.item())]['material_id'],d) for i,d in enumerate(refs)]
    results=[];started=time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(process,t):t for t in tasks}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            save(OUT/'selection_status.json',{'status':'selecting','completed':len(results),'total':len(tasks),'elapsed_seconds':time.time()-started})
            print('SELECTED',len(results),len(tasks),result['material_id'],result['after'],result['selected'],flush=True)
    results.sort(key=lambda r:r['structure_index']);inputs=[];records=[]
    for row in results:
        prep=torch.load(OUT/'selection'/row['material_id']/'selected_inputs.pt',map_location='cpu',weights_only=False)
        inputs.extend(prep['inputs'])
        for r in prep['records']:r['job_index']=len(records);records.append(r)
    rule='All screened schemes if count<=20; otherwise hierarchical geometry/PXRD top20 from up to256 uniform candidates, top64 refined with3 starts.'
    torch.save({'seed':SEED,'inputs':inputs,'records':records,'selected':results,'selection_rule':rule},OUT/'enumerated_inputs.pt')
    save(OUT/'screening_counts.json',{'structures':results})
    save(OUT/'selection_status.json',{'status':'complete','structures':len(results),'selected_schemes':len(records),
        'selected_geometry_failed':sum(r['selected_geometry_failed'] for r in results)})
    (OUT/'README.md').write_text('# 分层筛选与生成\n\n'+rule+'\n\n每方案重新生成一次，固定晶格、非H服从所选轨道，H独立采样。PXRD拟合坐标不作为固定框架。\n\nselection/含完整排序证据；几何拟合未通过但因不足20而入选的方案明确标记。候选数取不可避免重叠筛选后的精确计数。\n',encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/sample_wyckoff_exhaustive_lt20.py'),'--output',str(OUT),'--seed',str(SEED),'--batch-size','64'],check=True)

if __name__=='__main__':main()
