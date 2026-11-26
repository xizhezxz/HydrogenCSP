"""Conservative, coordinate-independent overlap screening of Wyckoff schemes.

Count schemes exactly after applying necessary geometric conditions only:
same-orbit constant separations, distances between fixed sites, periodic self images.
This does NOT assert chemical plausibility of all remaining schemes.
"""
from collections import Counter
from functools import lru_cache
from math import prod
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from ase.data import covalent_radii, chemical_symbols
from pymatgen.core import Lattice

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffcsp.common.wyckoff_reassignment import catalog_from_reference
from count_wyckoff_allocations import count_allocations,verify_counter


def screen_count(counts,positions,lattice,scale):
    zs=tuple(sorted(counts));targets=tuple(counts[z] for z in zs)
    radii=np.array([covalent_radii[z] for z in zs])
    if np.any(~np.isfinite(radii)) or np.any(radii<=0):
        raise ValueError('Unknown covalent radius')
    shortest=min(lattice.get_niggli_reduced_lattice().abc)
    # Constant displacement pairs cannot be repaired by any free coordinate.
    constant_min=[]
    for p in positions:
        ops=p['ops'];best=np.inf
        for i in range(len(ops)):
            for j in range(i):
                if np.allclose(ops[i,:3,:3],ops[j,:3,:3],atol=1e-7):
                    d=lattice.get_distance_and_image(ops[i,:3,3],ops[j,:3,3])[0]
                    best=min(best,d)
        constant_min.append(best)
    allowed=[[shortest+1e-8>=2*scale*r and d+1e-8>=2*scale*r
              for d in constant_min] for r in radii]
    # Unlimited free-parameter orbit types; the same letter can occur repeatedly.
    free=[]
    for zi,n in enumerate(targets):
        ways=[0]*(n+1);ways[0]=1
        for i,p in enumerate(positions):
            if p['dof']>0 and allowed[zi][i]:
                for k in range(p['multiplicity'],n+1):
                    ways[k]+=ways[k-p['multiplicity']]
        free.append(ways)
    finite=[i for i,p in enumerate(positions) if p['dof']==0
            and p['multiplicity']<=max(targets)]
    conflicts={}
    for ai,i in enumerate(finite):
        xi=positions[i]['ops'][:,:3,3]
        for aj,j in enumerate(finite[:ai]):
            xj=positions[j]['ops'][:,:3,3]
            distance=float(lattice.get_all_distances(xi,xj).min())
            for zi,ri in enumerate(radii):
                for zj,rj in enumerate(radii):
                    if distance+1e-8<scale*(ri+rj):
                        conflicts[(ai,zi,aj,zj)]=True

    @lru_cache(None)
    def count_finite(k,remaining,occupied):
        if k==len(finite):
            return prod(free[z][n] for z,n in enumerate(remaining))
        # Do not count orbit order or permutations of identical atoms.
        total=count_finite(k+1,remaining,occupied)
        i=finite[k];m=positions[i]['multiplicity']
        for zi,n in enumerate(remaining):
            if n<m or not allowed[zi][i]:continue
            valid=True
            for zj,mask in enumerate(occupied):
                for previous in range(k):
                    if (mask>>previous)&1 and (k,zi,previous,zj) in conflicts:
                        valid=False;break
                if not valid:break
            if valid:
                residual=list(remaining);residual[zi]-=m
                masks=list(occupied);masks[zi]|=1<<k
                total+=count_finite(k+1,tuple(residual),tuple(masks))
        return total
    result=count_finite(0,targets,tuple(0 for _ in targets))
    details={'shortest_lattice_translation_ang':shortest,
             'forbidden_element_orbit_types':[{'element':chemical_symbols[z],
                 'wyckoff_letter':positions[i]['letter'],
                 'input_multiplicity':positions[i]['multiplicity'],
                 'minimum_constant_pair_distance_ang':None if not np.isfinite(constant_min[i]) else constant_min[i],
                 'self_distance_cutoff_ang':2*scale*covalent_radii[z]}
                for zi,z in enumerate(zs) for i in range(len(positions)) if not allowed[zi][i]],
             'fixed_position_pair_conflicts':len(conflicts)}
    return result,details


def histogram(values):
    h={'0':0,'1':0,'2-10':0,'11-100':0,'101-1000':0,'1001-10000':0,'>10000':0}
    for n in values:
        key='0' if n==0 else '1' if n==1 else '2-10' if n<=10 else '11-100' if n<=100 else '101-1000' if n<=1000 else '1001-10000' if n<=10000 else '>10000'
        h[key]+=1
    return h


def test_counter():
    verify_counter()
    lattice=Lattice.cubic(2)
    def fixed(x,letter):
        op=np.eye(4);op[:3,:3]=0;op[:3,3]=x
        return {'ops':op[None],'dof':0,'multiplicity':1,'letter':letter}
    close=[fixed([0,0,0],'a'),fixed([.05,0,0],'b')]
    far=[fixed([0,0,0],'a'),fixed([.5,.5,.5],'b')]
    assert count_allocations([1,1],close)==2
    assert screen_count({6:1,8:1},close,lattice,.6)[0]==0
    assert screen_count({6:1,8:1},far,lattice,.6)[0]==2
    assert screen_count({6:1,8:1},far,lattice,0)[0]==2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale',type=float,default=.60)
    args=parser.parse_args()
    test_counter()
    folder=ROOT/'results/framework_wyckoff_reassigned'
    out=folder/'wyckoff_screening';out.mkdir(exist_ok=True)
    previous=json.loads((folder/'wyckoff_allocation_counts.json').read_text(encoding='utf-8'))
    lookup={r['structure_index']:r for r in previous['structures']}
    refs=torch.load(ROOT/'results/framework_fixed/samples_attempts1_to10.pt',map_location='cpu',weights_only=False)['input_data_batch'].to_data_list()
    records=[]
    for j,data in enumerate(refs):
        old=lookup[j]
        catalog=catalog_from_reference(data)
        positions=catalog['positions']
        counts=Counter(int(z) for z in data.atom_types.tolist() if z!=1)
        raw=count_allocations(list(counts.values()),positions)
        if raw!=old['feasible_allocation_count']:
            raise ValueError(f'Initial count mismatch for {j}')
        lattice=Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),*data.angles.reshape(-1).tolist())
        kept,details=screen_count(counts,positions,lattice,args.scale)
        if not 0<=kept<=raw:raise ValueError('Invalid screened count')
        # Check references under the same conservative rule, reporting exceptions.
        zs=data.atom_types.numpy();nonh=zs!=1;heavy_zs=zs[nonh]
        xyz=data.frac_coords.numpy()[nonh]
        distances=lattice.get_all_distances(xyz,xyz)
        np.fill_diagonal(distances,details['shortest_lattice_translation_ang'])
        radius=covalent_radii[heavy_zs]
        ratio=float(np.min(distances/(radius[:,None]+radius[None,:])))
        records.append({'structure_index':j,'material_id':old['material_id'],
                        'spacegroup':catalog['spacegroup'],'before':raw,'after':kept,
                        'removed':raw-kept,'retained_fraction':kept/raw,
                        'reference_min_distance_over_radius_sum':ratio,
                        'reference_passes_same_distance_rule':ratio+1e-8>=args.scale,
                        'screening_details':details})
        if (j+1)%100==0:print(f'Screened {j+1}/{len(refs)}',flush=True)
    before=[r['before'] for r in records];after=[r['after'] for r in records]
    summary={'structures':len(records),'radius_source':'ASE covalent_radii',
             'minimum_pair_distance':'scale * (covalent_radius_A + covalent_radius_B)',
             'scale':args.scale,'before_total':sum(before),'after_total':sum(after),
             'removed_total':sum(before)-sum(after),
             'removed_fraction':1-sum(after)/sum(before),
             'structures_reduced':sum(r['after']<r['before'] for r in records),
             'structures_with_no_remaining_schemes':sum(n==0 for n in after),
             'reference_structures_failing_same_rule':sum(not r['reference_passes_same_distance_rule'] for r in records),
             'before_median':sorted(before)[len(before)//2],
             'after_median':sorted(after)[len(after)//2],
             'before_max':max(before),'after_max':max(after),
             'before_histogram':histogram(before),'after_histogram':histogram(after),
             'scope':'non-H only; eliminate unavoidable overlaps. Remaining free coordinates are not fitted or energy screened. Different Wyckoff letters remain distinct.',
             'sampling_changed':False}
    (out/'screening_counts.json').write_text(json.dumps({'summary':summary,'structures':records},indent=2),encoding='utf-8')
    lines=['# 保守重叠筛选后的 Wyckoff 组合数量', '',
           f'最小允许距离：{args.scale:.2f} × (r_A + r_B)，r 使用 ASE 共价半径。只检查非 H 原子。',
           '排除周期镜像、轨道内部常量位移、固定特殊位之间不可避免的过短接触。',
           '保留并不等于化学合理；尚未优化自由坐标，也未施加价态/配位或能量筛选。',
           '此筛选单独统计，不改动正在运行的原始随机采样任务。', '',
           '| 结构 ID | 空间群 | 筛选前 | 筛选后 | 排除 | 参考结构通过同一距离规则 |',
           '|---|---:|---:|---:|---:|---|']
    for r in sorted(records,key=lambda x:(-x['after'],x['material_id'])):
        lines.append(f"| {r['material_id']} | {r['spacegroup']} | {r['before']} | {r['after']} | {r['removed']} | {'是' if r['reference_passes_same_distance_rule'] else '否'} |")
    (out/'screening_counts.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
