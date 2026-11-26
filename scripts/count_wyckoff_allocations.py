"""Exact discrete Wyckoff allocation counts, not chemical/energy feasibility."""
from collections import Counter, defaultdict
import json
from math import prod
from pathlib import Path

import numpy as np
from pyxtal.symmetry import Group
import torch


def count_allocations(counts, positions):
    """Unordered orbit multisets; fixed positions exclusive across elements."""
    targets=tuple(counts)
    largest=max(targets,default=0)
    # One unbounded coin type per free Wyckoff letter, not per orbit ordering.
    free=[0]*(largest+1);free[0]=1
    for p in positions:
        if p['dof']>0:
            m=p['multiplicity']
            for n in range(m,largest+1):free[n]+=free[n-m]
    # Each zero-DOF letter can be empty, or occupied by one element (once).
    fixed={tuple(0 for _ in targets):1}
    for p in positions:
        if p['dof']!=0:continue
        m=p['multiplicity'];updated=defaultdict(int)
        for state,ways in fixed.items():
            updated[state]+=ways
            for z,n in enumerate(targets):
                if state[z]+m<=n:
                    occupied=list(state);occupied[z]+=m
                    updated[tuple(occupied)]+=ways
        fixed=dict(updated)
    return sum(ways*prod(free[n-occupied] for n,occupied in zip(targets,state))
               for state,ways in fixed.items())


def verify_counter():
    # Two labelled fixed sites for two elements: AB or BA, not four.
    fixed=[{'multiplicity':1,'dof':0}]*2
    assert count_allocations([1,1],fixed)==2
    assert count_allocations([3],fixed)==0
    # Two free letters with identical multiplicity: 2a, a+b, or 2b.
    free=[{'multiplicity':1,'dof':1}]*2
    assert count_allocations([2],free)==3
    assert count_allocations([1,1],free)==4
    assert count_allocations([2],[{'multiplicity':2,'dof':1},
                                  {'multiplicity':1,'dof':0}])==1


def main():
    verify_counter()
    folder=Path(__file__).resolve().parents[1]/'results/framework_wyckoff_reassigned'
    prepared=torch.load(folder/'wyckoff_inputs.pt',map_location='cpu',weights_only=False)
    refs=prepared['inputs'][0]
    allocations=prepared['records']
    first={r['structure_index']:r for r in allocations if r['attempt']==1}
    observed=defaultdict(set)
    for r in allocations:
        observed[r['structure_index']].add(tuple(sorted((int(p['atomic_number']),p['wyckoff_letter'],
                                                      int(p['input_cell_multiplicity'])) for p in r['allocation'])))
    source=json.loads((folder.parent/'framework_fixed/generated_cifs/cif_export_manifest.json').read_text(encoding='utf-8'))
    ids={r['structure_index_zero_based']:r['material_id'] for r in source['records'] if r['attempt']==1}
    output=[];cache={}
    for j,data in enumerate(refs):
        record=first[j]
        ratio=abs(np.linalg.det(record['transformation_matrix']))
        key=(record['hall_number'],round(ratio,8))
        if key not in cache:
            group=Group(record['hall_number'],use_hall=True)
            positions=[]
            for wp in group.Wyckoff_positions:
                m=wp.multiplicity*ratio
                if np.isclose(m,round(m),atol=1e-5):
                    positions.append({'letter':wp.letter,'multiplicity':int(round(m)),
                                      'standard_multiplicity':int(wp.multiplicity),'dof':int(wp.get_dof())})
            cache[key]=positions
        counts=Counter(int(z) for z in data.atom_types.tolist() if z!=1)
        count=count_allocations([counts[z] for z in sorted(counts)],cache[key])
        if count<len(observed[j]) or count<1:
            raise ValueError(f'Exact count inconsistent with sampled schemes: {j}')
        output.append({'structure_index':j,'material_id':ids[j],'spacegroup':record['spacegroup'],
                       'hall_number':record['hall_number'],'num_atoms':int(data.num_atoms),
                       'non_h_counts_by_atomic_number':dict(sorted(counts.items())),
                       'feasible_allocation_count':count,'distinct_allocations_in_10_draws':len(observed[j]),
                       'sampled_allocation_coverage_fraction':len(observed[j])/count,
                       'wyckoff_positions_in_input_cell':cache[key]})
        if (j+1)%100==0:print(f'Counted {j+1}/{len(refs)}',flush=True)
    amounts=[x['feasible_allocation_count'] for x in output]
    buckets={'1':0,'2-10':0,'11-100':0,'101-1000':0,'1001-10000':0,'>10000':0}
    for count in amounts:
        bucket='1' if count==1 else '2-10' if count<=10 else '11-100' if count<=100 else '101-1000' if count<=1000 else '1001-10000' if count<=10000 else '>10000'
        buckets[bucket]+=1
    ordered=sorted(amounts)
    summary={'structures':len(output),'total_allocations_across_structures':sum(amounts),
             'minimum':min(amounts),'median':ordered[len(ordered)//2],'maximum':max(amounts),
             'histogram':buckets,'structures_fully_covered_by_10_draws':sum(x['feasible_allocation_count']==x['distinct_allocations_in_10_draws'] for x in output),
             'rules':['exact non-H element counts','input-cell Wyckoff multiplicities',
                      'zero-DOF letters used once across all elements','free letters reusable',
                      'orbit order and atom permutations not counted','different letters kept distinct; no affine-normalizer equivalence reduction',
                      'no geometric-distance, valence, or energy filtering; upper bound on chemically reasonable schemes',
                      'continuous coordinates excluded from discrete counts']}
    (folder/'wyckoff_allocation_counts.json').write_text(json.dumps({'summary':summary,'structures':output},indent=2),encoding='utf-8')
    lines=['# 非 H Wyckoff 占位组合数量', '',
           '这是满足空间群、计量和固定特殊位互斥的离散组合数，不是经过化学、距离或能量验证的方案数。',
           '有自由度的同类轨道可重复；轨道顺序和同元素原子排列不重复计数，连续坐标不计数。',
           '不同 Wyckoff 字母保持区分，未进一步合并整体原点变换等导致的结构等价方案。', '',
           f"结构数：{len(output)}；方案数最小/中位/最大：{summary['minimum']} / {summary['median']} / {summary['maximum']}。", '',
           '| 结构 ID | 空间群 | 非 H 原子数 | 可行占位组合数 | 10 次抽样中不同组合数 |',
           '|---|---:|---:|---:|---:|']
    for row in sorted(output,key=lambda x:(-x['feasible_allocation_count'],x['material_id'])):
        lines.append(f"| {row['material_id']} | {row['spacegroup']} | {sum(row['non_h_counts_by_atomic_number'].values())} | {row['feasible_allocation_count']} | {row['distinct_allocations_in_10_draws']} |")
    (folder/'wyckoff_allocation_counts.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)
    for row in sorted(output,key=lambda x:-x['feasible_allocation_count'])[:10]:
        print(row['material_id'],row['spacegroup'],row['feasible_allocation_count'],row['distinct_allocations_in_10_draws'],flush=True)


if __name__=='__main__':main()
