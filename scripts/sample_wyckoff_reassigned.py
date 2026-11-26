"""Sample given space groups with newly drawn non-H Wyckoff allocations.

Reference coordinates are used for SG/setting identification and later evaluation,
not to inherit equivalent atom groups. Each attempt has a fresh orbit assignment.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from torch_geometric.data import Batch

ROOT=Path(__file__).resolve().parents[1]
os.environ['PROJECT_ROOT']=str(ROOT)
sys.path.insert(0,str(ROOT))
from diffcsp.common.wyckoff_reassignment import catalog_from_reference,assign_wyckoff,orbit_residual


def load_sampling_helpers():
    # Legacy utilities override PROJECT_ROOT from a machine-specific .env.
    # Preserve this runner's explicit workspace setting without editing that file.
    import dotenv
    original_loader=dotenv.load_dotenv
    def keep_explicit_environment(*args,**kwargs):
        kwargs['override']=False
        return original_loader(*args,**kwargs)
    dotenv.load_dotenv=keep_explicit_environment
    try:
        from eval_utils import load_model,lattices_to_params_shape
    finally:
        dotenv.load_dotenv=original_loader
    return load_model,lattices_to_params_shape


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/framework_wyckoff_reassigned')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--attempts',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=torch.load(ROOT/'results/framework_fixed/samples_attempts1_to10.pt',map_location='cpu',weights_only=False)
    refs=baseline['input_data_batch'].to_data_list()
    if args.limit:refs=refs[:args.limit]
    with (ROOT/'data/mc3d_H_cifs_le20/mc3d_H_cifs_le20.csv').open(encoding='utf-8',newline='') as handle:
        source_rows=list(csv.DictReader(handle))
    material_ids=[source_rows[int(data.source_row.item())]['material_id'] for data in refs]
    prepared_path=args.output/'wyckoff_inputs.pt'
    if prepared_path.exists():
        prepared=torch.load(prepared_path,map_location='cpu',weights_only=False)
        if prepared['seed']!=args.seed or prepared['attempts']!=args.attempts or prepared['structures']!=len(refs):
            raise ValueError('Existing input assignments use different settings')
        inputs=prepared['inputs'];records=prepared['records']
    else:
        inputs=[[] for _ in range(args.attempts)];records=[];failures=[]
        for j,data in enumerate(refs):
            try:
                catalog=catalog_from_reference(data)
                local=[]
                for a in range(args.attempts):
                    seed=args.seed+1000003*a+j
                    item,assignment=assign_wyckoff(data,catalog,seed)
                    local.append(item)
                    records.append({'structure_index':j,'source_row':int(data.source_row.item()),
                                    'attempt':a+1,'seed':seed,'spacegroup':catalog['spacegroup'],
                                    'hall_number':catalog['hall_number'],
                                    'transformation_matrix':catalog['transformation_matrix'],
                                    'origin_shift':catalog['origin_shift'],
                                    'max_reference_operation_deviation_ang':catalog['max_reference_operation_deviation_ang'],
                                    'allocation':assignment})
                for a,item in enumerate(local):inputs[a].append(item)
            except Exception as exc:
                failures.append({'structure_index':j,'error':str(exc)})
                print('ALLOCATION_ERROR',j,str(exc),flush=True)
            if (j+1)%50==0:print(f'Validated allocations {j+1}/{len(refs)}',flush=True)
        (args.output/'allocation_failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
        if failures:raise RuntimeError(f'{len(failures)} structures failed assignment; no sampling started')
        torch.save({'inputs':inputs,'records':records,'seed':args.seed,
                    'attempts':args.attempts,'structures':len(refs)},prepared_path)
    (args.output/'wyckoff_assignments.json').write_text(json.dumps(records,indent=2,default=lambda x:x.item()),encoding='utf-8')
    print(f'ALL_ALLOCATIONS_VALIDATED {len(refs)} structures x {args.attempts} attempts',flush=True)
    signatures={}
    for row in records:
        signature=tuple(sorted((int(item['atomic_number']),item['wyckoff_letter'],int(item['input_cell_multiplicity'])) for item in row['allocation']))
        signatures.setdefault(row['structure_index'],set()).add(signature)
    validation_summary={'structures':len(refs),'attempts':args.attempts,'validated_allocations':len(records),
                        'failed':0,'structures_with_multiple_sampled_allocations':sum(len(x)>1 for x in signatures.values()),
                        'spacegroup_source':'full reference, including H; no reference orbit allocations inherited'}
    (args.output/'allocation_validation_summary.json').write_text(json.dumps(validation_summary,indent=2),encoding='utf-8')
    if args.validate_only:return
    (args.output/'run_status.json').write_text(json.dumps({'status':'loading_model','structures':len(refs),'attempts':args.attempts},indent=2),encoding='utf-8')
    load_model,lattices_to_params_shape=load_sampling_helpers()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model,_,_=load_model(ROOT/'hydra_jobs/fixlattice_alexmp20',load_data=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval()
    if model.keep_coords:raise ValueError('Model keeps all coordinates fixed')
    generated=[[] for _ in range(args.attempts)];cells=[[] for _ in range(args.attempts)]
    species=[[] for _ in range(args.attempts)];counts=[[] for _ in range(args.attempts)]
    started=time.time()
    for a in range(args.attempts):
        directory=args.output/'generated_cifs'/f'attempt_{a+1:02d}'
        directory.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(refs),args.batch_size):
            end=min(start+args.batch_size,len(refs))
            part=args.output/f'batch_{start//args.batch_size+1:03d}_attempt_{a+1:02d}.pt'
            if part.exists():
                output=torch.load(part,map_location='cpu',weights_only=False)
            else:
                print(f'SAMPLING attempt {a+1}/{args.attempts}, structures {start+1}-{end}/{len(refs)}',flush=True)
                (args.output/'run_status.json').write_text(json.dumps({'status':'sampling','current_attempt':a+1,
                    'current_batch_start':start+1,'current_batch_end':end,'structures':len(refs),
                    'attempts':args.attempts,'elapsed_seconds':time.time()-started},indent=2),encoding='utf-8')
                torch.manual_seed(args.seed+1000003*a+start)
                batch=Batch.from_data_list(inputs[a][start:end]).to(device)
                with torch.inference_mode():output,_=model.sample_spacegroup_hydrogen(batch,step_lr=1e-5)
                output={key:value.detach().cpu() for key,value in output.items() if isinstance(value,torch.Tensor)}
                local_offset=0;residual=0.0
                for j,count in enumerate(output['num_atoms'].tolist(),start):
                    residual=max(residual,orbit_residual(output['frac_coords'][local_offset:local_offset+count].numpy(),inputs[a][j]))
                    local_offset+=count
                if residual>2e-4:raise ValueError(f'Sampling violated allocated orbit maps: {residual}')
                output['orbit_max_fractional_error']=residual
                temporary=part.with_suffix('.tmp.pt');torch.save(output,temporary);temporary.replace(part)
            generated[a].append(output['frac_coords']);cells[a].append(output['lattices'])
            species[a].append(output['atom_types']);counts[a].append(output['num_atoms'])
            offset=0
            for j,count in enumerate(output['num_atoms'].tolist(),start):
                zs=output['atom_types'][offset:offset+count].numpy()
                crystal=Structure(Lattice(output['lattices'][j-start].numpy()),
                                  [Element.from_Z(int(z)) for z in zs],
                                  output['frac_coords'][offset:offset+count].numpy())
                CifWriter(crystal,symprec=None,significant_figures=10).write_file(directory/f'{material_ids[j]}.cif')
                offset+=count
            print(f'SAVED attempt {a+1}, structures {start+1}-{end}',flush=True)
            (args.output/'run_status.json').write_text(json.dumps({'status':'sampling','last_attempt':a+1,
                'last_structure':end,'structures':len(refs),'attempts':args.attempts,
                'elapsed_seconds':time.time()-started},indent=2),encoding='utf-8')
    lattice_tensor=torch.stack([torch.cat(x) for x in cells])
    lengths,angles=lattices_to_params_shape(lattice_tensor)
    payload={'eval_setting':{'sampler':'wyckoff_reassigned','seed':args.seed,'num_evals':args.attempts,
                             'spacegroup_source':'full reference; only SG setting and cell transform retained',
                             'allocation':'random stoichiometry-compatible Wyckoff occupations per attempt'},
             'input_data_batch':Batch.from_data_list(refs),
             'frac_coords':torch.stack([torch.cat(x) for x in generated]),
             'lattices':lattice_tensor,'lengths':lengths,'angles':angles,
             'atom_types':torch.stack([torch.cat(x) for x in species]),
             'num_atoms':torch.stack([torch.cat(x) for x in counts]),'time':time.time()-started}
    torch.save(payload,args.output/'samples_attempts1_to10.pt')
    (args.output/'run_status.json').write_text(json.dumps({'status':'complete','structures':len(refs),
        'attempts':args.attempts,'elapsed_seconds':time.time()-started},indent=2),encoding='utf-8')
    print('SAMPLING_COMPLETE',payload['time'],flush=True)


if __name__=='__main__':main()
