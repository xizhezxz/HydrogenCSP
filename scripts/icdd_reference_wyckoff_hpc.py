"""Use the original non-H Wyckoff allocation; independently sample H."""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import icdd_hpc_pipeline as pipeline
from diffcsp.common.wyckoff_enumeration import instantiate_allocation

ROOT = Path(__file__).resolve().parents[1]

def original_allocation(data, catalog, tolerance=.01):
    from pymatgen.core import Lattice
    lattice = Lattice.from_parameters(*data.lengths.reshape(-1).tolist(), *data.angles.reshape(-1).tolist())
    xyz = data.frac_coords.numpy().astype(float)
    zs = data.atom_types.numpy()
    remaining = set(np.flatnonzero(zs != 1).tolist())
    allocation, audit = [], []
    while remaining:
        i = min(remaining)
        images = np.einsum('nij,j->ni', catalog['symmetry_ops'], np.r_[xyz[i], 1])[:, :3]
        same = np.flatnonzero(zs == zs[i])
        distances = lattice.get_all_distances(images, xyz[same])
        if distances.min(axis=1).max() > tolerance:
            raise ValueError('Original framework does not close under declared symmetry')
        slots = sorted(set(same[distances.argmin(axis=1)].tolist()))
        if not set(slots) <= remaining:
            raise ValueError('Overlapping reference orbits')
        matches = []
        for k, wp in enumerate(catalog['positions']):
            if wp['multiplicity'] != len(slots):
                continue
            for anchor in slots:
                for shift in product((-1,0,1), repeat=3):
                    expanded = np.einsum('nij,j->ni', wp['ops'], np.r_[xyz[anchor]+shift, 1])[:, :3]
                    d = lattice.get_all_distances(expanded, xyz[slots])
                    a, b = linear_sum_assignment(d)
                    residual = float(d[a, b].max())
                    if residual <= tolerance:
                        matches.append((residual, k))
                        break
                if any(match_k == k for _, match_k in matches):
                    break
        choices = set(k for _, k in matches)
        if len(choices) != 1:
            raise ValueError(f'Reference orbit {slots}: expected unique Wyckoff match, got {len(choices)}')
        residual, k = min(matches)
        allocation.append((int(zs[i]), k))
        audit.append(dict(atomic_number=int(zs[i]), reference_site_indices=slots,
                          wyckoff_letter=catalog['positions'][k]['letter'],
                          multiplicity=len(slots), match_residual_A=residual))
        remaining.difference_update(slots)
    return allocation, audit

def prepare(rows, attempts):
    settings = ROOT/'run_settings.json'
    requested = dict(attempts=attempts, protocol='original-framework-wyckoff-v2-zero-dof-once')
    if settings.exists() and json.loads(settings.read_text()) != requested:
        if any((ROOT/'results/generation_records').rglob('*.json')):
            raise ValueError('Attempts changed after generation; use a fresh output folder')
    pipeline.save(settings, requested)
    failures = []
    for row in rows:
        directory = ROOT/'results/selection'/row['card']
        try:
            data, catalog, meta = pipeline.prepare_one(ROOT/row['input'], row['index'])
            allocation, audit = original_allocation(data, catalog)
            framework_dof = sum(int(catalog['positions'][k]['dof']) for _, k in allocation)
            effective_attempts = 1 if framework_dof == 0 else attempts
            inputs, records = [], []
            for attempt in range(1, effective_attempts+1):
                seed = 20261026 + row['index']*100003 + attempt
                item, assignment = instantiate_allocation(data, catalog, allocation, seed)
                inputs.append(item)
                records.append(dict(scheme_index=attempt, attempt=attempt,
                                    allocation=assignment, reference_allocation=audit,
                                    spacegroup=catalog['spacegroup'], seed=seed))
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(dict(inputs=inputs, records=records), directory/'selected_inputs.pt')
            pipeline.save(directory/'input_metadata.json', dict(**meta, attempts=effective_attempts,
                          requested_attempts=attempts, framework_degrees_of_freedom=framework_dof, reference_allocation=audit))
            print('PREPARED', row['card'], effective_attempts, flush=True)
        except Exception as exc:
            failures.append(dict(card=row['card'], error=str(exc)))
    pipeline.save(ROOT/'preparation_errors.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} reference assignments failed; inspect preparation_errors.json')

def collect(rows):
    records = [json.loads(p.read_text()) for p in (ROOT/'results/relax_records').rglob('*.json')]
    summary = dict(materials=len(rows), generated=len(list((ROOT/'results/generated_cifs').rglob('*.cif'))),
                   relaxed=sum(r.get('status')=='ok' for r in records),
                   converged=sum(bool(r.get('converged')) for r in records),
                   errors=sum(r.get('status')=='error' for r in records))
    pipeline.save(ROOT/'results/summary.json', summary)
    pipeline.save(ROOT/'results/all_relax_results.json', records)
    print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','generate','relax','collect'])
    parser.add_argument('--attempts', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=300)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    args.scheme_limit = None
    if args.batch_size < 1 or args.attempts < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid batch size, attempt count or shard')
    rows = json.loads((ROOT/'manifest.json').read_text())
    if args.stage == 'prepare': prepare(rows, args.attempts)
    elif args.stage == 'collect': collect(rows)
    else:
        rows = [r for r in rows if r['shard']==args.shard]
        if args.stage == 'generate':
            requested = dict(attempts=args.attempts, batch_size=args.batch_size, shards=args.shards,
                             protocol='original-framework-wyckoff-v2-zero-dof-once')
            lock = ROOT/'results'/f'generation_settings_shard_{args.shard}.json'
            if lock.exists() and json.loads(lock.read_text()) != requested:
                raise ValueError('Generation settings changed; use a fresh output folder')
            for row in rows:
                meta = json.loads((ROOT/'results/selection'/row['card']/'input_metadata.json').read_text())
                expected = 1 if meta['framework_degrees_of_freedom'] == 0 else args.attempts
                if meta['attempts'] != expected or meta['requested_attempts'] != args.attempts:
                    raise ValueError('Prepared attempt count differs from requested attempts')
            pipeline.save(lock, requested)
        if args.stage == 'generate': pipeline.generate(args, rows)
        else: pipeline.relax(args, rows)
