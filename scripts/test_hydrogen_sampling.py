"""Evaluate hydrogen-coordinate sampling on a CIF CSV dataset."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from eval_utils import lattices_to_params_shape, load_model
from diffcsp.common.data_utils import (
    build_crystal,
    crystal_to_diffusion_data,
    lattice_params_to_matrix_torch,
)
from diffcsp.pl_data.dataset import SpaceGroupConstraintDataset


class CsvCrystalDataset(Dataset):
    """Load the first requested CIF rows without creating a cache file."""

    def __init__(self, path, limit=None, max_atoms=20):
        frame = pd.read_csv(path)
        if limit is not None:
            frame = frame.iloc[:limit]
        self.data = []

        for row_index, row in tqdm(
                frame.iterrows(), total=len(frame), desc="Preprocessing CIFs"):
            crystal = build_crystal(row["cif"], niggli=True, primitive=False)
            item = crystal_to_diffusion_data(crystal)
            atom_types = torch.as_tensor(item["atom_types"], dtype=torch.long)
            if atom_types.numel() > max_atoms:
                raise ValueError(
                    f"CSV row {row_index} has {atom_types.numel()} atoms "
                    f"after preprocessing (limit {max_atoms})"
                )
            if not torch.any(atom_types.eq(1)):
                raise ValueError(f"CSV row {row_index} contains no hydrogen")
            if torch.unique(atom_types).numel() < 2:
                raise ValueError(
                    f"CSV row {row_index} contains fewer than two elements"
                )
            self.data.append(Data(
                frac_coords=torch.as_tensor(
                    item["frac_coords"], dtype=torch.float),
                atom_types=atom_types,
                lengths=torch.as_tensor(
                    item["lengths"], dtype=torch.float).view(1, -1),
                angles=torch.as_tensor(
                    item["angles"], dtype=torch.float).view(1, -1),
                num_atoms=int(item["num_atoms"]),
                num_nodes=int(item["num_atoms"]),
                source_row=torch.tensor([row_index], dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def periodic_delta(first, second):
    return torch.remainder(first - second + 0.5, 1.0) - 0.5


def symmetry_errors(coords, batch, atom_mask):
    """Return component errors against the stored original-cell operations."""
    frames = torch.arange(coords.size(0), device=coords.device)[atom_mask]
    anchors = batch.anchor_index[atom_mask]
    unique_anchors = torch.unique(anchors)
    anchor_coords = coords[unique_anchors]
    anchor_ops = batch.ops[unique_anchors]

    shifts = torch.cartesian_prod(*[
        torch.tensor(
            [0.0, -1.0, 1.0],
            device=coords.device,
            dtype=coords.dtype,
        )
        for _ in range(3)
    ])
    candidates = anchor_coords[:, None, :] + shifts[None, :, :]
    homogeneous_candidates = torch.cat([
        candidates,
        torch.ones(
            (*candidates.shape[:2], 1),
            device=coords.device,
            dtype=coords.dtype,
        ),
    ], dim=-1)
    projected = torch.einsum(
        "aij,asj->asi", anchor_ops, homogeneous_candidates
    )[:, :, :3]
    errors = periodic_delta(projected, anchor_coords[:, None, :])
    best = errors.square().sum(dim=-1).argmin(dim=-1)
    representatives = projected[
        torch.arange(unique_anchors.numel(), device=coords.device), best]
    representative_by_node = torch.zeros_like(coords)
    representative_by_node[unique_anchors] = representatives

    homogeneous = torch.cat([
        representative_by_node[anchors],
        torch.ones((frames.numel(), 1), device=coords.device),
    ], dim=-1)
    expected = torch.bmm(
        batch.ops[frames], homogeneous.unsqueeze(-1))[:, :3, 0]
    return periodic_delta(coords[frames], expected)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Keep the user-visible workspace path. Path.resolve() follows this
    # workspace's directory junction to an older checkout where newly created
    # data files may not exist.
    input_path = Path(args.input_path).absolute()
    model_path = Path(args.model_path).absolute()
    output_dir = Path(args.output_dir).absolute() if args.output_dir else model_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_diff_{args.label}.pt"
    metrics_path = output_dir / f"hydrogen_metrics_{args.label}.json"
    partial_path = output_path.with_suffix(".partial.pt")

    dataset = CsvCrystalDataset(
        input_path, limit=args.num_samples, max_atoms=args.max_atoms)
    if args.sampler == "spacegroup_hydrogen":
        dataset = SpaceGroupConstraintDataset(
            dataset, tolerance=args.symprec, preserve_cell=True)

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    model, _, _ = load_model(model_path, load_data=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sample_fn = (
        model.sample
        if args.sampler == "sample"
        else getattr(model, f"sample_{args.sampler}")
    )

    generated_coords = [[] for _ in range(args.num_evals)]
    generated_lattices = [[] for _ in range(args.num_evals)]
    generated_num_atoms = [[] for _ in range(args.num_evals)]
    generated_atom_types = [[] for _ in range(args.num_evals)]
    input_data = []
    per_structure_h_rms = []
    matched_h_squared_distances = []
    framework_squared_displacements = []
    framework_max_periodic_delta = 0.0
    lattice_max_delta = 0.0
    non_h_symmetry_components = []
    h_symmetry_components = []
    total_hydrogen = 0
    start_batch = 0
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial["num_evals"] != args.num_evals:
            raise ValueError("Partial checkpoint has a different num_evals")
        start_batch = int(partial["completed_batches"])
        for eval_index in range(args.num_evals):
            generated_coords[eval_index].append(partial["generated_coords"][eval_index])
            generated_lattices[eval_index].append(partial["generated_lattices"][eval_index])
            generated_num_atoms[eval_index].append(partial["generated_num_atoms"][eval_index])
            generated_atom_types[eval_index].append(partial["generated_atom_types"][eval_index])
        input_data.extend(partial["input_data_batch"].to_data_list())
        per_structure_h_rms = partial["per_structure_h_rms"]
        matched_h_squared_distances = partial["matched_h_squared_distances"]
        framework_squared_displacements = partial["framework_squared_displacements"]
        framework_max_periodic_delta = partial["framework_max_periodic_delta"]
        lattice_max_delta = partial["lattice_max_delta"]
        non_h_symmetry_components = partial["non_h_symmetry_components"]
        h_symmetry_components = partial["h_symmetry_components"]
        total_hydrogen = partial["total_hydrogen"]
        print(f"Resuming after {start_batch}/{len(loader)} completed batches", flush=True)
    start = time.time()

    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue
        input_data.extend(batch.to_data_list())
        batch = batch.to(device)
        input_coords = batch.frac_coords.clone()
        input_lattice = lattice_params_to_matrix_torch(
            batch.lengths, batch.angles)

        for eval_index in range(args.num_evals):
            print(
                f"Sampling batch {batch_index + 1}/{len(loader)}, "
                f"attempt {eval_index + 1}/{args.num_evals} "
                f"with {args.sampler}", flush=True)
            output, _ = sample_fn(batch, step_lr=args.step_lr)
            coords = output["frac_coords"]
            lattices = output["lattices"]
            lattice_max_delta = max(
                lattice_max_delta,
                float((lattices - input_lattice).abs().max().item()),
            )

            if args.sampler == "spacegroup_hydrogen":
                non_h_symmetry_components.append(symmetry_errors(
                    coords, batch, batch.atom_types.ne(1)).detach().cpu())
                h_symmetry_components.append(symmetry_errors(
                    coords, batch, batch.atom_types.eq(1)).detach().cpu())

            offset = 0
            for count, lattice_matrix in zip(
                    batch.num_atoms.detach().cpu().tolist(), lattices):
                predicted = coords[offset:offset + count]
                target = input_coords[offset:offset + count]
                atom_types = batch.atom_types[offset:offset + count]
                hydrogen_mask = atom_types.eq(1)
                framework_mask = ~hydrogen_mask
                if eval_index == 0:
                    total_hydrogen += int(hydrogen_mask.sum().item())

                framework_delta = periodic_delta(
                    predicted[framework_mask], target[framework_mask])
                if framework_delta.numel():
                    framework_max_periodic_delta = max(
                        framework_max_periodic_delta,
                        float(framework_delta.abs().max().item()),
                    )
                    framework_squared_displacements.extend(
                        framework_delta.square().sum(dim=-1).detach().cpu().tolist()
                    )

                lattice = Lattice(lattice_matrix.detach().cpu().numpy())
                distances = lattice.get_all_distances(
                    target[hydrogen_mask].detach().cpu().numpy(),
                    predicted[hydrogen_mask].detach().cpu().numpy(),
                )
                rows, columns = linear_sum_assignment(distances ** 2)
                squared = distances[rows, columns] ** 2
                matched_h_squared_distances.extend(squared.tolist())
                per_structure_h_rms.append(float(np.sqrt(np.mean(squared))))
                offset += count

            generated_coords[eval_index].append(coords.detach().cpu())
            generated_lattices[eval_index].append(lattices.detach().cpu())
            generated_num_atoms[eval_index].append(
                output["num_atoms"].detach().cpu())
            generated_atom_types[eval_index].append(
                output["atom_types"].detach().cpu())

        torch.save({
            "num_evals": args.num_evals,
            "completed_batches": batch_index + 1,
            "generated_coords": [torch.cat(parts) for parts in generated_coords],
            "generated_lattices": [torch.cat(parts) for parts in generated_lattices],
            "generated_num_atoms": [torch.cat(parts) for parts in generated_num_atoms],
            "generated_atom_types": [torch.cat(parts) for parts in generated_atom_types],
            "input_data_batch": Batch.from_data_list(input_data),
            "per_structure_h_rms": per_structure_h_rms,
            "matched_h_squared_distances": matched_h_squared_distances,
            "framework_squared_displacements": framework_squared_displacements,
            "framework_max_periodic_delta": framework_max_periodic_delta,
            "lattice_max_delta": lattice_max_delta,
            "non_h_symmetry_components": non_h_symmetry_components,
            "h_symmetry_components": h_symmetry_components,
            "total_hydrogen": total_hydrogen,
        }, partial_path)

    frac_coords = torch.stack([torch.cat(parts) for parts in generated_coords])
    lattices = torch.stack([torch.cat(parts) for parts in generated_lattices])
    num_atoms = torch.stack([torch.cat(parts) for parts in generated_num_atoms])
    atom_types = torch.stack([torch.cat(parts) for parts in generated_atom_types])
    lengths, angles = lattices_to_params_shape(lattices)
    elapsed = time.time() - start

    metrics = {
        "sampler": args.sampler,
        "input_path": str(input_path),
        "num_structures": len(dataset),
        "num_evals": args.num_evals,
        "num_hydrogen": total_hydrogen,
        "seed": args.seed,
        "step_lr": args.step_lr,
        "elapsed_seconds": elapsed,
        "lattice_max_abs_delta": lattice_max_delta,
        "framework_max_periodic_frac_delta": framework_max_periodic_delta,
        "framework_periodic_frac_rms": float(np.sqrt(np.mean(
            framework_squared_displacements))),
        "hydrogen_assignment_rms_angstrom_mean": float(np.mean(
            per_structure_h_rms)),
        "hydrogen_assignment_rms_angstrom_median": float(np.median(
            per_structure_h_rms)),
        "hydrogen_assignment_rms_angstrom_atom_weighted": float(np.sqrt(
            np.mean(matched_h_squared_distances))),
    }
    if non_h_symmetry_components:
        non_h_errors = torch.cat(non_h_symmetry_components)
        h_errors = torch.cat(h_symmetry_components)
        metrics.update({
            "non_h_symmetry_max_periodic_frac_error": float(
                non_h_errors.abs().max().item()),
            "non_h_symmetry_periodic_frac_rms": float(
                non_h_errors.square().mean().sqrt().item()),
            "h_unconstrained_symmetry_max_periodic_frac_error": float(
                h_errors.abs().max().item()),
            "h_unconstrained_symmetry_periodic_frac_rms": float(
                h_errors.square().mean().sqrt().item()),
        })

    torch.save({
        "eval_setting": vars(args),
        "input_data_batch": Batch.from_data_list(input_data),
        "frac_coords": frac_coords,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
        "lattices": lattices,
        "lengths": lengths,
        "angles": angles,
        "time": elapsed,
    }, output_path)
    partial_path.unlink(missing_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved samples to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--sampler",
        choices=("sample", "hydrogen", "spacegroup_hydrogen"),
        required=True,
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_evals", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--max_atoms", type=int, default=20)
    parser.add_argument("--step_lr", type=float, default=1e-5)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True)
    main(parser.parse_args())
