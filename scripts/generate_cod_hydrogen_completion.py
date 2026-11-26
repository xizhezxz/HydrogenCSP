"""Generate H coordinates while keeping the primitive cell and framework fixed."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Lattice, Structure
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import lattices_to_params_shape, load_model
from diffcsp.common.data_utils import (
    build_crystal,
    crystal_to_diffusion_data,
    lattice_params_to_matrix_torch,
)


class CompletionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_atoms: int):
        self.data = []
        for row_index, row in tqdm(
                frame.iterrows(), total=len(frame), desc="Preprocessing CIFs"):
            crystal = build_crystal(row["cif"], niggli=False, primitive=False)
            item = crystal_to_diffusion_data(crystal)
            atom_types = torch.as_tensor(item["atom_types"], dtype=torch.long)
            if atom_types.numel() > max_atoms:
                raise ValueError(
                    f"CSV row {row_index} has {atom_types.numel()} atoms "
                    f"(limit {max_atoms})"
                )
            if not torch.any(atom_types.eq(1)):
                raise ValueError(f"CSV row {row_index} contains no hydrogen")
            self.data.append(Data(
                frac_coords=torch.as_tensor(item["frac_coords"], dtype=torch.float),
                atom_types=atom_types,
                lengths=torch.as_tensor(item["lengths"], dtype=torch.float).view(1, -1),
                angles=torch.as_tensor(item["angles"], dtype=torch.float).view(1, -1),
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


def save_cifs(frame, coords, lattices, atom_types, num_atoms, output_dir, attempt):
    attempt_dir = output_dir / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    for index, count in enumerate(num_atoms.tolist()):
        cod_id = str(frame.iloc[index]["cod_id"])
        species = [Element.from_Z(int(z)) for z in atom_types[offset:offset + count]]
        structure = Structure(
            Lattice(lattices[index].numpy()),
            species,
            coords[offset:offset + count].numpy(),
            coords_are_cartesian=False,
        )
        path = attempt_dir / f"{cod_id}.cif"
        path.write_text(structure.to(fmt="cif"), encoding="utf-8")
        records.append({
            "cod_id": cod_id,
            "attempt": attempt,
            "output_cif": str(path),
            "num_atoms": count,
            "num_h": int(sum(int(z) == 1 for z in atom_types[offset:offset + count])),
        })
        offset += count
    return records


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    input_path = Path(args.input_path).absolute()
    model_path = Path(args.model_path).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    if args.num_structures is not None:
        frame = frame.iloc[:args.num_structures].reset_index(drop=True)
    dataset = CompletionDataset(frame, args.max_atoms)
    loader = DataLoader(
        dataset, batch_size=min(args.batch_size, len(dataset)),
        shuffle=False, num_workers=0,
    )
    model, _, _ = load_model(model_path, load_data=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_coords = [[] for _ in range(args.num_evals)]
    all_lattices = [[] for _ in range(args.num_evals)]
    all_num_atoms = [[] for _ in range(args.num_evals)]
    all_atom_types = [[] for _ in range(args.num_evals)]
    input_items = []
    framework_max_delta = 0.0
    lattice_max_delta = 0.0
    cif_records = []
    batch_row_offset = 0
    started = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            input_items.extend(batch.to_data_list())
            batch = batch.to(device)
            fixed_coords = batch.frac_coords.clone()
            fixed_lattice = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            framework_mask = batch.atom_types.ne(1)
            for attempt in range(args.num_evals):
                print(
                    f"Batch {batch_index + 1}/{len(loader)}, "
                    f"attempt {attempt + 1}/{args.num_evals}", flush=True)
                output, _ = model.sample_hydrogen(batch, step_lr=args.step_lr)
                coords = output["frac_coords"]
                lattices = output["lattices"]
                framework_max_delta = max(
                    framework_max_delta,
                    float(periodic_delta(
                        coords[framework_mask], fixed_coords[framework_mask]
                    ).abs().max().item()),
                )
                lattice_max_delta = max(
                    lattice_max_delta,
                    float((lattices - fixed_lattice).abs().max().item()),
                )
                all_coords[attempt].append(coords.cpu())
                all_lattices[attempt].append(lattices.cpu())
                all_num_atoms[attempt].append(output["num_atoms"].cpu())
                all_atom_types[attempt].append(output["atom_types"].cpu())

                batch_size_actual = int(output["num_atoms"].numel())
                batch_frame = frame.iloc[
                    batch_row_offset:batch_row_offset + batch_size_actual
                ].reset_index(drop=True)
                batch_records = save_cifs(
                    batch_frame,
                    coords.detach().cpu(),
                    lattices.detach().cpu(),
                    output["atom_types"].detach().cpu(),
                    output["num_atoms"].detach().cpu(),
                    output_dir,
                    attempt + 1,
                )
                for record in batch_records:
                    record["batch"] = batch_index + 1
                cif_records.extend(batch_records)
                pd.DataFrame(cif_records).to_csv(
                    output_dir / "generated_cifs_manifest.csv", index=False)
                torch.save({
                    "batch_index": batch_index + 1,
                    "attempt": attempt + 1,
                    "cod_ids": batch_frame["cod_id"].astype(str).tolist(),
                    "frac_coords": coords.detach().cpu(),
                    "lattices": lattices.detach().cpu(),
                    "num_atoms": output["num_atoms"].detach().cpu(),
                    "atom_types": output["atom_types"].detach().cpu(),
                }, output_dir / (
                    f"batch_{batch_index + 1:03d}_attempt_{attempt + 1:02d}.pt"
                ))
                print(
                    f"Saved batch {batch_index + 1}, attempt {attempt + 1}",
                    flush=True,
                )
            batch_row_offset += int(batch.num_graphs)

    frac_coords = torch.stack([torch.cat(parts) for parts in all_coords])
    lattices = torch.stack([torch.cat(parts) for parts in all_lattices])
    num_atoms = torch.stack([torch.cat(parts) for parts in all_num_atoms])
    atom_types = torch.stack([torch.cat(parts) for parts in all_atom_types])
    lengths, angles = lattices_to_params_shape(lattices)
    result_path = output_dir / "hydrogen_completion.pt"
    torch.save({
        "eval_setting": vars(args),
        "input_data_batch": Batch.from_data_list(input_items),
        "cod_ids": frame["cod_id"].astype(str).tolist(),
        "frac_coords": frac_coords,
        "lattices": lattices,
        "lengths": lengths,
        "angles": angles,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
    }, result_path)


    metrics = {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "device": str(device),
        "num_structures": len(dataset),
        "num_evals": args.num_evals,
        "seed": args.seed,
        "step_lr": args.step_lr,
        "elapsed_seconds": time.time() - started,
        "framework_max_periodic_frac_delta": framework_max_delta,
        "lattice_max_abs_delta": lattice_max_delta,
        "result_path": str(result_path),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_structures", type=int)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_atoms", type=int, default=20)
    parser.add_argument("--step_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
