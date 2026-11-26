"""Relax generated CIFs with EquiformerV3, moving hydrogen atoms only."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from pymatgen.io.ase import AseAtomsAdaptor


MODEL_NAME = "EquiformerV3-OMat24-MPtrj-sAlex"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EQUIFORMER_ROOT = ROOT / "equiformer_v3"
DEFAULT_CHECKPOINT = ROOT / "models" / "omat24-mptrj-salex_gradient.pt"


def import_equiformer(equiformer_root: Path):
    project_root = Path(__file__).resolve().parent.parent
    saved_path = list(sys.path)
    sys.path.insert(0, str(equiformer_root / "src"))
    sys.path.insert(0, str(equiformer_root))
    sys.path[:] = [
        path for path in sys.path
        if path not in ("", ".") and Path(path).resolve() != project_root
    ]
    try:
        sys.modules.pop("src", None)
        sys.modules.pop("fairchem", None)
        import experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
        from src.fairchem.core.common.relaxation.ase_utils import OCPCalculator
    finally:
        sys.path[:] = saved_path
    return OCPCalculator


def save_status(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main(args):
    input_manifest = Path(args.input_manifest).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_cif_dir = output_dir / "relaxed_cifs"
    output_cif_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "relaxation_manifest.csv"
    summary_path = output_dir / "relaxation_summary.json"
    equiformer_root = Path(args.equiformer_root).absolute()
    checkpoint = Path(args.checkpoint).absolute()

    if not equiformer_root.is_dir():
        raise FileNotFoundError(f"Equiformer source not found: {equiformer_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    source = pd.read_csv(input_manifest)
    if args.limit is not None:
        source = source.iloc[:args.limit].copy()
    source["cod_id"] = source["cod_id"].astype(str)
    columns = [
        "cod_id", "input_cif", "output_cif", "initial_energy_ev",
        "relaxed_energy_ev", "converged", "optimizer_steps",
        "max_h_force_ev_per_ang", "elapsed_seconds", "lattice_max_abs_delta",
        "framework_max_abs_cart_delta", "completed", "error",
    ]
    if status_path.exists():
        status = pd.read_csv(status_path, dtype={"cod_id": str})
        if len(status) != len(source):
            raise ValueError("Existing status has a different structure count")
    else:
        status = pd.DataFrame({column: [np.nan] * len(source) for column in columns})
        status["cod_id"] = source["cod_id"].to_numpy()
        status["input_cif"] = source["output_cif"].astype(str).to_numpy()
        status["output_cif"] = [
            str(output_cif_dir / f"{cod_id}.cif") for cod_id in source["cod_id"]
        ]
        status["converged"] = False
        status["completed"] = False
        status["error"] = ""
        status["optimizer_steps"] = -1
        save_status(status, status_path)

    print(f"Loading {MODEL_NAME} from {checkpoint}", flush=True)
    OCPCalculator = import_equiformer(equiformer_root)
    calculator = OCPCalculator(checkpoint_path=str(checkpoint), cpu=False, seed=0)
    adaptor = AseAtomsAdaptor()

    for index in range(len(source)):
        if bool(status.at[index, "completed"]):
            continue
        started = time.time()
        try:
            structure = adaptor.get_atoms(
                __import__("pymatgen.core", fromlist=["Structure"])
                .Structure.from_file(status.at[index, "input_cif"])
            )
            numbers = structure.get_atomic_numbers()
            hydrogen_mask = numbers == 1
            if not hydrogen_mask.any():
                raise ValueError("Structure contains no hydrogen")
            fixed_mask = ~hydrogen_mask
            original_cell = structure.cell.array.copy()
            original_positions = structure.positions.copy()
            structure.calc = calculator
            status.at[index, "initial_energy_ev"] = structure.get_potential_energy()
            structure.set_constraint(FixAtoms(mask=fixed_mask))

            optimizer = BFGS(structure, logfile=None)
            converged = optimizer.run(fmax=args.fmax, steps=args.steps)
            constrained_forces = structure.get_forces(apply_constraint=True)
            status.at[index, "relaxed_energy_ev"] = structure.get_potential_energy()
            status.at[index, "converged"] = bool(converged)
            status.at[index, "optimizer_steps"] = optimizer.nsteps
            status.at[index, "max_h_force_ev_per_ang"] = float(
                np.linalg.norm(constrained_forces[hydrogen_mask], axis=1).max())
            status.at[index, "lattice_max_abs_delta"] = float(
                np.abs(structure.cell.array - original_cell).max())
            status.at[index, "framework_max_abs_cart_delta"] = float(
                np.abs(structure.positions[fixed_mask]
                       - original_positions[fixed_mask]).max(initial=0.0))
            relaxed = adaptor.get_structure(structure)
            Path(status.at[index, "output_cif"]).write_text(
                relaxed.to(fmt="cif"), encoding="utf-8")
        except Exception as exc:
            status.at[index, "error"] = f"{type(exc).__name__}: {exc}"[:1000]
        status.at[index, "elapsed_seconds"] = time.time() - started
        status.at[index, "completed"] = True
        save_status(status, status_path)
        print(json.dumps({
            "index": index,
            "cod_id": str(status.at[index, "cod_id"]),
            "converged": bool(status.at[index, "converged"]),
            "steps": int(status.at[index, "optimizer_steps"]),
            "E_initial": status.at[index, "initial_energy_ev"],
            "E_relaxed": status.at[index, "relaxed_energy_ev"],
            "error": str(status.at[index, "error"]),
        }, default=float), flush=True)

    errors = status["error"].fillna("").astype(str).ne("")
    summary = {
        "model_name": MODEL_NAME,
        "checkpoint": str(checkpoint),
        "input_manifest": str(input_manifest),
        "num_structures": len(status),
        "completed": int(status["completed"].fillna(False).astype(bool).sum()),
        "converged": int(status["converged"].fillna(False).astype(bool).sum()),
        "errors": int(errors.sum()),
        "fmax": args.fmax,
        "max_steps": args.steps,
        "lattice_max_abs_delta": float(
            pd.to_numeric(status["lattice_max_abs_delta"], errors="coerce").max()),
        "framework_max_abs_cart_delta": float(
            pd.to_numeric(status["framework_max_abs_cart_delta"], errors="coerce").max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--equiformer-root", default=DEFAULT_EQUIFORMER_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
