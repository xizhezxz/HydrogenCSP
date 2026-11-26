"""Prepare COD structures for fixed-framework hydrogen reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


def placeholder_coords(count: int) -> np.ndarray:
    """Return deterministic, distinct coordinates; the sampler randomizes them."""
    index = np.arange(1, count + 1, dtype=float)
    return np.column_stack((
        np.mod(index * 0.6180339887498949, 1.0),
        np.mod(index * 0.4142135623730950, 1.0),
        np.mod(index * 0.7320508075688772, 1.0),
    ))


def main(args):
    manifest_path = Path(args.manifest).absolute()
    cif_dir = Path(args.cif_dir).absolute()
    output_dir = Path(args.output_dir).absolute()
    prepared_dir = output_dir / "prepared_cifs"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    excluded = []
    for record in pd.read_csv(manifest_path).to_dict("records"):
        cod_id = str(record["cod_id"])
        source = cif_dir / f"{cod_id}.cif"
        structure = Structure.from_file(source)
        if args.cell_mode == "primitive_reduced":
            structure = structure.get_primitive_structure()
            structure = structure.get_reduced_structure()

        framework_indices = [
            i for i, site in enumerate(structure)
            if site.specie.symbol != "H"
        ]
        framework = Structure.from_sites(
            [structure[i] for i in framework_indices]
        )
        expected_h_float = float(record["expected_h"])
        expected_h = int(round(expected_h_float))
        completed_atoms = len(framework) + expected_h

        reason = None
        if (abs(expected_h_float - expected_h) > args.integer_tolerance
                and not args.allow_noninteger):
            reason = "non_integer_expected_h"
        elif expected_h < 1:
            reason = "no_expected_h"
        elif completed_atoms > args.max_atoms:
            reason = "completed_structure_exceeds_max_atoms"

        common = {
            **record,
            "primitive_framework_atoms": len(framework),
            "expected_h_rounded": expected_h,
            "completed_atoms": completed_atoms,
        }
        if reason is not None:
            excluded.append({**common, "exclusion_reason": reason})
            continue

        completed = framework.copy()
        for coord in placeholder_coords(expected_h):
            completed.append("H", coord, coords_are_cartesian=False)

        prepared_path = prepared_dir / f"{cod_id}.cif"
        cif_text = completed.to(fmt="cif")
        prepared_path.write_text(cif_text, encoding="utf-8")
        rows.append({
            **common,
            "source_cif": str(source),
            "prepared_cif": str(prepared_path),
            "cif": cif_text,
        })

    prepared_csv = output_dir / "hydrogen_completion_input.csv"
    excluded_csv = output_dir / "hydrogen_completion_excluded.csv"
    pd.DataFrame(rows).to_csv(prepared_csv, index=False)
    pd.DataFrame(excluded).to_csv(excluded_csv, index=False)

    summary = {
        "source_manifest": str(manifest_path),
        "source_cif_dir": str(cif_dir),
        "max_atoms": args.max_atoms,
        "integer_tolerance": args.integer_tolerance,
        "cell_mode": args.cell_mode,
        "allow_noninteger": args.allow_noninteger,
        "prepared_structures": len(rows),
        "excluded_structures": len(excluded),
        "excluded_by_reason": (
            pd.Series([row["exclusion_reason"] for row in excluded])
            .value_counts().to_dict()
        ),
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared input: {prepared_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cif_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_atoms", type=int, default=20)
    parser.add_argument("--integer_tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--cell_mode", choices=("original", "primitive_reduced"),
        default="original",
    )
    parser.add_argument("--allow_noninteger", action="store_true")
    main(parser.parse_args())
