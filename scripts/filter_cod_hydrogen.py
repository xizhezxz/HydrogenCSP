"""Filter COD CIFs into complete-H and missing-H ordered structures."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pymatgen.io.cif import CifParser


FORMULA_RE = re.compile(
    r"(?im)^[ \t]*_chemical_formula_sum[ \t]+"
    r"(?:'([^']+)'|\"([^\"]+)\"|([^\r\n#]+))"
)
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(?:[ \t]*([0-9]+(?:\.[0-9]+)?))?")
NONE_VALUES = {"", ".", "?", "0", "-1"}


def iter_cifs(root):
    for directory, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".cif"):
                yield Path(directory) / name


def read_formula(path, max_bytes=131072):
    with path.open("rb") as handle:
        text = handle.read(max_bytes).decode("utf-8", errors="replace")
    match = FORMULA_RE.search(text)
    if not match:
        return None
    return next(value.strip() for value in match.groups() if value is not None)


def formula_counts(formula):
    counts = {}
    for symbol, amount in ELEMENT_RE.findall(formula):
        value = float(amount) if amount else 1.0
        counts[symbol] = counts.get(symbol, 0.0) + value
    return counts


def prefilter_file(path_text):
    path = Path(path_text)
    try:
        formula = read_formula(path)
        if not formula:
            return "no_formula_sum", None
        comp = formula_counts(formula)
        if comp.get("H", 0.0) <= 0:
            return "formula_without_h", None
        if sum(comp.values()) > 20.25:
            return "formula_over_20_atoms", None
        return "prefilter_candidates", (path_text, formula, comp)
    except Exception:
        return "prefilter_error", None


def tag_values(block, tag):
    values = block.get(tag, [])
    if isinstance(values, str):
        return [values]
    return list(values)


def classify_candidate(item):
    path_text, formula, formula_comp = item
    path = Path(path_text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(
                path,
                occupancy_tolerance=1.0,
                site_tolerance=1e-4,
                frac_tolerance=1e-4,
                check_cif=False,
            )
            structures = parser.parse_structures(
                primitive=True,
                symmetrized=False,
                check_occu=True,
                on_error="raise",
            )
        if len(structures) != 1:
            return {"status": "rejected", "reason": "not_one_structure", "path": path_text}
        structure = structures[0]
        if len(structure) > 20:
            return {"status": "rejected", "reason": "more_than_20_sites", "path": path_text}
        if not structure.is_ordered:
            return {"status": "rejected", "reason": "disordered_or_partial", "path": path_text}
        for site in structure:
            occupancies = list(site.species.values())
            if len(occupancies) != 1 or abs(float(occupancies[0]) - 1.0) > 1e-8:
                return {"status": "rejected", "reason": "not_full_occupancy", "path": path_text}

        blocks = parser.as_dict()
        block = next(iter(blocks.values()))
        for tag in ("_atom_site_disorder_group", "_atom_site_disorder_assembly"):
            if any(str(value).strip() not in NONE_VALUES for value in tag_values(block, tag)):
                return {"status": "rejected", "reason": "disorder_tag", "path": path_text}
        for value in tag_values(block, "_atom_site_occupancy"):
            try:
                if abs(float(str(value).split("(")[0]) - 1.0) > 1e-8:
                    return {"status": "rejected", "reason": "not_full_occupancy", "path": path_text}
            except ValueError:
                return {"status": "rejected", "reason": "invalid_occupancy", "path": path_text}

        actual = {element.symbol: float(amount) for element, amount in structure.composition.items()}
        formula_non_h = {key for key, value in formula_comp.items() if key != "H" and value > 0}
        actual_non_h = {key for key, value in actual.items() if key != "H" and value > 0}
        if formula_non_h != actual_non_h or not formula_non_h:
            return {"status": "rejected", "reason": "non_h_element_mismatch", "path": path_text}
        ratios = np.asarray([actual[key] / formula_comp[key] for key in sorted(formula_non_h)])
        scale = float(np.median(ratios))
        if not np.allclose(ratios, scale, rtol=0.02, atol=0.02):
            return {"status": "rejected", "reason": "non_h_stoichiometry_mismatch", "path": path_text}

        expected_h = float(formula_comp["H"] * scale)
        actual_h = float(actual.get("H", 0.0))
        delta_h = expected_h - actual_h
        base = {
            "path": path_text,
            "cod_id": path.stem,
            "formula": formula,
            "n_sites": len(structure),
            "expected_h": expected_h,
            "actual_h": actual_h,
            "missing_h": max(0.0, delta_h),
        }
        if actual_h > 0 and abs(delta_h) <= 0.25:
            return {"status": "selected", "class": "h_complete", **base}
        if delta_h > 0.25:
            missing_class = "fully_missing" if actual_h < 0.25 else "partially_missing"
            return {"status": "selected", "class": "h_missing", "missing_class": missing_class, **base}
        return {"status": "rejected", "reason": "hydrogen_excess_or_ambiguous", **base}
    except Exception as exc:
        return {
            "status": "rejected",
            "reason": "parse_error",
            "path": path_text,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_manifest(path, rows):
    fields = [
        "cod_id", "class", "missing_class", "formula", "n_sites",
        "expected_h", "actual_h", "missing_h", "source_path", "output_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cod-dir", type=Path, default=Path("COD"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cod_hydrogen_primitive_le20"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    cod_dir = args.cod_dir.resolve()
    output_dir = args.output_dir.resolve()
    complete_dir = output_dir / "h_complete_le20"
    missing_dir = output_dir / "h_missing_le20"
    complete_dir.mkdir(parents=True, exist_ok=True)
    missing_dir.mkdir(parents=True, exist_ok=True)

    scan_counts = Counter()
    candidates = []
    listing = subprocess.run(
        ["rg", "--files", "-g", "*.cif", str(cod_dir)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    cif_paths = [line for line in listing.stdout.splitlines() if line]
    scan_counts["cif_files"] = len(cif_paths)
    io_workers = max(8, min(32, (os.cpu_count() or 4) * 2))
    print(f"Prefiltering {len(cif_paths):,} CIF headers with {io_workers} I/O workers", flush=True)
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        for index, (status, candidate) in enumerate(
            executor.map(prefilter_file, cif_paths, chunksize=128), start=1
        ):
            scan_counts[status] += 1
            if candidate is not None:
                candidates.append(candidate)
            if index % 50000 == 0:
                print(f"Header scan {index:,}/{len(cif_paths):,}; candidates: {len(candidates):,}", flush=True)

    print(f"Parsing {len(candidates):,} prefiltered candidates with {args.workers} workers", flush=True)
    rejection_counts = Counter()
    selected = {"h_complete": [], "h_missing": []}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(classify_candidate, candidates, chunksize=32), start=1):
            if result["status"] == "selected":
                selected[result["class"]].append(result)
            else:
                rejection_counts[result["reason"]] += 1
            if index % 5000 == 0:
                print(
                    f"Parsed {index:,}/{len(candidates):,}; complete={len(selected['h_complete']):,}, "
                    f"missing={len(selected['h_missing']):,}",
                    flush=True,
                )

    for class_name, rows in selected.items():
        destination = complete_dir if class_name == "h_complete" else missing_dir
        for row in rows:
            source = Path(row["path"])
            target = destination / source.name
            shutil.copy2(source, target)
            row["source_path"] = str(source.relative_to(cod_dir))
            row["output_path"] = str(target.relative_to(output_dir))
        write_manifest(output_dir / f"{class_name}_manifest.csv", rows)

    summary = {
        "definition": {
            "atom_limit": 20,
            "atom_count_basis": "primitive-cell sites after CIF symmetry parsing",
            "ordered": True,
            "full_occupancy": True,
            "hydrogen_expected_from": "_chemical_formula_sum scaled to the parsed non-H composition",
            "h_complete": "actual H > 0 and |expected H - actual H| <= 0.25",
            "h_missing": "expected H - actual H > 0.25",
        },
        "scan": dict(scan_counts),
        "selected": {
            "h_complete": len(selected["h_complete"]),
            "h_missing": len(selected["h_missing"]),
            "h_partially_missing": sum(
                row.get("missing_class") == "partially_missing" for row in selected["h_missing"]
            ),
            "h_fully_missing": sum(
                row.get("missing_class") == "fully_missing" for row in selected["h_missing"]
            ),
        },
        "rejections_after_prefilter": dict(rejection_counts),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
