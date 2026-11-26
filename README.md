# HydrogenCSP

Code for *A Framework-Constrained Diffusion Model for Predicting Hydrogen-Containing Crystal Structures* by Xizhe Zhang and Yizhou Zhu.

HydrogenCSP is a single diffusion model built on DiffCSP. Given the composition and lattice, it predicts atomic positions using one of two sampling modes:

- **Framework-fixed mode** (`--sampler hydrogen`). When the non-H framework is known, this mode predicts H positions while keeping the framework fixed.
- **Framework-symmetry-constrained mode** (`--sampler spacegroup_hydrogen`). When the non-H coordinates are unknown but the space group is known, this mode predicts all atomic positions. It restricts non-H atoms to their assigned Wyckoff positions and leaves H positions free of those symmetry constraints.

After sampling in either mode, a foundation interatomic potential can refine the H positions while keeping the lattice and non-H framework fixed.

## Installation

Use Python 3.10. Install PyTorch and the matching torch-scatter build for your CPU or GPU, then install the packages in `requirements.txt`. The versions listed are from the development environment. Run commands from the repository root. [PyTorch installation](https://pytorch.org/get-started/locally/) 路 [PyG extension wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```bash
python -m pip install -r requirements.txt
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

## Try assignment selection

The example is a synthetic two-atom CIF with an explicit space group, lattice and composition. You can run the selection step without model weights or a crystallographic database subscription.

```bash
cp examples/manifest.json manifest.json
python scripts/icdd_hpc_pipeline.py select --shard 0
```

Inspect `results/selection/illustrative/ranking.json` for the number of compatible assignments and those retained. The example is small enough to retain them all. For larger sets, the pipeline draws a random sample and ranks assignments using single-start and multi-start PXRD fitting against the simulated reference-framework profile. Use a fresh output directory or checkout when changing the inputs because the pipeline resumes previous work.

## Generate structures

Both sampling modes use the same trained HydrogenCSP checkpoint. Download the [inference weights from the v1.0.0 release](https://github.com/xizhezxz/HydrogenCSP/releases/download/v1.0.0/last.ckpt) into the directory shown below. The matching Hydra configuration is included in `hydra_jobs/fixlattice_alexmp20/hparams.yaml`.

```bash
mkdir -p hydra_jobs/fixlattice_alexmp20/checkpoints
curl --fail --location --retry 3 \
  --output hydra_jobs/fixlattice_alexmp20/checkpoints/last.ckpt \
  https://github.com/xizhezxz/HydrogenCSP/releases/download/v1.0.0/last.ckpt
```

The file is 49,221,147 bytes. Its SHA-256 digest is `99f0087c91643f74739d59586aba566a3999b0c1aa592d6f292e5f26ce28dac7`.

```bash
python scripts/test_hydrogen_sampling.py \
  --model_path hydra_jobs/fixlattice_alexmp20 \
  --input_path examples/structures.csv \
  --sampler hydrogen --num_samples 1 --batch_size 1 \
  --label example --output_dir results/fixed
```

Use `--sampler spacegroup_hydrogen` with a separate output directory to generate both framework and H coordinates under the assignment derived from the input structure. For a CUDA-based search over assignments, run `python scripts/icdd_hpc_pipeline.py generate --shard 0` after the selection example. The CSV sampler writes a PyTorch file containing sampled coordinates and a JSON metrics file. The assignment-search runner writes CIFs under `results/generated_cifs/`.

## Paper workflows

| Task | Entry points and inputs |
| --- | --- |
| H-position prediction | `scripts/test_hydrogen_sampling.py` accepts a CSV with a `cif` column. Set `--num_evals 10` for ten attempts per target. |
| Completion of missing H sites | `scripts/filter_cod_hydrogen.py` identifies records with missing H coordinates. `scripts/prepare_cod_hydrogen_completion.py` prepares the composition, CIF and expected H count. `scripts/generate_cod_hydrogen_completion.py` runs HydrogenCSP in framework-fixed mode. |
| Full structure prediction | `scripts/icdd_hpc_pipeline.py` exposes the `select`, `generate`, `relax` and `collect` stages. Supply your own CIFs through a manifest following `examples/manifest.json`. `scripts/icdd_reference_wyckoff_hpc.py` performs the complementary search using the reference assignment. Run the two searches in separate checkouts because they use the same output paths. |
| PXRD scoring | `diffcsp/common/pxrd_wyckoff.py` implements profile calculation and framework fitting. `scripts/score_icdd_combined.py` reproduces the final comparison from the two searches' prepared candidate records, experimental profiles and scoring masks in the paper's directory layout. |

For H-only relaxation, install the upstream EquiformerV3 implementation and its dependencies. The assignment-search runner expects the source tree at `equiformer_v3/` and the `omat24-mptrj-salex_gradient.pt` checkpoint at `models/`. The standalone CIF relaxation script also accepts `--equiformer-root` and `--checkpoint`. Experimental ICDD peak tables and database records must be supplied separately. [EquiformerV3 code and model instructions](https://github.com/atomicarchitects/equiformer_v3)

## Attribution and license

The diffusion implementation builds on Rui Jiao and colleagues' DiffCSP. The MIT license and the upstream copyright notice are included in `LICENSE`. Please cite the HydrogenCSP manuscript and the original DiffCSP paper when using these methods. [DiffCSP repository](https://github.com/jiaor17/DiffCSP) 路 [DiffCSP paper](https://arxiv.org/abs/2309.04475)
