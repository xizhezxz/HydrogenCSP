# HydrogenCSP

Code for *Framework-Constrained Diffusion Models for Predicting Hydrogen-Containing Crystal Structures* by Xizhe Zhang and Yizhou Zhu.

HydrogenCSP extends DiffCSP to predict atomic coordinates at a supplied composition and lattice. With a known non-H framework, the framework-fixed sampler generates H coordinates while keeping the framework fixed. With a specified space group and Wyckoff assignment, the symmetry-constrained sampler generates all coordinates, applying the symmetry constraints only to non-H atoms. Both workflows support subsequent H-only optimization.

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

Sampling requires the trained fixed-lattice DiffCSP checkpoint and its matching Hydra configuration. The configuration is in `hydra_jobs/fixlattice_alexmp20/hparams.yaml`. Supply the checkpoint as `hydra_jobs/fixlattice_alexmp20/checkpoints/last.ckpt`. This source release does not distribute the trained weights.

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
| Completion of missing H sites | `scripts/filter_cod_hydrogen.py` identifies records with missing H coordinates. `scripts/prepare_cod_hydrogen_completion.py` prepares the composition, CIF and expected H count. `scripts/generate_cod_hydrogen_completion.py` runs the framework-fixed sampler. |
| Full structure prediction | `scripts/icdd_hpc_pipeline.py` exposes the `select`, `generate`, `relax` and `collect` stages. Supply your own CIFs through a manifest following `examples/manifest.json`. `scripts/icdd_reference_wyckoff_hpc.py` performs the complementary search using the reference assignment. Run the two searches in separate checkouts because they use the same output paths. |
| PXRD scoring | `diffcsp/common/pxrd_wyckoff.py` implements profile calculation and framework fitting. `scripts/score_icdd_combined.py` reproduces the final comparison from the two searches' prepared candidate records, experimental profiles and scoring masks in the paper's directory layout. |

For H-only relaxation, install the upstream EquiformerV3 implementation and its dependencies. The assignment-search runner expects the source tree at `equiformer_v3/` and the `omat24-mptrj-salex_gradient.pt` checkpoint at `models/`. The standalone CIF relaxation script also accepts `--equiformer-root` and `--checkpoint`. Experimental ICDD peak tables and database records must be supplied separately. [EquiformerV3 code and model instructions](https://github.com/atomicarchitects/equiformer_v3)

## Attribution and license

The diffusion implementation builds on Rui Jiao and colleagues' DiffCSP. The MIT license and the upstream copyright notice are included in `LICENSE`. Please cite the HydrogenCSP manuscript and the original DiffCSP paper when using these methods. [DiffCSP repository](https://github.com/jiaor17/DiffCSP) 路 [DiffCSP paper](https://arxiv.org/abs/2309.04475)
