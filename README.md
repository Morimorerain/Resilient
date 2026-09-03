# Resilient

[简体中文](README.zh-CN.md) | **English**

Resilient is a reproducibility-first research codebase built on [FastWAM](https://github.com/yuantianyuan01/FastWAM) for LIBERO evaluation and follow-up research. The FastWAM baseline is integrated at the repository root because its Hydra configs and evaluation entry points use repository-relative paths.

## Current status

- FastWAM upstream is pinned to commit `7faa71108368fbb3b6885649f112af607427a2d4`.
- The baseline target is `libero_uncond_2cam224.pt` with its released dataset statistics.
- Reproduction uses the original checkpoint setting `EVALUATION.sigma_shift=5.0`.
- FastWAM core behavior is unchanged. Project-specific code lives under `src/resilient/` and `scripts/resilient/`.
- Environment setup and asset verification are being validated on the hardware listed below. Reproduction results will be added after the full benchmark completes.

## Repository layout

```text
Resilient/
├── configs/                       # Upstream Hydra configs
├── experiments/libero/            # Upstream LIBERO evaluation
├── src/fastwam/                   # Pinned upstream FastWAM implementation
├── src/resilient/                 # Resilient extensions and adapters
├── scripts/resilient/             # Reproducible download/verification helpers
├── environment/                   # Conda specification and environment notes
├── manifests/                     # Pinned upstream and external asset metadata
├── reproduce/fastwam_libero/      # Minimal and full reproduction recipes
├── reports/baselines/             # Curated, small reproduction records
├── checkpoints/fastwam_release/   # Downloaded weights; ignored by Git
├── data/lerobot_v30/              # LIBERO LeRobot 3.0 data; ignored by Git
├── runs/                          # Training outputs; ignored by Git
├── evaluate_results/              # Raw evaluation outputs; ignored by Git
└── AILOG/WORKLOG.md                # Local Chinese work log; ignored by Git
```

## Hardware baseline

| Component | Required/recommended | Validated machine |
| --- | --- | --- |
| OS | Linux x86_64 | Linux 5.4, x86_64 |
| Python | CPython 3.10.8 | CPython 3.10.8 |
| GPU | NVIDIA GPU with BF16 support; 1 GPU for minimal evaluation | 8 × RTX 6000 Ada, 48 GB each |
| NVIDIA driver | Compatible with CUDA 12.8 PyTorch wheels | 570.133.07 |
| System memory | To be confirmed by full evaluation | 503 GiB |
| Disk | At least 30 GB for environment + checkpoint; more for datasets/results | 695 GB free before setup |

FastWAM defaults to eight persistent LIBERO workers. Set `MULTIRUN.num_gpus` to the number of available GPUs for full evaluation.

## Environment setup

The environment is split into three synchronized files:

- `environment/environment.yml` pins Python and pip.
- `requirements.txt` pins all FastWAM runtime dependencies, including the CUDA 12.8 PyTorch build.
- `requirements-dev.txt` adds development and test tools.

```bash
conda env create -f environment/environment.yml
conda activate resilient-fastwam

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# Optional development tools
python -m pip install -r requirements-dev.txt
```

The official LIBERO package is installed separately at the exact commit recorded in `manifests/upstream.json`; MuJoCo must remain at `3.3.2`. The final validated installation command will be recorded here after compatibility testing.

## External assets

### Released FastWAM checkpoint

Source: <https://huggingface.co/yuanty/fastwam>

```bash
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 \
  --local-dir ./checkpoints/fastwam_release
```

Expected paths:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

The checkpoint is 12,041,735,140 bytes and has SHA-256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`. The statistics checksum will be populated after download and verified by `scripts/resilient/verify_assets.py`.

### LIBERO LeRobot 3.0 dataset

Training and follow-up work use the snapshot at <https://huggingface.co/datasets/yuanty/LIBERO-fastwam>:

```bash
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --include "lerobot_v30/**" \
  --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
  --local-dir ./data
```

Expected suite directories:

```text
data/lerobot_v30/libero_10_no_noops_lerobot/
data/lerobot_v30/libero_goal_no_noops_lerobot/
data/lerobot_v30/libero_object_no_noops_lerobot/
data/lerobot_v30/libero_spatial_no_noops_lerobot/
```

Evaluation of the released checkpoint does not read the training dataset, but the dataset is prepared for subsequent work. Asset revisions, sizes, and checksums are tracked in `manifests/assets.json`; actual files are never committed.

## Reproducing the LIBERO baseline

First validate the environment and assets:

```bash
python scripts/resilient/capture_environment.py --output AILOG/environment.json
python scripts/resilient/verify_assets.py
```

Run a one-task, one-episode integration evaluation:

```bash
bash reproduce/fastwam_libero/evaluate_minimal.sh
```

Run the full four-suite benchmark (40 tasks × 50 episodes):

```bash
NUM_GPUS=8 bash reproduce/fastwam_libero/evaluate_full.sh
```

Raw videos, worker logs, and per-task outputs remain under `evaluate_results/` and are ignored. Curated metrics and provenance are copied to `reports/baselines/` after validation.

## Extension switches and baseline protection

The following policy is mandatory:

1. Prefer implementing new work in `src/resilient/` without editing `src/fastwam/`.
2. If an upstream FastWAM file must change, gate the new behavior behind an explicit Hydra/CLI switch.
3. Every Resilient switch must default to `false` or the exact upstream value, so the documented baseline command remains unchanged.
4. Add the switch to the table below, both README versions, its Hydra config, and tests in the same commit.
5. Record unavoidable upstream patches in `docs/upstream-patches.md`.

| Switch | Default | Scope | Baseline effect |
| --- | --- | --- | --- |
| None | — | No FastWAM source modification has been made | None |

## Development checks

The repository separates lightweight CI checks from the full GPU environment:

```bash
python -m pip install -r requirements-ci.txt
PYTHONPATH=src python -m pytest -m "not gpu and not libero"
ruff check src/resilient tests scripts/resilient
```

Delete one-off debug scripts and outputs after use. Durable GPU/LIBERO checks belong in `tests/integration/` and must be explicitly marked.

## Reproducibility contract

- Use repository-relative or user-supplied paths; never commit machine-specific absolute paths.
- Record Git SHA, upstream SHA, commands, Hydra overrides, seeds, hardware, environment, asset revisions/checksums, and results.
- Update the environment files and both README versions whenever a dependency changes.
- Keep datasets, checkpoints, simulator assets, caches, videos, logs, and raw experiment outputs out of Git.
- Use English for source code and code comments. Maintain `README.md` in English, `README.zh-CN.md` in Simplified Chinese, and the ignored `AILOG/WORKLOG.md` in Chinese.

## Upstream and license

FastWAM is integrated from <https://github.com/yuantianyuan01/FastWAM> and retains its MIT license and Git history. See `LICENSE` and `manifests/upstream.json` for provenance.
