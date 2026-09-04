# Resilient

[简体中文](README.zh-CN.md) | **English**

Resilient is a reproducibility-first research codebase built on [FastWAM](https://github.com/yuantianyuan01/FastWAM) for LIBERO evaluation and follow-up research. The FastWAM baseline is integrated at the repository root because its Hydra configs and evaluation entry points use repository-relative paths.

## Current status

- FastWAM upstream is pinned to commit `7faa71108368fbb3b6885649f112af607427a2d4`.
- The baseline target is `libero_uncond_2cam224.pt` with its released dataset statistics.
- Reproduction uses the original checkpoint setting `EVALUATION.sigma_shift=5.0`.
- FastWAM core behavior is unchanged. Project-specific code lives under `src/resilient/` and `scripts/resilient/`.
- The standalone Python/CUDA stack, all pinned assets, a headless LIBERO EGL reset, the one-episode integration evaluation, and the full 2,000-episode benchmark are validated on the hardware below.

## Repository layout

```text
Resilient/
├── configs/                       # Upstream Hydra configs
├── experiments/libero/            # Upstream LIBERO evaluation
├── src/fastwam/                   # Pinned upstream FastWAM implementation
├── src/resilient/                 # Resilient extensions and adapters
├── scripts/resilient/             # Reproducible evaluation/download/verification tools
├── environment/                   # Environment specification, lock, and notes
├── manifests/                     # Pinned upstream and external asset metadata
├── reproduce/fastwam_libero/      # Minimal and full reproduction recipes
├── reports/baselines/             # Curated, small reproduction records
├── checkpoints/fastwam_release/   # Downloaded weights; ignored by Git
├── data/lerobot_v30/              # LIBERO LeRobot 3.0 data; ignored by Git
├── third_party/LIBERO/            # Pinned simulator checkout; ignored by Git
├── runs/                          # Training outputs; ignored by Git
├── evaluate_results/              # Raw evaluation outputs; ignored by Git
└── AILOG/WORKLOG.md                # Local Chinese work log; ignored by Git
```

## Hardware baseline

| Component | Required/recommended | Validated machine |
| --- | --- | --- |
| OS | Linux x86_64 | Linux 5.4, x86_64 |
| Python | CPython 3.10.20 | uv-managed standalone CPython 3.10.20 |
| Environment tool | uv 0.11.7 | uv 0.11.7 |
| GPU | NVIDIA GPU with BF16 support; at least 32 GB per worker recommended; 1 GPU works and 8 GPUs reproduce the parallel run | 8 × RTX 6000 Ada, 48 GB each; 24,728–25,593 MiB observed per worker |
| NVIDIA driver | Compatible with CUDA 12.8 PyTorch wheels | 570.133.07 |
| System memory | At least 128 GiB recommended for 8 workers; reduce the worker count on smaller hosts | 503 GiB; about 90 GiB used in a running snapshot |
| Disk | At least 80 GB for environment, weights, Git LFS objects, dataset, and outputs | 695 GB free before setup |

Git LFS is required for the pinned Wan component download; version 3.6.1 was validated here.

FastWAM defaults to eight persistent LIBERO workers. Set `MULTIRUN.num_gpus` to the number of available GPUs for full evaluation. The validated eight-GPU run took approximately 46 minutes including worker startup and model loading; fewer GPUs reduce memory requirements but increase wall time.

## Environment setup

The environment is split into synchronized specifications:

- `.python-version` pins the validated Python runtime.
- `requirements.txt` pins all FastWAM runtime dependencies, including the CUDA 12.8 PyTorch build.
- `requirements-libero.txt` pins the simulator-only LIBERO dependencies without downgrading FastWAM's current stack.
- `requirements-dev.txt` adds development and test tools.
- `environment/pip-freeze-cu128.txt` locks the validated direct and transitive package set.
- `environment/environment.yml` provides an optional equivalent Conda bootstrap.

```bash
uv python install 3.10.20
uv venv --python 3.10.20 .venv
source .venv/bin/activate

uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -r requirements-libero.txt
uv pip install --no-deps -e .

# Optional development tools
uv pip install --index-strategy unsafe-best-match -r requirements-dev.txt
```

For an exact replay after the lock has been validated, replace the first two install commands with:

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r environment/pip-freeze-cu128.txt
```

The `unsafe-best-match` flag is needed because the requirements file adds the PyTorch CUDA wheel index; all package versions remain pinned. The standalone runtime avoids host Conda library leakage observed with FFmpeg/torchcodec.

Install the official LIBERO simulator at the exact tested commit. Its historical full requirements file must not be installed because it would downgrade the FastWAM stack:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --no-deps --editable third_party/LIBERO \
  --config-settings editable_mode=compat
python scripts/resilient/configure_libero.py
export LIBERO_CONFIG_PATH="$(pwd)/AILOG/libero"
```

The compatibility editable mode is required by LIBERO's namespace-style source layout. The configuration helper verifies the pinned commit and generates machine-specific absolute simulator asset paths only in ignored `AILOG/libero/config.yaml`; no machine path is committed. MuJoCo remains pinned to `3.3.2`.

## External assets

### Released FastWAM checkpoint

Source: <https://huggingface.co/yuanty/fastwam>

The model card currently declares no weight license. Download for this reproduction, but do not redistribute the checkpoint until the publisher clarifies its terms.

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

The checkpoint is 12,041,735,140 bytes and has SHA-256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`. The statistics file is 40,939 bytes with SHA-256 `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`. Verify both with `scripts/resilient/verify_assets.py`.

### Wan inference components

Evaluation also needs the Wan UMT5 text encoder, tokenizer, and Wan 2.2 VAE. Download their exact ModelScope Git commits with Git LFS sparse checkout:

```bash
git lfs version
bash scripts/resilient/download_model_components.sh
```

The script refuses to replace a non-empty unmanaged directory or use a checkout at the wrong commit. These components require about 12.8 GB. Their exact paths, sizes, SHA-256 values, revisions, and Apache-2.0 licenses are recorded in `manifests/assets.json`. The evaluation wrappers retain FastWAM's original ModelScope default and use the local files after verification.

### LIBERO LeRobot 3.0 dataset

Training and follow-up work use the CC-BY-4.0 snapshot at <https://huggingface.co/datasets/yuanty/LIBERO-fastwam>:

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

Evaluation of the released checkpoint does not read the training dataset, but the dataset is prepared for subsequent work. The snapshot contains 46 files totaling 4,706,213,231 bytes; its deterministic path-plus-content tree SHA-256 is `fb0532da60ac971d785f9135d0f015746a105d4a768fb3ec0ba15399dea40e7b`. Repository `.gitkeep` placeholders are excluded. Asset metadata is tracked in `manifests/assets.json`, and actual dataset files are never committed.

## Reproducing the LIBERO baseline

First validate the environment and assets:

```bash
python scripts/resilient/capture_environment.py --output AILOG/environment.json
python scripts/resilient/verify_assets.py
```

The evaluation scripts set `LIBERO_CONFIG_PATH`, `DIFFSYNTH_MODEL_BASE_PATH`, `DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`, and `MUJOCO_GL=egl` from the repository location. Each can be overridden through the environment; no machine-specific path is stored in tracked code.

Run a one-task, one-episode integration evaluation:

```bash
GPU_ID=0 bash reproduce/fastwam_libero/evaluate_minimal.sh
```

`GPU_ID` defaults to `0` and is also used as `CUDA_VISIBLE_DEVICES` unless that variable is already set. This ensures the selected physical GPU is used rather than merely changing the result-file label.

Run the full four-suite benchmark (40 tasks × 50 episodes):

```bash
NUM_GPUS=8 bash reproduce/fastwam_libero/evaluate_full.sh
```

Raw videos, worker logs, and per-task outputs remain under `evaluate_results/` and are ignored. Curated metrics and provenance are copied to `reports/baselines/` after validation.

### Validated full result

On 2026-09-04, all 40 tasks and 2,000 episodes completed with an empty failure queue. Independent validation confirmed 40 unique result files, 50 episodes for every task, complete success/failure episode partitions, and 2,000 non-empty rollout videos. The reproduced overall success rate is **97.15%** (1,943/2,000), 0.45 percentage points below the 97.6% reported in [Fast-WAM arXiv v2, Table 2](https://arxiv.org/html/2603.16666).

| Suite | Reproduced | Successes | Published | Difference |
| --- | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 97.0% | 485/500 | 98.2% | -1.2 pp |
| LIBERO-Object | 99.8% | 499/500 | 100.0% | -0.2 pp |
| LIBERO-Goal | 96.6% | 483/500 | 97.0% | -0.4 pp |
| LIBERO-Long (`libero_10`) | 95.2% | 476/500 | 95.2% | 0.0 pp |
| **Average** | **97.15%** | **1,943/2,000** | **97.6%** | **-0.45 pp** |

The run used seed 42, 10 inference steps, sigma shift 5.0, CFG 1.0, action compilation, MuJoCo 3.3.2, EGL rendering, and eight persistent workers. Approximate wall time was 46 minutes; summed task runtime was 19,253.27 seconds because tasks ran in parallel. See `reports/baselines/fastwam-libero-full.json` for the machine-readable record and exact provenance.

### Validated minimal result

On 2026-09-04, `libero_spatial` task 0 completed successfully in its single episode with seed 42, 10 inference steps, sigma shift 5.0, action compilation enabled, and the released checkpoint. The rollout phase took 114.52 seconds including first-use TorchInductor compilation. See `reports/baselines/minimal-validation.json` for compact provenance. This is an integration check, not a statistically meaningful benchmark result.

## Evaluating camera-pose faults

The tracked `scripts/resilient/evaluate_camera_pose_fault.py` entry point evaluates one
deterministic camera position/orientation fault through the same persistent-worker FastWAM
pipeline used by the baseline. For example, rotate the wrist camera by +15 degrees about its
local X axis and evaluate the four standard suites on GPUs 0--3:

```bash
python scripts/resilient/evaluate_camera_pose_fault.py \
  --camera robot0_eye_in_hand \
  --position-offset 0.0 0.0 0.0 \
  --rotation-offset-deg 15.0 0.0 0.0 \
  --gpus 0,1,2,3 \
  --checkpoint ./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

Use `--gpus 0,1,2,3,4,5,6,7` for eight GPUs. Each listed GPU hosts one persistent model worker
and therefore needs the same approximately 32 GB recommended memory as the baseline. The default
run is 40 tasks x 50 episodes across `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`; use `--suites` and `--num-trials` for smaller experiments.

Fault parameter semantics are fixed and recorded in every output manifest:

- `--camera` accepts `agentview` or `robot0_eye_in_hand`.
- `--position-offset X Y Z` is in meters along the original local camera X/Y/Z axes. The script
  converts it into the MuJoCo parent frame, so one offset has consistent physical meaning across
  suites and tasks.
- `--rotation-offset-deg X Y Z` applies right-handed rotations about the camera's local axes in
  X-then-Y-then-Z order. MuJoCo cameras look along local `-Z`; local `+X` is raw-image right and
  local `+Y` is raw-image up. FastWAM's 180-degree image preprocessing does not change the
  physical rotation axes.
- `--checkpoint` selects the model. The script looks for
  `<checkpoint_stem>_dataset_stats.json` and then `dataset_stats.json` beside it; pass
  `--dataset-stats` when neither convention applies.
- `--output-dir` is an output root and defaults to `evaluate_results/camera_pose_faults/`.
  `--preview-suite`, `--preview-task-id`, and `--preview-init-state` select the deterministic
  state shown in the comparison image; their defaults are the first evaluated suite, task 0,
  and initial state 0.

The script creates a deterministic, resumable subdirectory whose name contains the camera,
translation, rotation, and checkpoint, for example:

```text
robot0_eye_in_hand-pos_xp0p000_yp0p000_zp0p000m-rot_xp15p0_yp0p0_zp0p0deg-model_libero_uncond_2cam224/
```

The directory contains raw per-task results and videos, `summary.json`, a compact
`camera_fault_summary.md`, a complete `camera_fault_manifest.json`, and
`camera_pose_comparison.png`. The comparison is a 2 x 2 grid containing original and faulted
`agentview`/wrist observations from the exact same initial state. The affected panel is annotated
at its upper-left corner with the position and rotation offsets. The compact Markdown summary
reports per-suite and exact success-weighted total rates. All these generated outputs remain
under an ignored directory and must not be committed.

## Extension switches and baseline protection

The following policy is mandatory:

1. Prefer implementing new work in `src/resilient/` without editing `src/fastwam/`.
2. If an upstream FastWAM file must change, gate the new behavior behind an explicit Hydra/CLI switch.
3. Every Resilient switch must default to `false` or the exact upstream value, so the documented baseline command remains unchanged.
4. Add the switch to the table below, both README versions, its Hydra config, and tests in the same commit.
5. Record unavoidable upstream patches in `docs/upstream-patches.md`.

| Switch | Default | Scope | Baseline effect |
| --- | --- | --- | --- |
| `EVALUATION.camera_pose_fault.enabled` | `false` | Apply the documented position/orientation offset to one LIBERO camera | None when disabled; the baseline scripts do not enable it |

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
