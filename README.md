# Resilient

[简体中文](README.zh-CN.md) | **English**

Resilient is a reproducibility-first research codebase built on [FastWAM](https://github.com/yuantianyuan01/FastWAM) for LIBERO evaluation and follow-up research. The FastWAM baseline is integrated at the repository root because its Hydra configs and evaluation entry points use repository-relative paths.

## Current status

- FastWAM upstream is pinned to commit `7faa71108368fbb3b6885649f112af607427a2d4`.
- The baseline target is `libero_uncond_2cam224.pt` with its released dataset statistics.
- Reproduction uses the original checkpoint setting `EVALUATION.sigma_shift=5.0`.
- FastWAM's default behavior is unchanged; the only core addition is a documented, default-off
  denoising-trace return switch. Project-specific implementations live under `src/resilient/`.
- The standalone Python/CUDA stack, all pinned assets, a headless LIBERO EGL reset, the one-episode integration evaluation, and the full 2,000-episode benchmark are validated on the hardware below.
- The severity-aware Fault layer and OPSD-Flow implementation have reached CPU/configuration/
  simulator smoke validation; full multi-GPU OPSD training is not yet validated.

## Repository layout

```text
Resilient/
├── configs/                       # Hydra configs, including Fault/adapter/OPSD parameters
├── experiments/libero/            # LIBERO evaluation plus a default-off Fault hook
├── src/fastwam/                   # Pinned FastWAM plus registered default-off patches
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

## Reusable fault pipeline and evaluation

Fault definitions are model-independent YAML files under `configs/fault/`. A `FaultPipeline`
orders plugins with reset, observation, action, pre-step, post-step, suspend, checkpoint-state,
and detach hooks. This supports later image occlusion/contamination, actuator limits, joint-motion
degradation, and dynamics faults without adding fault-specific branches to Fast-WAM. Every fault
has a stable `family` and a separate physical `severity` (`name`, numeric `value`, `unit`, and an
optional categorical `level`), so +10/+20/+30 degree rotations are three severities of the same
fault family.

The committed camera example is `configs/fault/visual/wrist_camera_local_z.yaml`. Evaluate its
default +30 degree severity on four GPUs with:

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --gpus 0,1,2,3 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

Use `--severity 10`, `--severity 20`, or `--severity 30` to override the one fault's physical
magnitude without changing its family. Use eight IDs for an eight-worker evaluation. The default
output root is ignored `evaluate_results/faults/`; a canonical child name includes family, target,
severity, unit, and checkpoint, for example
`visual-camera_pose-robot0_eye_in_hand-rotation_angle-p30p0degree__model-libero_uncond_2cam224/`.
It contains raw task results/videos, `summary.json`, `fault_summary.md`, `fault_manifest.json`, and
`fault_comparison.png`. The 2 x 2 image uses one simulator state for nominal/faulted third-person
and wrist images and annotates the faulted panels with the resolved fault metadata.
Add `--create-only` to validate the comparison render and distributed task configuration without
loading Fast-WAM or running episodes.

To evaluate an OPSD-trained LoRA adapter, keep the released Fast-WAM checkpoint as the base and
pass both the final adapter and its resolved training config:

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --opsd-adapter runs/opsd/wrist_camera_local_z_30deg/checkpoints/state/step_00000005/adapter.pt \
  --opsd-config runs/opsd/wrist_camera_local_z_30deg/resolved_config.yaml
```

`--opsd-adapter` and `--opsd-config` must be supplied together. The canonical output name adds
the adapter checkpoint identifier, and the manifest records the base model, adapter, and config.
Without these arguments, evaluation follows the unchanged Fast-WAM checkpoint path.

For an unseen-state evaluation of an existing checkpoint, first generate a validation state bank
from seeded LIBERO resets. The command also reconstructs the exact official-state exposure and
environment seeds from the completed OPSD run metrics:

```bash
python scripts/resilient/generate_libero_validation_states.py \
  --training-run runs/opsd/my_run \
  --output-dir data/libero_state_banks/my_unseen_validation \
  --states-per-task 50 \
  --base-seed 104729
```

Then add both manifests to the normal evaluation command:

```bash
  --state-bank-manifest data/libero_state_banks/my_unseen_validation/validation_manifest.json \
  --training-state-manifest data/libero_state_banks/my_unseen_validation/training_reference_manifest.json
```

The state-bank path is ignored by Git as dataset content. The evaluator verifies every simulator
state fingerprint and refuses to start if any state hash or generation seed overlaps training.
The switch is disabled by default, preserving the official Fast-WAM benchmark path and allowing
existing base/LoRA checkpoints to be evaluated without retraining.

For camera pose faults, translation is expressed in metres along the original camera-local axes;
rotation uses right-handed local X-then-Y-then-Z rotations in degrees. MuJoCo cameras look along
local `-Z`; local `+X` is raw-image right and local `+Y` is raw-image up. The plugin restores the
nominal pose temporarily when privileged observations are requested and never advances the
simulator while making the pair.

## OPSD-Flow adaptation for Fast-WAM

The implementation is a clean adaptation of the OPSD idea from
<https://github.com/siyan-zhao/OPSD> at the revision in `manifests/upstream.json`. That revision
does not include a repository license, so no OPSD source was copied. The important algorithmic
invariant is preserved: the Teacher does **not** generate another answer/action trajectory. The
Student runs Fast-WAM's original sampler once; the frozen Teacher only evaluates every latent on
that exact Student trajectory with privileged conditioning.

The main components are:

| Path | Responsibility |
| --- | --- |
| `src/resilient/faults/` | Reusable, severity-aware robot fault plugins and paired observations |
| `src/resilient/opsd/teacher_inputs/` | Replaceable privileged-input policy; current provider changes only faulted images to clean images |
| `src/resilient/opsd/adapters.py` | Fast-WAM-aligned LoRA discovery, freezing audit, enable/disable, and adapter state |
| `src/resilient/opsd/model_adapter.py` | Score a Student latent with Fast-WAM's action vector field, without a scheduler step |
| `src/resilient/opsd/losses.py` | Per-coordinate clipped flow-matching objective |
| `src/resilient/opsd/trainer.py` | Memory-bounded scoring, distributed gradient update, and resume state |
| `configs/opsd/fastwam_libero.yaml` | Multi-epoch parameters (default: one epoch) |
| `scripts/resilient/train_opsd.sh` | Strict 4/8-GPU Accelerate launcher |

Fast-WAM training freezes the VAE/text encoder, trains both MoT experts, and trains the proprio
encoder. The OPSD adapter freezes the entire released checkpoint and inserts LoRA into every
`nn.Linear` in `video_expert`, `action_expert`, and `proprio_encoder`; it excludes the VAE and text
encoder. This matches the original trainable high-level scope, while deliberately adapting only
linear weights (not expert Conv3d, normalization, modulation, or bias tensors). An audit JSON lists
every target and trainable parameter count before training.

For Student denoising step `k`, latent `x_k` is scored twice: `v_S(x_k,c_fault)` with active LoRA
and `stopgrad(v_T(x_k,c_clean))` with LoRA disabled. With optional valid-action mask `m`, the step
loss is the mean of `min((v_S-v_T)^2, 0.05)` over valid horizon/action coordinates. Step losses are
weighted by `|delta_sigma_k| / sum_j |delta_sigma_j|` and then averaged across the batch. Thus the
Teacher supplies a target vector field on Student points; it does not denoise, take a scheduler
step, or regenerate actions.

The rollout uses `opsd.rollout.num_inference_steps: ${eval_num_inference_steps}` and validates the
captured timesteps/deltas against Fast-WAM's scheduler. For the released LIBERO task config this is
exactly **10**, not an independent OPSD choice. The action horizon likewise interpolates to
`data.train.num_frames - 1`, currently 32. Environment and inference seeds both interpolate from
the project seed (42 by default), matching baseline Fast-WAM evaluation; process-local training
RNGs use `seed + rank`.

Model construction also matches released-checkpoint evaluation: `load_text_encoder=true`,
`skip_dit_load_from_pretrain=true`, and `action_dit_pretrained_path=null`. The architecture and
documented shared VAE/text assets are loaded first, then `ckpt` supplies both trained experts. A
runtime invariant rejects configurations that would redundantly download/load separate pretrained
video or action DiT weights.

First perform the no-CUDA configuration smoke check:

```bash
python scripts/resilient/train_opsd.py opsd.validate_only=true
```

When GPUs are available, start training on four or eight GPUs. The default is one epoch with 50
independently reset trajectories per task (2,000 trajectories total);
override `opsd.num_epochs` explicitly for longer runs:

```bash
bash scripts/resilient/train_opsd.sh 4
bash scripts/resilient/train_opsd.sh 8 output_dir=runs/opsd/my_run
bash scripts/resilient/train_opsd.sh 4 opsd.num_epochs=20 max_checkpoints=1
```

Select non-default physical devices through `CUDA_VISIBLE_DEVICES`, for example
`CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/resilient/train_opsd.sh 4`.
The launcher uses an OPSD-specific DeepSpeed ZeRO-2 config with
`train_micro_batch_size_per_gpu=1`, matching the runtime's one-trajectory-at-a-time
updates. The shared Fast-WAM training DeepSpeed config is left unchanged.

All parameters remain in YAML: fault and severity under `configs/fault/`, LoRA under
`configs/adapter/`, Teacher input under `configs/teacher_input/`, and optimization/rollout settings
under `configs/opsd/fastwam_libero.yaml`. Override `ckpt`, `opsd.dataset_stats_path`, and
`output_dir` with repository-relative paths as needed. Resume with
`resume=runs/opsd/my_run/checkpoints/state/step_XXXXXXXX` or `resume=auto` while reusing the same
explicit output directory. The runtime creates immutable `(suite, task, rollout, initial-state,
environment-seed, inference-seed)` descriptors, deterministically shuffles the global schedule,
then assigns it to ranks by striding. Each task uses LIBERO initial states 0--49 exactly once.
Seeds derive from descriptor identity rather than execution order, so resumed samples are exact.
Resume is accepted at completed trajectory boundaries and rejects a resolved-config hash mismatch.
Raw training state, adapter checkpoints, metrics, and provenance remain under ignored `runs/`.
Each epoch visits all 40 standard tasks 50 times and writes a complete resumable state at its boundary.
`max_checkpoints` (default 2) removes older states only after a new state is safely written. This
matters because one reference ZeRO state occupies approximately 50 GB.

The reference hardware remains Linux, 8 x NVIDIA RTX 6000 Ada 48 GB, CUDA 12.8, and bf16. Four-
and eight-GPU execution are the supported defaults; full OPSD training has not yet been run or
timed, so its final peak-memory requirement is not yet claimed.
The current validation covers CPU unit tests, Hydra composition, LoRA injection/auditing, and a
real LIBERO paired-camera render, but not model loading or a distributed optimizer step.

## Extension switches and baseline protection

The following policy is mandatory:

1. Prefer implementing new work in `src/resilient/` without editing `src/fastwam/`.
2. If an upstream FastWAM file must change, gate the new behavior behind an explicit Hydra/CLI switch.
3. Every Resilient switch must default to `false` or the exact upstream value, so the documented baseline command remains unchanged.
4. Add the switch to the table below, both README versions, its Hydra config, and tests in the same commit.
5. Record unavoidable upstream patches in `docs/upstream-patches.md`.

| Switch | Default | Scope | Baseline effect |
| --- | --- | --- | --- |
| `EVALUATION.fault.pipeline.enabled` | `false` | Enable an ordered robot-fault pipeline in the LIBERO evaluator | The evaluator follows its exact upstream reset/step path when disabled |
| `return_denoising_trace` in `FastWAM.infer_action` | `false` | Return detached pre-step Student latents/timesteps/deltas for OPSD | Return schema and action sampling are unchanged when false |
| `EVALUATION.opsd_adapter.enabled` | `false` | Inject and load an OPSD LoRA adapter after the base Fast-WAM checkpoint | No modules are injected and base evaluation is unchanged when false |

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
