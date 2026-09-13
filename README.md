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
- The core simulator-owned Fault catalog now provides five visual and five embodiment faults,
  reusable YAML configurations, and aligned image/video demonstrations.
- Outcome-Guided FPO for the simulator-owned joint-1 50% motion-retention Fault is implemented and
  covered by CPU/configuration tests; a real multi-GPU optimization smoke test is still pending.

## Repository layout

```text
Resilient/
├── configs/                       # Hydra configs, including Fault/OPSD/outcome-FPO parameters
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

> **Core project interface:** all Faults are installed in `FaultedEnvironment` when the LIBERO
> simulator is created. Fast-WAM, FPO, OPSD, and any later policy use the unchanged environment
> API and contain no Fault-specific implementation. The same YAML can therefore be reused for
> evaluation, training, shadow rollouts, or compound-Fault experiments.

Fault definitions are model-independent YAML files under `configs/fault/`. At environment
construction, `get_libero_env(..., fault_pipeline=pipeline)` installs an enabled pipeline into a
transparent `FaultedEnvironment`. The environment then owns reset, state-load, observation,
action, simulator-step, suspend, checkpoint-state, and detach lifecycles. A policy only calls the
normal `reset`, `set_init_state`, and `step` API; it contains no fault hook calls. This supports
later image occlusion/contamination, actuator limits, joint-motion degradation, dynamics faults,
and ordered compound faults without adding fault-specific branches to Fast-WAM or another policy.
Every fault has a stable `family` and a separate physical `severity` (`name`, numeric `value`,
`unit`, and an optional categorical `level`), so +10/+20/+30 degree rotations are three severities
of the same fault family.

### Ten-Fault catalog

The ready-to-run catalog lives under `configs/fault/catalog/`. Its severe settings intentionally
make demonstrations easy to inspect; change `severity` and the physical parameters for experiments.

| ID | Family | Bottom-layer behavior | Severe demonstration config |
| --- | --- | --- | --- |
| V1 Camera Rotation | `visual.camera_pose` | Mutates MuJoCo camera extrinsic quaternion | `visual/camera_rotation.yaml`: both cameras, local +Z 45 deg |
| V2 Camera Translation | `visual.camera_pose` | Mutates MuJoCo camera extrinsic position | `visual/camera_translation.yaml`: both cameras, local +X 0.12 m |
| V3 Defocus Blur | `visual.defocus_blur` | Gaussian optical degradation in the environment-owned camera sensor | `visual/defocus_blur.yaml`: sigma 8 px |
| V4 Local Occlusion | `visual.local_occlusion` | Soft-edged round ink spots in the environment-owned camera sensor | `visual/local_occlusion.yaml`: eight configured spots, largest diameter 0.44 image width |
| V5 Illumination Change | `visual.illumination` | Affine RGB response before the observation leaves the environment | `visual/illumination_change.yaml`: gain 0.2 with color shift |
| E1 Joint Motion Degradation | `structure.joint_motion` | Retains a fraction of realized per-step displacement/velocity | `structure/joint_motion_degradation.yaml`: joint 1 retention 0.2 |
| E2 Joint Position Bias | `structure.joint_position_bias` | Introduces one fixed, non-cumulative joint zero offset per loaded state | `structure/joint_position_bias.yaml`: joint 1 +20 deg |
| E3 Joint Backlash | `structure.joint_backlash` | Consumes joint travel after each direction reversal | `structure/joint_backlash.yaml`: joint 1 gap 3 deg |
| E4 Joint Range Limitation | `structure.joint_range_limit` | Clips realized joint position to reduced absolute bounds | `structure/joint_range_limitation.yaml`: joint 1 in [-10, +10] deg |
| E5 Periodic Joint Freeze | `structure.periodic_joint_freeze` | Freezes when the joint crosses angularly periodic defective-gear positions | `structure/periodic_joint_freeze.yaml`: every 10 deg, hold 35 control steps |

V1/V2 alter MuJoCo rendering geometry. V3--V5 model camera hardware/sensor output and are applied
inside `FaultedEnvironment`, before any model preprocessing. E1--E5 modify realized MuJoCo joint
state after every environment dynamics step. Thus none of the ten Faults is implemented in a model,
controller-specific evaluator, or recovery algorithm. Stateful E2/E3/E5 internals are included in
`fault_runtime_state_dict()`, so same-state shadow environments can reproduce them exactly.

Load one catalog Fault, or compose several by concatenating their `faults` entries:

```python
from omegaconf import OmegaConf

from experiments.libero.libero_utils import get_libero_env
from resilient.faults import build_fault_pipeline

fault_config = OmegaConf.to_container(
    OmegaConf.load("configs/fault/catalog/structure/joint_backlash.yaml"),
    resolve=True,
)
pipeline = build_fault_pipeline(fault_config)
env, task_description = get_libero_env(task, 256, seed=42, fault_pipeline=pipeline)
observation = env.reset()
observation, reward, done, info = env.step(action)
```

`targets` accepts multiple camera or scalar MuJoCo joint names where documented. Every plugin
validates units and ranges and reports portable metadata including `family`, targets, physical
severity, parameters, and injection layer. See `configs/fault/README.md` for exact formulas and
extension points.

Generate all five annotated 2 x 2 visual comparisons and five time-aligned embodiment videos:

```bash
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:third_party/LIBERO:. \
python scripts/resilient/demonstrate_fault_catalog.py \
  --catalog configs/fault/demo_catalog.yaml \
  --faults all \
  --output-root evaluate_results/fault_catalog
```

Pass catalog IDs after `--faults` to render a subset. Visual artifacts are
`nominal_vs_fault.png`; embodiment artifacts are `nominal_vs_fault.mp4`. Each panel is labeled in
its top-left corner. Embodiment panels start from the same simulator state, replay the same
`JOINT_POSITION` commands, use the same frame count, and report target-joint and end-effector
divergence. Every embodiment action has only the failed joint-1 dimension nonzero; the other six
arm-joint command dimensions remain zero. The motion phases are explicit in
`configs/fault/demo_catalog.yaml`; reversal is included for backlash and periodic stick.
An all-Fault run writes `catalog_manifest.json`; a subset uses
`catalog_manifest__<selected-ids>.json`, so it cannot overwrite the full manifest. The default
output root and its JSON summaries are generated artifacts and remain ignored by Git.

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

### Joint-motion fault semantics

The `structure.joint_motion` family is installed by the environment when the MuJoCo simulator is
created. It acts at the environment dynamics-step boundary, independently of the controller,
Fast-WAM, OPSD, or any later policy. It accepts one or more MuJoCo joint names. The first built-in law is
`proportional`; the example retains 50% of `robot0_joint1` displacement and velocity produced by
every environment control step. The degradation law has its own registry, so future deterministic
non-linear functions do not require changes to the simulator runner or model code. The catalog
demonstration command above supersedes the former OSC trajectory demo: it uses `JOINT_POSITION`
with only joint 1 commanded, making the realized effect attributable to the selected Fault.

The proportional law is defined at the environment dynamics-step boundary:

```text
q_fault_after = q_before + retention * (q_nominal_after - q_before)
```

It also scales the selected joint velocity by `retention`. This intentionally models kinematic
motion loss rather than actuator torque loss. The installation survives environment resets and is
removed only when the environment closes. Multiple configured simulator faults compose in YAML
order. Because a closed-loop controller can react during subsequent control steps, the final
faulty joint displacement is not expected to equal exactly `retention * clean_final_displacement`.
The original no-fault path remains unchanged when no pipeline is installed.

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

## Fault-severity curves from Video DiT latents

`scripts/resilient/evaluate_fault_severity.py` evaluates one fixed LIBERO task over an ordered
severity sweep. Each severity uses the same 50 initial states by default. The released Fast-WAM
checkpoint is the default model; another compatible checkpoint or a base-plus-OPSD adapter can be
selected explicitly. Severity jobs are assigned to the requested physical GPUs, so independent
severity values may run in parallel.

For example, evaluate wrist-camera local +Z rotations of 10, 20, and 30 degrees on Spatial task 0:

```bash
python scripts/resilient/evaluate_fault_severity.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --severities 10 20 30 \
  --suite libero_spatial \
  --task-id 0 \
  --gpus 0,1,2
```

The default root is the Git-ignored `evaluate_results/fault_detection/`; `--output-dir` selects a
different root. The canonical study directory records the fault family, target, severity field,
checkpoint, suite, and task. It contains an immutable `study_manifest.json`, one reproducible
subdirectory per severity, `metrics_summary.json`, `metrics_summary.csv`, residual matrix data,
human-readable `metrics_summary.md`, and one PNG per requested metric. Episode videos are disabled by default to avoid large incidental
outputs; add `--save-videos` when they are needed. Interrupted studies reuse a severity only after
both its result JSON and residual prototype file exist.

The metric list and statistical parameters live in
`configs/fault_detection/latent_v1.yaml`; `--metrics ...` can override only the selected metrics.
The first version provides:

| Metric | Definition and aggregation |
| --- | --- |
| LPE | Mean squared error between aligned future `Z_pred` and `Z_real` |
| LCD | One minus flattened cosine similarity between the aligned latents |
| RM | Mean absolute value of `R = Z_real - Z_pred` |
| RCS | Cosine similarity of each episode residual prototype against the same episode at clean severity |
| Residual matrix | Mean paired-episode cosine for every severity pair, plus a 50x50 episode heatmap for each severity |
| RTC | Mean cosine similarity between consecutive complete-window residuals within an episode |
| Spearman rho | Rank correlation between severity and episode RM, with deterministic bootstrap uncertainty |
| FaultScore | Episode RM standardized by the clean-run RM mean and sample standard deviation |
| ROC-AUC | Clean-versus-fault discrimination from RM, separately for each nonzero severity, with deterministic bootstrap uncertainty |

LPE, LCD, RM, RCS, RTC, and FaultScore are first averaged within an episode and then summarized
over episodes. The JSON/CSV records count, mean, sample variance, standard deviation, and SEM.
Error-bar plots use +/- one standard deviation by default; set `error_bar: sem` in the metrics YAML
for SEM. Spearman and ROC-AUC store bootstrap variance. FaultScore and ROC-AUC require a clean run;
the default configuration automatically adds severity 0 if the command omitted it. Pass
`--no-auto-clean` to require it explicitly.

Fast-WAM's Video DiT predicts a vector field at each diffusion step, not a final latent directly.
Here `Z_pred` means the final `latents_video` after the original video scheduler completes all
denoising steps. `Z_real` is produced by the same frozen VAE from the actually observed, faulted
clip. With the released model, the clip contains 9 video frames aligned to action steps
`0,4,...,32`. The script therefore executes each complete 32-action prediction window before
replanning; incomplete terminal windows are excluded. Both tensors exclude temporal latent index
0 because it is the encoded conditioning image, leaving only future latents. This full-window
protocol is necessary for shape and time alignment, but its success rate is not directly
comparable with the standard Fast-WAM benchmark, which replans after 10 actions.

The action used by the environment still comes from the unchanged `infer_action` path. Joint
inference is called separately only to collect the video prediction. The opt-in evaluator setting
`EVALUATION.fault_detection.enabled` defaults to `false`; Fast-WAM's added
`return_video_latents=false` and `decode_video=true` defaults preserve its original return keys and
decoding. New aligned clip metrics can be registered in
`src/resilient/fault_detection/metrics.py`; cross-severity metrics and plots remain isolated in
`study.py`. Existing unseen-state manifests can be supplied with `--state-bank-manifest` and
`--training-state-manifest` exactly as in fault evaluation.

The integration smoke test used Linux, CUDA 12.8, bf16, and one NVIDIA RTX 6000 Ada 48 GB. The
worker occupied approximately 25.2 GB after compilation. Each concurrent severity owns one GPU;
additional listed GPUs increase severity-level parallelism rather than splitting one episode. A
complete 50-episode severity sweep has not yet been timed, so no full-run duration is claimed.

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

To continue from a completed LoRA as a fresh optimization stage, set
`adapter.initial_checkpoint` and advance `opsd.epoch_offset`. This loads only the Student LoRA,
starts a fresh optimizer/cosine schedule, and uses the next epoch's shuffled order and derived
seeds. It differs intentionally from `resume`, which restores optimizer and scheduler state within
an interrupted stage.

The reference hardware remains Linux, 8 x NVIDIA RTX 6000 Ada 48 GB, CUDA 12.8, and bf16. Four-
and eight-GPU execution are the supported defaults; full OPSD training has not yet been run or
timed, so its final peak-memory requirement is not yet claimed.
The current validation covers CPU unit tests, Hydra composition, LoRA injection/auditing, and a
real LIBERO paired-camera render, but not model loading or a distributed optimizer step.

## Outcome-Guided FPO for embodied Fault recovery

This path implements a single-stage recovery policy without Fault-representation pretraining. It
adapts the conditional-flow policy-gradient estimator from
[Flow Matching Policy Gradients / FPO](https://github.com/akanazawa/fpo), pinned to commit
`418c2554f7cd22d52e14c07d951280929d73bf2f` in `manifests/upstream.json`. The implementation is an
independent PyTorch adaptation of the repository's Apache-2.0 Playground objective; no MuJoCo or
JAX runtime code is vendored.

The frozen base Fast-WAM is the Teacher and evaluator. From the same current image, language, and
proprioception as the Student, its Video DiT first predicts a nominal nine-frame future with the
released 10-step video scheduler. The Teacher target is the temporal-change representation at
Video DiT block 19, evaluated at training timestep 500. Each Student samples four final normalized
32x7 action chunks with Fast-WAM's unchanged 10-step action sampler. One shadow simulator is
restored to the same LIBERO state and the same environment-owned Fault state before each of the
four candidates; each candidate is executed for all 32 actions and recorded at steps
`0,4,...,32`. The frozen Teacher
network encodes each realized video at the same layer, future tokens, timestep, and noise. The only
reward is the mean future-token cosine similarity to the nominal temporal change. Success,
progress, collision, Fault metadata, and simulator reward are never added to the policy reward.

FPO treats the native deterministic sampler as a black box; it does not invent an SDE and does not
use denoising-transition log probabilities. For every sampled final action `A`, eight reproducible
Monte Carlo pairs draw continuous action timestep `t` and Gaussian noise `epsilon`. Fast-WAM's
native training scheduler constructs `x_t=(1-t)A+t*epsilon`, its action expert predicts velocity
`v`, and the scorer uses `epsilon_hat=x_t+(1-t)v`. The per-pair conditional flow-matching score is
the mean squared error between `epsilon_hat` and `epsilon`. Old and current scores share the exact
same action, timestep, and noise; their eight losses are averaged before computing
`ratio=exp(loss_old-loss_current)`. The PPO-style ratio clip is 0.05 and its numerical log-ratio
guard is `[-3,3]`.

The four same-state rewards provide a leave-one-out standardized advantage. This is only a
same-state control variate, not a GRPO model of the ten denoising steps. Reward-tied groups below
the configured standard-deviation floor perform a zero update. Each on-policy collection is reused
for two FPO update epochs. The frozen checkpoint, VAE, and text encoder never update; a zero-start
Recovery LoRA is trained over the same high-level scope as Fast-WAM training:
`video_expert`, `action_expert`, and `proprio_encoder`. The default adapter is rank 64, alpha 128,
dropout 0; AdamW uses learning rate `1e-6`, zero weight decay, bf16, and gradient norm 0.1.

The implementation is separated by responsibility:

| Path | Responsibility |
| --- | --- |
| `src/resilient/outcome_fpo/outcome.py` | Frozen nominal Video DiT target and cosine-only outcome reward |
| `src/resilient/outcome_fpo/collector.py` | Same-state shadow restoration and 32-action/9-frame alignment |
| `src/resilient/outcome_fpo/policy.py` | Native action sampling and Fast-WAM conditional-flow scores |
| `src/resilient/outcome_fpo/advantage.py` | Leave-one-out same-state advantages and tied-group guard |
| `src/resilient/outcome_fpo/objective.py` | FPO loss ratio and clipped surrogate objective |
| `src/resilient/outcome_fpo/trainer.py` | Distributed updates, short-lived buffers, and complete resume state |
| `src/resilient/outcome_fpo/runtime.py` | LIBERO schedule, split checks, provenance, logging, and orchestration |
| `configs/outcome_fpo/fastwam_libero_joint1_half.yaml` | Reproducible first experiment and all tunable parameters |
| `scripts/resilient/train_outcome_fpo.sh` | Strict 4/8-GPU Accelerate/ZeRO-2 launcher |

The first experiment reuses the generic Fault pipeline through
`configs/fault/structure/panda_joint1_half_motion.yaml`: MuJoCo joint
`robot0_joint1` retains 0.5 of its nominal per-step displacement. The Fault is installed in the
simulator and therefore applies independently of Fast-WAM or this optimizer. No Fault-specific
logic is duplicated in the FPO modules.

Training matches the earlier visual-Fault OPSD scale: one epoch, all 40 standard LIBERO tasks, and
official initial states 0--49 once per task, for 2,000 same-state groups and 8,000 candidate
rollouts. Base seed 42 deterministically derives schedule, environment, action-inference, Teacher,
and Monte Carlo seeds from sample identity. Validation uses the existing
`data/libero_state_banks/opsd_step500_unseen_v1/validation_manifest.json`, generated from base seed
104729. Startup requires both its training-reference manifest and validation manifest and rejects
any state-hash or environment-seed overlap. Validation states are never loaded by training.

After installing the normal project environment and downloading the already documented Fast-WAM
checkpoint/statistics and LIBERO assets, validate the full Hydra configuration without CUDA:

```bash
python scripts/resilient/train_outcome_fpo.py outcome_fpo.validate_only=true
```

Start on exactly four or eight visible GPUs. Paths remain repository-relative unless explicitly
overridden:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/resilient/train_outcome_fpo.sh 4 \
  output_dir=runs/outcome_fpo/joint1_half_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/resilient/train_outcome_fpo.sh 8 \
  output_dir=runs/outcome_fpo/joint1_half_seed42_8gpu
```

The default global collection contains eight groups: two groups per rank on four GPUs or one group
per rank on eight GPUs. Intermediate checkpoints are written only after a complete collection under
`<output_dir>/checkpoints/state/step_XXXXXXXX/`; this rolling set retains the newest three states by
default. Every completed epoch is saved separately under
`<output_dir>/checkpoints/epochs/epoch_XXXX_step_XXXXXXXX/` and is never removed by rolling
retention. Both checkpoint types contain Accelerate/ZeRO optimizer and RNG state,
`recovery_adapter.pt`, and `trainer_state.json`; resolved configuration/provenance remains in the run
root. `resume=auto` selects the most advanced complete checkpoint across both locations. An explicit
path may point to either type, and a resolved-config hash mismatch fails closed.
`rollouts_rank_XX.jsonl` records state, seed, four rewards, and diagnostic success flags; the success
flags are not part of the reward.

Evaluate a completed Recovery LoRA with the existing generic Fast-WAM LoRA loader and the unseen
state bank. The option retains its historical `opsd` name but accepts this identical adapter
layout:

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/structure/panda_joint1_half_motion.yaml \
  --gpus 0,1,2,3 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --opsd-adapter runs/outcome_fpo/joint1_half_seed42/checkpoints/epochs/epoch_0001_step_XXXXXXXX/recovery_adapter.pt \
  --opsd-config runs/outcome_fpo/joint1_half_seed42/resolved_config.yaml \
  --state-bank-manifest data/libero_state_banks/opsd_step500_unseen_v1/validation_manifest.json \
  --training-state-manifest data/libero_state_banks/opsd_step500_unseen_v1/training_reference_manifest.json \
  --output-dir evaluate_results/outcome_fpo/joint1_half_seed42
```

The reference platform remains Linux, CUDA 12.8, bf16, and four or eight RTX 6000 Ada 48 GB GPUs.
The implementation intentionally reuses the existing OPSD ZeRO-2 launcher configuration and adds
no Python dependency, so `requirements*.txt`, the lock, and environment setup are unchanged. Real
multi-GPU model loading, peak memory, optimizer-step behavior, throughput, and final recovery
quality have not yet been smoke-tested; do not treat the current code-only validation as an
experimental result. Generated runs and checkpoints remain ignored by Git.

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
| `EVALUATION.fault_detection.enabled` | `false` | Collect time-aligned Video DiT/VAE latent metrics during LIBERO evaluation | No joint video inference, latent encoding, or metric files are produced when false |
| `return_video_latents` / `decode_video` in `FastWAM.infer_joint` | `false` / `true` | Expose final video latents and optionally avoid decoding for fault metrics | Original `video`/`action` result and decoding are unchanged at defaults |

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
