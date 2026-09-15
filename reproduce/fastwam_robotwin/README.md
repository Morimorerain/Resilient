# FastWAM RoboTwin reproduction

This recipe evaluates the official `robotwin_uncond_3cam_384.pt` checkpoint against Fast-WAM
Tables 1 and 3. It reuses the checkpoint authors' policy, three-camera preprocessing, qpos control,
single-task evaluator, and multi-GPU manager under `experiments/robotwin/`.

## Fixed protocol

- RoboTwin commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- FastWAM checkpoint repository revision: `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`
- Tasks: the 50 ordered entries in `third_party/RoboTwin/task_config/_eval_step_limit.yml`
- Conditions: `demo_clean` and `demo_randomized`
- Trials: 100 accepted expert-valid simulator seeds per task and condition
- Instructions: unseen
- Cameras: head, left wrist, and right wrist, concatenated to 384 x 320
- Action: 14-dimensional dual-arm qpos, horizon 32, replan every 24 actions
- Sampling: 10 flow steps, sigma shift 5.0, CFG 1.0, CPU noise
- Faults and adapters: disabled

The complete run contains 10,000 policy episodes. RoboTwin first executes its expert to reject
invalid scene seeds; those expert checks are not counted as policy episodes and are part of the
official evaluator behavior.

## Staged validation

Install the pinned shared Wan components with `download_model_components.sh`, then run
`verify_robotwin.py --verify-hashes` before using a GPU. Execute `smoke`, `coverage`, and `paper`
modes in that order as documented in the root README. The launcher checks selected GPUs for active
compute processes. Never bypass that check merely to increase concurrency.

The manager stores a protocol fingerprint beside every run. A completed clean or randomized phase
is reused only under the identical fingerprint. A different checkpoint, seed, task selection,
episode count, camera/action setting, or inference setting requires a new run directory.

Video saving is an artifact-only option and defaults off for the 10,000-episode run. It does not
change observations or actions; enable it only for smoke debugging. Raw outputs, logs, videos,
weights, and simulator assets remain ignored by Git.

## Reporting

`paper_reference.csv` is transcribed from Fast-WAM Appendix Table 3 and validated to average
91.88% clean and 91.78% randomized. The paper displays their overall mean as 91.8%. A complete
paper run writes:

```text
summary.csv
summary.json
paper_comparison.csv
paper_comparison.json
paper_comparison.md
paper_comparison_per_task.csv
failed_tasks.txt
resolved_config.yaml
protocol_manifest.json
```

Protocol completion and numerical agreement are separate claims. Do not report a partial task set
as a paper reproduction, and do not infer successes from video filenames when result files exist.
