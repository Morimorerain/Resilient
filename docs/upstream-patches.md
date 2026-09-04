# Upstream patch register

FastWAM is pinned in `manifests/upstream.json`. Project-specific behavior should live outside `src/fastwam/` whenever possible.

## LIBERO camera-pose fault hook

- Affected files: `configs/sim_libero.yaml` and
  `experiments/libero/eval_libero_single.py` (`run_single_task` and
  `run_single_episode`).
- Reason: persistent evaluation workers create and reset LIBERO environments inside the
  upstream evaluator, so the fault must be applied after every reset and before the selected
  initial state is rendered. The offset math and validation remain isolated in
  `src/resilient/camera_pose_fault.py`.
- Switch: `EVALUATION.camera_pose_fault.enabled`, default `false`.
- Baseline preservation: when disabled, no applier is constructed and the original
  `env.reset()` -> `env.set_init_state(...)` sequence is unchanged. The existing reproduction
  scripts do not enable the switch.
- Coverage: `tests/test_camera_pose_fault.py` checks the disabled default, quaternion convention,
  non-accumulating reset behavior, output naming, GPU parsing, and weighted result table. Usage
  and coordinate conventions are synchronized in `README.md` and `README.zh-CN.md`.

For every future patch, record:

- affected upstream file and function;
- reason an adapter was insufficient;
- Hydra/CLI switch name and default;
- proof that the switch-off path matches the upstream baseline;
- tests and README sections covering the switch.
