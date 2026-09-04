# Upstream patch register

FastWAM is pinned in `manifests/upstream.json`. Project-specific behavior should live outside `src/fastwam/` whenever possible.

## Generic LIBERO fault lifecycle hook

- Affected files: `configs/sim_libero.yaml` and `experiments/libero/eval_libero_single.py`
  (`run_single_task` and `run_single_episode`).
- Reason: persistent evaluation workers own resets and steps, so observation, action,
  environment, and dynamics plugins need lifecycle calls at those exact boundaries. All fault
  implementations remain under `src/resilient/faults/`.
- Switch: `EVALUATION.fault.pipeline.enabled`, default `false`.
- Baseline preservation: the disabled branch executes the original `env.reset()`,
  `env.set_init_state(...)`, and `env.step(...)` calls directly. It does not copy observations,
  transform actions, or invoke plugin hooks. Baseline reproduction scripts leave it disabled.
- Coverage: `tests/test_faults.py` checks the disabled no-op, generic observation/action extension
  points, severity identity, quaternion convention, reversible/non-accumulating camera state,
  canonical names, GPU parsing, and weighted result tables. Both root README files document it.

## Optional Fast-WAM Student denoising trace

- Affected file: `src/fastwam/models/wan22/fastwam.py`, method `FastWAM.infer_action`.
- Reason: OPSD must evaluate the Teacher at the Student's own pre-step latents, timesteps, and
  scheduler deltas. These values exist only inside the upstream sampling loop; an external
  adapter cannot recover them exactly from the final action.
- Switch: Python argument `return_denoising_trace`, default `false`.
- Baseline preservation: the default loop, scheduler calls, returned CPU action, and result key
  set remain unchanged. When enabled, the method additionally clones detached pre-step values
  into `denoising_trace`; it never changes the sampled latent.
- Coverage: `tests/test_opsd.py` asserts the switch default. Runtime validation compares each
  captured timestep/delta against `infer_action_scheduler.build_inference_schedule`. Both README
  files document the switch and its exact Fast-WAM-derived 10-step setting.

## Optional OPSD LoRA evaluation loader

- Affected files: `configs/sim_libero.yaml` and
  `experiments/libero/eval_libero_single.py` (`eval_single_process`).
- Reason: evaluation workers construct and own the Fast-WAM model, so a LoRA-only OPSD
  checkpoint must be injected after the unchanged base checkpoint is loaded and before inference.
- Switch: `EVALUATION.opsd_adapter.enabled`, default `false`; enabling it also requires an adapter
  checkpoint and the resolved OPSD training config that defines the exact LoRA layout.
- Baseline preservation: the disabled branch returns before importing PEFT, injecting modules, or
  loading adapter weights. Existing Fast-WAM evaluation commands therefore retain their original
  model structure and checkpoint path.
- Coverage: `tests/test_opsd.py` checks strict adapter save/load round-tripping. Both README files
  document the paired CLI arguments and base-plus-adapter loading order.

For every future patch, record:

- affected upstream file and function;
- reason an adapter was insufficient;
- Hydra/CLI switch name and default;
- proof that the switch-off path matches the upstream baseline;
- tests and README sections covering the switch.
