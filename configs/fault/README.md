# Fault configuration schema

Faults are ordered, model-independent plugins. Evaluation and OPSD training consume the same
pipeline document. An enabled pipeline is installed when the environment is created, and a
transparent `FaultedEnvironment` owns all hooks so policy code uses only the standard environment
API. Multiple faults compose in their YAML order.

```yaml
pipeline:
  id: stable_experiment_name
  enabled: true
  seed: 42
  faults:
    - id: unique_component_name
      family: visual.camera_pose
      enabled: true
      target: robot0_eye_in_hand
      operation: {type: local_axis_rotation, axis: z, direction: positive}
      severity: {name: rotation_angle, value: 30.0, unit: degree, level: severe}
      parameters: {position_offset: [0.0, 0.0, 0.0]}
```

`family` selects a registered implementation. `id` identifies one component inside a compound
pipeline. `severity` is the resolved physical magnitude; configurations at 10, 20, and 30 degrees
retain the same family/operation while producing different manifests and output slugs. A compound
fault keeps one severity record per component instead of inventing a dimensionless global score.

## Implemented catalog

The ten severe, demonstration-oriented pipelines are under `catalog/`; they are normal pipeline
files and can be passed directly to evaluation or training. `demo_catalog.yaml` only adds the
task, initial state, render settings, controller, and deterministic action phases used to visualize
them. Embodiment entries use `JOINT_POSITION` with only the target joint command nonzero.

| Family | Required parameters | Exact environment semantics |
| --- | --- | --- |
| `visual.camera_pose` | one `target`; rotation operation or `position_offset` | Changes MuJoCo `cam_quat`/`cam_pos`; restored on suspend/detach |
| `visual.defocus_blur` | `targets`; sigma severity in pixels | Gaussian blur at the environment camera-sensor boundary |
| `visual.local_occlusion` | normalized round `spots`, RGB color, feather radius | Composites reproducible soft-edged ink spots; severity is the largest spot diameter |
| `visual.illumination` | optional 3x3 `color_matrix` and 3-vector `color_bias` | Computes `clip(I A^T + b, 0, 255)` before policy observation |
| `structure.joint_motion` | `targets`; registered motion law | Replaces selected realized displacement and velocity after each MuJoCo step |
| `structure.joint_position_bias` | `bias_deg` per target | Introduces a fixed offset once after a state load; it never accumulates per step |
| `structure.joint_backlash` | `gap_deg` per target | On reversal, consumes displacement until the configured gap is exhausted |
| `structure.joint_range_limit` | absolute `lower_deg`, `upper_deg` per target | Clips realized qpos and zeros velocity at a bound |
| `structure.periodic_joint_freeze` | `period_deg`, `phase_deg`, `hold_steps` | Crossing `phase + k*period` holds qpos and zeros qvel |

The position-bias implementation treats the loaded state as the command-coordinate reference,
then introduces the configured physical zero offset on the first realized dynamics step. The
offset is not added again on later steps, so it cannot drift linearly with episode length. A
closed-loop controller may subsequently compensate for part of the physical error.

Periodic freeze is angle-triggered, not time-triggered. Defective gear positions are

```text
q_bad(k) = phase_deg + k * period_deg,  k in integers
```

When one environment step crosses a previously untriggered `q_bad(k)`, the joint returns to its
pre-step angle and its velocity becomes zero for `hold_steps`. It is then allowed to pass that
tooth and rearms after moving `release_fraction * period_deg` beyond it. The same tooth can trigger
again after the joint moves away and later reverses across it. This models periodic sticking at
defective gear positions rather than a wall-clock on/off schedule.

All stateful transmission variables (`initialized`, backlash direction/remaining gap, and periodic
freeze countdown/tooth/direction) participate in `state_dict()` and `load_state_dict()`. Copy both
the simulator state and `FaultedEnvironment.fault_runtime_state_dict()` when creating shadow
rollouts.

Generate the complete catalog demonstration with:

```bash
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:third_party/LIBERO:. \
python scripts/resilient/demonstrate_fault_catalog.py --faults all
```

The default output is `evaluate_results/fault_catalog/` and is intentionally ignored by Git.

Implement a new plugin by subclassing `resilient.faults.FaultRuntime` and registering its factory
with `register_fault`. Use only the hooks needed by that fault; hooks are invoked by the
environment, not the policy:

- `attach` for simulator-startup installation;
- `on_reset` for camera or simulator parameter mutations;
- `on_state_loaded` for state-dependent refresh after loading an initial state;
- `transform_observation` for contamination, occlusion, noise, or sensor dropout;
- `transform_action` for actuator commands, limits, delay, or quantization;
- `before_step`/`after_step` for robot structure and dynamics behavior;
- `suspend` to expose privileged nominal information without advancing the environment;
- `state_dict`/`load_state_dict` for deterministic stateful faults;
- `detach` to restore environment-owned state.

Factories must validate target names, units, ranges, and operation-specific severity semantics.
Random plugins must derive randomness from the supplied seed, episode index, and step index and
must store any additional mutable state. Add tests and synchronize both root README files whenever
a new public fault family or parameter is added.

## Joint-motion degradation

`structure.joint_motion` reduces the displacement achieved by selected scalar robot joints during
each environment dynamics step. It is installed with `FaultedEnvironment` when the simulator is
constructed, independently of the controller or policy. The committed example is
`configs/fault/structure/panda_joint1_half_motion.yaml`:

```yaml
family: structure.joint_motion
targets: [robot0_joint1]
operation: {type: proportional}
severity: {name: motion_retention, value: 0.5, unit: ratio}
```

`targets` accepts one or more MuJoCo joint names. A proportional retention of `0.5` applies
`q_after = q_before + 0.5 * (q_nominal_after - q_before)` and scales the selected joint velocity by
the same factor. This is a deterministic kinematic motion-retention fault, not an actuator-torque
efficiency model. Closed-loop OSC may command compensating motion on later steps, so total motion
over a trajectory need not be exactly half of the clean total. New deterministic non-linear laws
can be added through `register_joint_motion_law` without changing the fault pipeline or consumers.
