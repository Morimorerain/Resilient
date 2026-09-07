# Fault configuration schema

Faults are ordered, model-independent plugins. Evaluation and OPSD training consume the same
pipeline document.

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

Implement a new plugin by subclassing `resilient.faults.FaultRuntime` and registering its factory
with `register_fault`. Use only the hooks needed by that fault:

- `on_reset` for camera or simulator parameter mutations;
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
each LIBERO control step. The committed example is
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
