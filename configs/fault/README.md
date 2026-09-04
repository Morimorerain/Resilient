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
