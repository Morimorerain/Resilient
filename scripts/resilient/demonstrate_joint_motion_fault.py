#!/usr/bin/env python3
"""Render a synchronized LIBERO OSC_POSE joint-motion fault comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402

from experiments.libero.libero_utils import (  # noqa: E402
    capture_libero_observation,
    get_libero_env,
)
from resilient.faults import FaultTransition, build_fault_pipeline  # noqa: E402
from resilient.visualization import compose_comparison_frame  # noqa: E402

DEFAULT_FAULT_CONFIG = Path("configs/fault/structure/panda_joint1_half_motion.yaml")
DEFAULT_OUTPUT_ROOT = Path("evaluate_results/fault_demos")


@dataclass(frozen=True)
class FrameRecord:
    """One rendered frame and aligned robot telemetry sample."""

    image: np.ndarray
    joint_positions_rad: np.ndarray
    eef_position_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-config", type=Path, default=DEFAULT_FAULT_CONFIG)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--render-camera",
        default="frontview",
        help="LIBERO camera used for the side-by-side visualization.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--motion-steps", type=int, default=125)
    parser.add_argument("--settle-steps", type=int, default=15)
    parser.add_argument(
        "--osc-command",
        type=float,
        nargs=6,
        default=(-0.3, 0.4, 0.05, 0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        help="Normalized OSC_POSE command repeated during the motion phase.",
    )
    parser.add_argument("--gripper", type=float, default=-1.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _fresh_observation(env: Any) -> dict[str, Any]:
    observation = capture_libero_observation(env)
    if not isinstance(observation, dict):
        raise TypeError("LIBERO demonstration expects a dictionary observation.")
    return observation


def _capture_record(
    env: Any,
    observation: dict[str, Any],
    *,
    render_camera: str,
    resolution: int,
) -> FrameRecord:
    robot = env.env.robots[0]
    joint_positions = np.asarray(
        env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64
    ).copy()
    image = env.sim.render(
        camera_name=render_camera,
        width=resolution,
        height=resolution,
    )[::-1]
    return FrameRecord(
        image=np.ascontiguousarray(image),
        joint_positions_rad=joint_positions,
        eef_position_m=np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
    )


def _build_actions(args: argparse.Namespace) -> list[np.ndarray]:
    hold = np.asarray([0.0] * 6 + [args.gripper], dtype=np.float64)
    motion = np.asarray([*args.osc_command, args.gripper], dtype=np.float64)
    return (
        [hold.copy() for _ in range(args.warmup_steps)]
        + [motion.copy() for _ in range(args.motion_steps)]
        + [hold.copy() for _ in range(args.settle_steps)]
    )


def _load_task_and_state(args: argparse.Namespace):
    benchmark_factory = benchmark.get_benchmark_dict().get(args.suite)
    if benchmark_factory is None:
        raise ValueError(f"Unknown LIBERO suite: {args.suite}")
    suite = benchmark_factory()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks - 1}].")
    task = suite.get_task(args.task_id)
    states_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    states = torch.load(states_path, weights_only=False)
    if not 0 <= args.initial_state_index < len(states):
        raise ValueError(f"initial-state-index must be in [0, {len(states) - 1}].")
    return task, states[args.initial_state_index]


def _run_trajectory(
    *,
    task: Any,
    initial_state: Any,
    actions: list[np.ndarray],
    args: argparse.Namespace,
    fault_config: dict[str, Any] | None,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    env, task_description = get_libero_env(task, args.resolution, args.environment_seed)
    pipeline = build_fault_pipeline(fault_config)
    records: list[FrameRecord] = []
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        if not isinstance(observation, dict):
            raise TypeError("LIBERO demonstration expects a dictionary observation.")
        if pipeline.enabled:
            pipeline.on_reset(env, pipeline.context(episode_index=0, step_index=0))
            observation = pipeline.transform_observation(
                observation, env, pipeline.context(episode_index=0, step_index=0)
            )
        records.append(
            _capture_record(
                env,
                observation,
                render_camera=args.render_camera,
                resolution=args.resolution,
            )
        )

        for step_index, policy_action in enumerate(actions):
            context = pipeline.context(episode_index=0, step_index=step_index)
            executed_action = pipeline.transform_action(policy_action, env, context)
            pipeline.before_step(env, executed_action, context)
            next_raw, reward, done, info = env.step(executed_action)
            next_observation = pipeline.transform_observation(next_raw, env, context)
            transition = FaultTransition(
                raw_observation=observation,
                student_observation=observation,
                policy_action=policy_action,
                executed_action=executed_action,
                next_raw_observation=next_raw,
                next_student_observation=next_observation,
                reward=float(reward),
                done=bool(done),
                info={} if info is None else dict(info),
                fault_metadata=pipeline.metadata()["faults"],
            )
            pipeline.after_step(env, transition, context)
            if pipeline.requires_post_step_observation_refresh:
                next_raw = _fresh_observation(env)
                next_observation = pipeline.transform_observation(next_raw, env, context)
            observation = next_observation
            records.append(
                _capture_record(
                    env,
                    observation,
                    render_camera=args.render_camera,
                    resolution=args.resolution,
                )
            )

        robot = env.env.robots[0]
        controller_name = robot.controller.name
        metadata = {
            "task_description": task_description,
            "controller": controller_name,
            "control_frequency_hz": int(robot.control_freq),
            "joint_names": list(robot.robot_joints),
            "fault": pipeline.metadata(),
        }
        return records, metadata
    finally:
        pipeline.detach(env)
        env.close()


def _round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def main() -> None:
    args = parse_args()
    if args.resolution <= 0 or args.fps <= 0:
        raise ValueError("resolution and fps must be positive.")
    if min(args.warmup_steps, args.motion_steps, args.settle_steps) < 0:
        raise ValueError("trajectory phase lengths must be non-negative.")
    fault_path = (PROJECT_ROOT / args.fault_config).resolve()
    fault_config = OmegaConf.to_container(OmegaConf.load(fault_path), resolve=True)
    if not isinstance(fault_config, dict):
        raise TypeError("Fault configuration root must be a mapping.")
    fault_pipeline = build_fault_pipeline(fault_config)
    if not fault_pipeline.enabled or len(fault_pipeline.faults) != 1:
        raise ValueError("Demonstration requires exactly one enabled joint-motion fault.")
    if fault_pipeline.faults[0].family != "structure.joint_motion":
        raise ValueError("Demonstration fault must use family structure.joint_motion.")

    task, initial_state = _load_task_and_state(args)
    actions = _build_actions(args)
    clean_records, clean_metadata = _run_trajectory(
        task=task,
        initial_state=initial_state,
        actions=actions,
        args=args,
        fault_config=None,
    )
    fault_records, fault_metadata = _run_trajectory(
        task=task,
        initial_state=initial_state,
        actions=actions,
        args=args,
        fault_config=fault_config,
    )
    if len(clean_records) != len(fault_records):
        raise RuntimeError("Clean and fault trajectories are not time-aligned.")

    scenario = f"{args.suite}_task{args.task_id}_state{args.initial_state_index}"
    output_dir = (PROJECT_ROOT / args.output_root / fault_pipeline.slug() / scenario).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "clean_vs_fault.mp4"
    target_joint = fault_pipeline.metadata()["faults"][0]["targets"][0]
    retention_ratio = float(
        fault_pipeline.metadata()["faults"][0]["operation"]["retention_ratio"]
    )
    joint_index = fault_metadata["joint_names"].index(target_joint)
    initial_clean = clean_records[0]
    initial_fault = fault_records[0]

    with imageio.get_writer(
        video_path,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for frame_index, (clean, fault) in enumerate(
            zip(clean_records, fault_records, strict=True)
        ):
            clean_q_delta = np.rad2deg(
                clean.joint_positions_rad[joint_index]
                - initial_clean.joint_positions_rad[joint_index]
            )
            fault_q_delta = np.rad2deg(
                fault.joint_positions_rad[joint_index]
                - initial_fault.joint_positions_rad[joint_index]
            )
            eef_gap_mm = 1000.0 * np.linalg.norm(clean.eef_position_m - fault.eef_position_m)
            frame = compose_comparison_frame(
                clean.image,
                fault.image,
                title="LIBERO OSC_POSE: joint-motion degradation",
                clean_label="CLEAN | same command sequence",
                fault_label=(
                    f"FAULT | {target_joint} | motion retained: {retention_ratio:.0%}"
                ),
                clean_telemetry={
                    f"{target_joint} delta (deg)": float(clean_q_delta),
                    "EEF x/y/z (m)": "/".join(f"{v:.3f}" for v in clean.eef_position_m),
                },
                fault_telemetry={
                    f"{target_joint} delta (deg)": float(fault_q_delta),
                    "EEF x/y/z (m)": "/".join(f"{v:.3f}" for v in fault.eef_position_m),
                    "EEF gap from clean (mm)": float(eef_gap_mm),
                },
                timestamp_seconds=frame_index / args.fps,
            )
            writer.append_data(frame)

    final_clean = clean_records[-1]
    final_fault = fault_records[-1]
    summary = {
        "schema_version": 1,
        "task": {
            "suite": args.suite,
            "task_id": args.task_id,
            "description": clean_metadata["task_description"],
            "initial_state_index": args.initial_state_index,
            "environment_seed": args.environment_seed,
        },
        "controller": {
            "name": clean_metadata["controller"],
            "control_frequency_hz": clean_metadata["control_frequency_hz"],
        },
        "trajectory": {
            "warmup_steps": args.warmup_steps,
            "motion_steps": args.motion_steps,
            "settle_steps": args.settle_steps,
            "normalized_osc_pose_command": [*args.osc_command, args.gripper],
            "render_camera": args.render_camera,
            "num_video_frames": len(clean_records),
            "fps": args.fps,
        },
        "fault": fault_metadata["fault"],
        "comparison": {
            "target_joint": target_joint,
            "clean_joint_delta_deg": float(
                np.rad2deg(
                    final_clean.joint_positions_rad[joint_index]
                    - initial_clean.joint_positions_rad[joint_index]
                )
            ),
            "fault_joint_delta_deg": float(
                np.rad2deg(
                    final_fault.joint_positions_rad[joint_index]
                    - initial_fault.joint_positions_rad[joint_index]
                )
            ),
            "clean_eef_start_m": _round_list(initial_clean.eef_position_m),
            "clean_eef_end_m": _round_list(final_clean.eef_position_m),
            "fault_eef_start_m": _round_list(initial_fault.eef_position_m),
            "fault_eef_end_m": _round_list(final_fault.eef_position_m),
            "final_eef_gap_m": float(
                np.linalg.norm(final_clean.eef_position_m - final_fault.eef_position_m)
            ),
        },
        "artifacts": {"video": video_path.name},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Video: {video_path}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
