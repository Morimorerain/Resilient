#!/usr/bin/env python3
"""Generate reproducible visual and embodiment demonstrations for a Fault catalog."""

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

from experiments.libero.libero_utils import get_libero_env, get_libero_image  # noqa: E402
from resilient.faults import build_fault_pipeline  # noqa: E402
from resilient.visualization import (  # noqa: E402
    compose_comparison_frame,
    compose_visual_fault_grid,
)

DEFAULT_CATALOG = Path("configs/fault/demo_catalog.yaml")
DEFAULT_OUTPUT_ROOT = Path("evaluate_results/fault_catalog")


@dataclass(frozen=True)
class FrameRecord:
    """One rendered embodiment frame with synchronized robot telemetry."""

    image: np.ndarray
    joint_positions_rad: np.ndarray
    eef_position_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--faults",
        nargs="+",
        default=["all"],
        help="Catalog IDs to render, or 'all'.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}.")
    return payload


def _load_task_and_state(entry: dict[str, Any]):
    suite_name = str(entry["suite"])
    benchmark_factory = benchmark.get_benchmark_dict().get(suite_name)
    if benchmark_factory is None:
        raise ValueError(f"Unknown LIBERO suite: {suite_name}")
    suite = benchmark_factory()
    task_id = int(entry["task_id"])
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {suite.n_tasks - 1}].")
    task = suite.get_task(task_id)
    states_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    states = torch.load(states_path, weights_only=False)
    state_index = int(entry["initial_state_index"])
    if not 0 <= state_index < len(states):
        raise ValueError(f"initial_state_index must be in [0, {len(states) - 1}].")
    return task, states[state_index]


def _fault_label(pipeline: Any) -> str:
    faults = pipeline.metadata()["faults"]
    families = {str(fault["family"]).split(".")[-1] for fault in faults}
    severities = {
        (float(fault["severity"]["value"]), str(fault["severity"]["unit"]))
        for fault in faults
    }
    family_label = "+".join(sorted(families))
    severity_names = {str(fault["severity"]["name"]) for fault in faults}
    severity_label = "+".join(
        f"{value:g} {unit}" for value, unit in sorted(severities)
    )
    return f"{family_label} | {'+'.join(sorted(severity_names))}={severity_label}"


def _output_dir(root: Path, entry: dict[str, Any], pipeline: Any) -> Path:
    scenario = (
        f"{entry['suite']}_task{entry['task_id']}_state{entry['initial_state_index']}"
    )
    return root / str(entry["id"]) / pipeline.slug() / scenario


def _render_visual(
    entry: dict[str, Any],
    fault_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task, initial_state = _load_task_and_state(entry)
    resolution = int(entry["resolution"])
    seed = int(entry["environment_seed"])
    pipeline = build_fault_pipeline(fault_config)
    nominal_env, description = get_libero_env(task, resolution, seed)
    try:
        nominal_env.reset()
        nominal_images = get_libero_image(nominal_env.set_init_state(initial_state))
    finally:
        nominal_env.close()
    fault_env, _ = get_libero_env(task, resolution, seed, fault_pipeline=pipeline)
    try:
        fault_env.reset()
        faulted_images = get_libero_image(fault_env.set_init_state(initial_state))
    finally:
        fault_env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "nominal_vs_fault.png"
    grid = compose_visual_fault_grid(
        nominal_images,
        faulted_images,
        fault_label=_fault_label(pipeline),
        title=f"Fault catalog | {entry['id']}",
    )
    imageio.imwrite(image_path, grid)
    return {
        "kind": "visual",
        "artifact": image_path.name,
        "task_description": description,
        "fault": pipeline.metadata(),
    }


def _build_actions(entry: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    actions: list[np.ndarray] = []
    phases: list[str] = []
    action_dimension = int(entry.get("action_dimension", 7))
    for phase in entry.get("phases", []):
        action = np.asarray(phase["action"], dtype=np.float64)
        if action.shape != (action_dimension,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"Phase {phase.get('name')} must define a finite "
                f"{action_dimension}D action."
            )
        steps = int(phase["steps"])
        if steps < 0:
            raise ValueError("Phase steps must be non-negative.")
        actions.extend(action.copy() for _ in range(steps))
        phases.extend(str(phase["name"]) for _ in range(steps))
    if not actions:
        raise ValueError("Embodiment demonstrations require at least one action.")
    return actions, phases


def _capture_record(env: Any, observation: dict[str, Any], entry: dict[str, Any]) -> FrameRecord:
    robot = env.env.robots[0]
    joint_positions = np.asarray(
        env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64
    ).copy()
    image = env.sim.render(
        camera_name=str(entry["render_camera"]),
        width=int(entry["resolution"]),
        height=int(entry["resolution"]),
    )[::-1]
    return FrameRecord(
        image=np.ascontiguousarray(image),
        joint_positions_rad=joint_positions,
        eef_position_m=np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
    )


def _run_trajectory(
    entry: dict[str, Any],
    task: Any,
    initial_state: Any,
    actions: list[np.ndarray],
    fault_config: dict[str, Any] | None,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    pipeline = build_fault_pipeline(fault_config)
    env, description = get_libero_env(
        task,
        int(entry["resolution"]),
        int(entry["environment_seed"]),
        fault_pipeline=pipeline,
        controller=str(entry.get("controller", "OSC_POSE")),
    )
    records: list[FrameRecord] = []
    try:
        robot = env.env.robots[0]
        if int(robot.action_dim) != len(actions[0]):
            raise ValueError(
                f"Controller {robot.controller.name} expects {robot.action_dim}D actions, "
                f"but the catalog defines {len(actions[0])}D."
            )
        env.reset()
        observation = env.set_init_state(initial_state)
        records.append(_capture_record(env, observation, entry))
        for action in actions:
            observation, _, _, _ = env.step(action)
            records.append(_capture_record(env, observation, entry))
        return records, {
            "task_description": description,
            "controller": robot.controller.name,
            "control_frequency_hz": int(robot.control_freq),
            "action_dimension": int(robot.action_dim),
            "joint_names": list(robot.robot_joints),
            "fault": pipeline.metadata(),
        }
    finally:
        env.close()


def _render_embodiment(
    entry: dict[str, Any],
    fault_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task, initial_state = _load_task_and_state(entry)
    actions, phases = _build_actions(entry)
    clean_records, clean_meta = _run_trajectory(entry, task, initial_state, actions, None)
    fault_records, fault_meta = _run_trajectory(
        entry, task, initial_state, actions, fault_config
    )
    if len(clean_records) != len(fault_records):
        raise RuntimeError("Clean and fault trajectories are not time-aligned.")
    pipeline = build_fault_pipeline(fault_config)
    target_joint = pipeline.metadata()["faults"][0]["targets"][0]
    joint_index = fault_meta["joint_names"].index(target_joint)
    other_joint_indices = [
        index for index in range(len(fault_meta["joint_names"])) if index != joint_index
    ]
    fps = int(entry["fps"])
    control_hz = int(clean_meta["control_frequency_hz"])
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "nominal_vs_fault.mp4"
    initial_clean = clean_records[0]
    initial_fault = fault_records[0]
    with imageio.get_writer(
        video_path, fps=fps, codec="libx264", quality=8, macro_block_size=2
    ) as writer:
        for frame_index, (clean, fault) in enumerate(
            zip(clean_records, fault_records, strict=True)
        ):
            action_index = min(frame_index, len(phases) - 1)
            eef_gap_mm = 1000.0 * np.linalg.norm(clean.eef_position_m - fault.eef_position_m)
            clean_delta = np.rad2deg(
                clean.joint_positions_rad[joint_index]
                - initial_clean.joint_positions_rad[joint_index]
            )
            fault_delta = np.rad2deg(
                fault.joint_positions_rad[joint_index]
                - initial_fault.joint_positions_rad[joint_index]
            )
            clean_other_delta = np.rad2deg(
                np.max(
                    np.abs(
                        clean.joint_positions_rad[other_joint_indices]
                        - initial_clean.joint_positions_rad[other_joint_indices]
                    )
                )
            )
            fault_other_delta = np.rad2deg(
                np.max(
                    np.abs(
                        fault.joint_positions_rad[other_joint_indices]
                        - initial_fault.joint_positions_rad[other_joint_indices]
                    )
                )
            )
            frame = compose_comparison_frame(
                clean.image,
                fault.image,
                title=(
                    f"{entry['id']} | same state | {clean_meta['controller']} "
                    f"joint1-only commands"
                ),
                clean_label="NOMINAL SIMULATOR",
                fault_label=f"FAULT SIMULATOR | {_fault_label(pipeline)}",
                clean_telemetry={
                    "phase": phases[action_index],
                    "arm command": "joint1 only",
                    f"{target_joint} delta (deg)": float(clean_delta),
                    "other joints max delta (deg)": float(clean_other_delta),
                },
                fault_telemetry={
                    "phase": phases[action_index],
                    "arm command": "joint1 only",
                    f"{target_joint} delta (deg)": float(fault_delta),
                    "other joints max delta (deg)": float(fault_other_delta),
                    "EEF gap from nominal (mm)": float(eef_gap_mm),
                },
                timestamp_seconds=frame_index / control_hz,
            )
            writer.append_data(frame)
    final_clean = clean_records[-1]
    final_fault = fault_records[-1]
    eef_gaps = np.asarray(
        [
            np.linalg.norm(clean.eef_position_m - fault.eef_position_m)
            for clean, fault in zip(clean_records, fault_records, strict=True)
        ]
    )
    joint_gaps_deg = np.rad2deg(
        np.asarray(
            [
                abs(
                    clean.joint_positions_rad[joint_index]
                    - fault.joint_positions_rad[joint_index]
                )
                for clean, fault in zip(clean_records, fault_records, strict=True)
            ]
        )
    )
    max_gap_frame = int(np.argmax(eef_gaps))
    nominal_other_joint_motion = np.asarray(
        [
            np.max(
                np.abs(
                    record.joint_positions_rad[other_joint_indices]
                    - initial_clean.joint_positions_rad[other_joint_indices]
                )
            )
            for record in clean_records
        ]
    )
    fault_other_joint_motion = np.asarray(
        [
            np.max(
                np.abs(
                    record.joint_positions_rad[other_joint_indices]
                    - initial_fault.joint_positions_rad[other_joint_indices]
                )
            )
            for record in fault_records
        ]
    )
    return {
        "kind": "embodiment",
        "artifact": video_path.name,
        "task_description": clean_meta["task_description"],
        "controller": clean_meta["controller"],
        "control_frequency_hz": control_hz,
        "action_dimension": clean_meta["action_dimension"],
        "num_environment_steps": len(actions),
        "fault": fault_meta["fault"],
        "comparison": {
            "target_joint": target_joint,
            "nominal_final_joint_delta_deg": float(
                np.rad2deg(
                    final_clean.joint_positions_rad[joint_index]
                    - initial_clean.joint_positions_rad[joint_index]
                )
            ),
            "fault_final_joint_delta_deg": float(
                np.rad2deg(
                    final_fault.joint_positions_rad[joint_index]
                    - initial_fault.joint_positions_rad[joint_index]
                )
            ),
            "final_eef_gap_m": float(
                np.linalg.norm(final_clean.eef_position_m - final_fault.eef_position_m)
            ),
            "max_eef_gap_m": float(eef_gaps[max_gap_frame]),
            "max_eef_gap_time_s": float(max_gap_frame / control_hz),
            "max_joint_position_gap_deg": float(np.max(joint_gaps_deg)),
            "max_nominal_other_joint_delta_deg": float(
                np.rad2deg(np.max(nominal_other_joint_motion))
            ),
            "max_fault_other_joint_delta_deg": float(
                np.rad2deg(np.max(fault_other_joint_motion))
            ),
        },
    }


def main() -> None:
    args = parse_args()
    catalog_path = _resolve_project_path(args.catalog)
    catalog = _load_mapping(catalog_path)
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported fault demonstration catalog schema.")
    defaults = dict(catalog.get("defaults", {}))
    entries = [{**defaults, **dict(item)} for item in catalog.get("entries", [])]
    requested = set(args.faults)
    selected = entries if requested == {"all"} else [x for x in entries if x["id"] in requested]
    missing = requested - {str(entry["id"]) for entry in selected} - {"all"}
    if missing:
        raise ValueError(f"Unknown catalog fault IDs: {sorted(missing)}")
    output_root = _resolve_project_path(args.output_root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "catalog": str(catalog_path.relative_to(PROJECT_ROOT)),
        "entries": [],
    }
    for entry in selected:
        config_path = _resolve_project_path(entry["fault_config"])
        fault_config = _load_mapping(config_path)
        pipeline = build_fault_pipeline(fault_config)
        if not pipeline.enabled:
            raise ValueError(f"Catalog entry {entry['id']} has no enabled fault.")
        output_dir = _output_dir(output_root, entry, pipeline)
        if entry["kind"] == "visual":
            summary = _render_visual(entry, fault_config, output_dir)
        elif entry["kind"] == "embodiment":
            summary = _render_embodiment(entry, fault_config, output_dir)
        else:
            raise ValueError(f"Unsupported demonstration kind: {entry['kind']}")
        summary.update(
            {
                "id": entry["id"],
                "fault_config": str(config_path.relative_to(PROJECT_ROOT)),
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["entries"].append(summary)
        print(f"[{entry['id']}] {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    if len(selected) == len(entries):
        manifest_name = "catalog_manifest.json"
    else:
        selected_slug = "__".join(str(entry["id"]) for entry in selected)
        manifest_name = f"catalog_manifest__{selected_slug}.json"
    manifest_path = output_root / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Catalog manifest: {manifest_path}")


if __name__ == "__main__":
    main()
