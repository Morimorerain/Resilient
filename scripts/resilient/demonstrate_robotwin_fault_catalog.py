#!/usr/bin/env python3
"""Generate annotated RoboTwin demonstrations for the shared ten-Fault catalog."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
for search_path in (SRC_ROOT, ROBOTWIN_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from resilient.robotwin.faults import (  # noqa: E402
    RoboTwinFaultController,
    install_fault_pipeline,
)
from resilient.visualization import (  # noqa: E402
    compose_comparison_frame,
    compose_multi_camera_fault_grid,
)

DEFAULT_CATALOG = Path("configs/fault/robotwin/demo_catalog.yaml")
DEFAULT_OUTPUT_ROOT = Path("evaluate_results/robotwin/fault_catalog")


@dataclass(frozen=True)
class FrameRecord:
    """One rendered frame and physical telemetry for an affected arm."""

    image: np.ndarray
    joint_position_rad: float
    end_effector_position_m: np.ndarray
    phase: str
    commanded_delta_deg: float
    fault_state: str


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Minimal deterministic state needed by isolated joint-motion demonstrations."""

    entity_states: tuple[tuple[Any, np.ndarray, np.ndarray], ...]
    left_arm_qpos: np.ndarray
    right_arm_qpos: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--faults", nargs="+", default=["all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}.")
    return payload


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return payload


def _task_kwargs(entry: dict[str, Any]) -> dict[str, Any]:
    task_config = str(entry.get("task_config", "demo_clean"))
    args = _yaml(ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml")
    embodiment_types = args["embodiment"]
    embodiment_index = _yaml(ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml")

    def embodiment_path(name: str) -> Path:
        configured = Path(str(embodiment_index[name]["file_path"]))
        return (ROBOTWIN_ROOT / configured).resolve()

    if len(embodiment_types) == 1:
        left_path = right_path = embodiment_path(str(embodiment_types[0]))
        args["dual_arm_embodied"] = True
    elif len(embodiment_types) == 3:
        left_path = embodiment_path(str(embodiment_types[0]))
        right_path = embodiment_path(str(embodiment_types[1]))
        args["embodiment_dis"] = embodiment_types[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("RoboTwin embodiment must contain one or three entries.")
    args.update(
        {
            "task_name": str(entry["task_name"]),
            "task_config": task_config,
            "now_ep_num": 0,
            "seed": int(entry["seed"]),
            "is_test": True,
            "eval_mode": False,
            "save_data": False,
            "render_freq": 0,
            "left_robot_file": str(left_path),
            "right_robot_file": str(right_path),
            "left_embodiment_config": _yaml(left_path / "config.yml"),
            "right_embodiment_config": _yaml(right_path / "config.yml"),
        }
    )
    return args


def _create_task(entry: dict[str, Any]) -> Any:
    task_name = str(entry["task_name"])
    module = importlib.import_module(f"envs.{task_name}")
    task = getattr(module, task_name)()
    task.setup_demo(**_task_kwargs(entry))
    return task


def _camera_frames(observation: dict[str, Any], order: list[str]) -> dict[str, np.ndarray]:
    sensors = observation["observation"]
    return {
        name: np.ascontiguousarray(np.asarray(sensors[name]["rgb"], dtype=np.uint8))
        for name in order
    }


def _fault_label(controller: RoboTwinFaultController) -> str:
    faults = controller.metadata()["faults"]
    families = "+".join(sorted({str(item["family"]).split(".")[-1] for item in faults}))
    severities = "+".join(
        sorted(
            {f"{float(item['severity']['value']):g} {item['severity']['unit']}" for item in faults}
        )
    )
    return f"{families} | severity={severities}"


def _materialize_arm_config(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be left or right.")
    resolved = copy.deepcopy(config)
    pipeline = resolved.get("pipeline", resolved)
    pipeline["id"] = str(pipeline.get("id", "robotwin_fault")).replace("left", arm)
    for fault in pipeline.get("faults", []):
        if not str(fault.get("family", "")).startswith("structure."):
            continue
        fault["id"] = str(fault.get("id", "joint_fault")).replace("left", arm)
        fault["targets"] = [f"{arm}_arm_joint1"]
        fault.pop("target", None)
    return resolved


def _unique_entities(task: Any) -> tuple[Any, ...]:
    entities = (task.robot.left_entity, task.robot.right_entity)
    unique: list[Any] = []
    for entity in entities:
        if all(entity is not existing for existing in unique):
            unique.append(entity)
    return tuple(unique)


def _arm_real_qpos(task: Any, arm: str) -> np.ndarray:
    robot = task.robot
    entity = getattr(robot, f"{arm}_entity")
    active = list(entity.get_active_joints())
    arm_joints = list(getattr(robot, f"{arm}_arm_joints"))
    indices = [active.index(joint) for joint in arm_joints]
    return np.asarray(entity.get_qpos(), dtype=np.float64)[indices].copy()


def _capture_snapshot(task: Any) -> SimulatorSnapshot:
    states = []
    for entity in _unique_entities(task):
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)
        states.append(
            (
                entity,
                np.asarray(entity.get_qpos(), dtype=np.float64).copy(),
                qvel.copy(),
            )
        )
    return SimulatorSnapshot(
        entity_states=tuple(states),
        left_arm_qpos=_arm_real_qpos(task, "left"),
        right_arm_qpos=_arm_real_qpos(task, "right"),
    )


def _restore_snapshot(task: Any, snapshot: SimulatorSnapshot) -> None:
    for entity, qpos, qvel in snapshot.entity_states:
        entity.set_qpos(qpos.copy())
        entity.set_qvel(qvel.copy())
    task.robot.set_arm_joints(snapshot.left_arm_qpos, np.zeros_like(snapshot.left_arm_qpos), "left")
    task.robot.set_arm_joints(
        snapshot.right_arm_qpos,
        np.zeros_like(snapshot.right_arm_qpos),
        "right",
    )
    for _ in range(20):
        task.scene.step()
    task._update_render()


def _embodiment_frame(task: Any, camera_name: str) -> np.ndarray:
    task._update_render()
    if camera_name == "observer_camera":
        camera = task.cameras.observer_camera
    elif camera_name in {"world_camera1", "world_camera2"}:
        camera = getattr(task.cameras, camera_name)
    else:
        camera = next(
            (
                candidate
                for name, candidate in zip(
                    task.cameras.static_camera_name,
                    task.cameras.static_camera_list,
                    strict=True,
                )
                if name == camera_name
            ),
            None,
        )
    if camera is None:
        raise RuntimeError(f"RoboTwin demonstration camera {camera_name!r} is not active.")
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return np.ascontiguousarray((rgba[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8))


def _phase_targets(entry: dict[str, Any]) -> tuple[list[float], list[str]]:
    targets: list[float] = []
    phases: list[str] = []
    previous = 0.0
    for phase in entry["phases"]:
        frames = int(phase["frames"])
        target = float(phase["target_deg"])
        if frames <= 0:
            raise ValueError("Every embodiment phase must contain positive frames.")
        values = np.linspace(previous, target, frames + 1, dtype=np.float64)[1:]
        targets.extend(values.tolist())
        phases.extend([str(phase["name"])] * frames)
        previous = target
    return targets, phases


def _summarize_fault_diagnostics(items: list[dict[str, Any]]) -> str:
    """Summarize actuator state accumulated during one rendered video frame."""
    if not items:
        return "not installed"
    family = str(items[-1].get("family", "unknown")).rsplit(".", 1)[-1]
    if family == "joint_motion":
        nominal = np.asarray(items[-1].get("nominal_increment_deg", [0.0]))
        applied = np.asarray(items[-1].get("applied_increment_deg", [0.0]))
        return f"dnom/dapp={float(nominal[0]):+.3f}/{float(applied[0]):+.3f}deg"
    if family == "joint_position_bias":
        bias = np.asarray(items[-1].get("bias_deg", [0.0]))
        return f"bias={float(bias[0]):+.1f}deg"
    if family == "joint_backlash":
        reversed_now = any(any(item.get("direction_reversal", [])) for item in items)
        remaining = max(float(np.max(item.get("remaining_gap_deg", [0.0]))) for item in items)
        return f"rev={reversed_now}, gap={remaining:.2f}deg"
    if family == "joint_range_limit":
        clipped = any(any(item.get("clipped", [])) for item in items)
        return f"clipped={clipped}"
    if family == "periodic_joint_freeze":
        triggered = any(any(item.get("triggered", [])) for item in items)
        remaining = max(int(np.max(item.get("remaining_hold_steps", [0]))) for item in items)
        return f"trigger={triggered}, hold={remaining}"
    return family


def _run_arm_motion(
    task: Any,
    entry: dict[str, Any],
    snapshot: SimulatorSnapshot,
    arm: str,
) -> list[FrameRecord]:
    start = snapshot.left_arm_qpos if arm == "left" else snapshot.right_arm_qpos
    targets_deg, phases = _phase_targets(entry)
    substeps = int(entry["physics_steps_per_frame"])
    if substeps <= 0:
        raise ValueError("physics_steps_per_frame must be positive.")
    camera_name = str(entry.get("embodiment_camera", "observer_camera"))
    controller = getattr(task, "_resilient_fault_controller", None)
    records = [
        FrameRecord(
            image=_embodiment_frame(task, camera_name),
            joint_position_rad=float(_arm_real_qpos(task, arm)[0]),
            end_effector_position_m=np.asarray(
                getattr(task.robot, f"get_{arm}_ee_pose")()[:3], dtype=np.float64
            ),
            phase="initial",
            commanded_delta_deg=0.0,
            fault_state="not installed" if controller is None else "awaiting command",
        )
    ]
    previous_deg = 0.0
    for target_deg, phase in zip(targets_deg, phases, strict=True):
        diagnostics_start = 0 if controller is None else len(controller.last_joint_diagnostics)
        for substep in range(substeps):
            fraction = (substep + 1) / substeps
            sub_target_deg = previous_deg + fraction * (target_deg - previous_deg)
            position = start.copy()
            position[0] = start[0] + np.deg2rad(sub_target_deg)
            task.robot.set_arm_joints(position, np.zeros_like(position), arm)
            task.scene.step()
        frame_diagnostics = (
            [] if controller is None else controller.last_joint_diagnostics[diagnostics_start:]
        )
        records.append(
            FrameRecord(
                image=_embodiment_frame(task, camera_name),
                joint_position_rad=float(_arm_real_qpos(task, arm)[0]),
                end_effector_position_m=np.asarray(
                    getattr(task.robot, f"get_{arm}_ee_pose")()[:3], dtype=np.float64
                ),
                phase=phase,
                commanded_delta_deg=float(target_deg),
                fault_state=_summarize_fault_diagnostics(frame_diagnostics),
            )
        )
        previous_deg = target_deg
    return records


def _render_visual(
    entry: dict[str, Any],
    fault_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task = _create_task(entry)
    controller = None
    try:
        camera_order = [str(name) for name in entry["camera_order"]]
        nominal = _camera_frames(task.get_obs(), camera_order)
        controller = install_fault_pipeline(task, fault_config)
        if controller is None:
            raise RuntimeError("Visual demonstration Fault pipeline is disabled.")
        faulted = _camera_frames(task.get_obs(), camera_order)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "nominal_vs_fault.png"
        imageio.imwrite(
            artifact,
            compose_multi_camera_fault_grid(
                nominal,
                faulted,
                camera_order=camera_order,
                fault_label=_fault_label(controller),
                title=f"RoboTwin Fault catalog | {entry['id']}",
            ),
        )
        return {
            "kind": "visual",
            "artifact": artifact.name,
            "cameras": camera_order,
            "fault": controller.metadata(),
        }
    finally:
        if controller is not None:
            controller.detach()
        task.close_env(clear_cache=False)
        del task
        gc.collect()


def _write_arm_video(
    entry: dict[str, Any],
    arm: str,
    controller: RoboTwinFaultController,
    clean: list[FrameRecord],
    faulted: list[FrameRecord],
    output_dir: Path,
) -> dict[str, Any]:
    if len(clean) != len(faulted):
        raise RuntimeError("Nominal and Fault RoboTwin trajectories are not time-aligned.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "nominal_vs_fault.mp4"
    fps = int(entry["fps"])
    initial_clean = clean[0]
    initial_fault = faulted[0]
    joint_gaps: list[float] = []
    eef_gaps: list[float] = []
    with imageio.get_writer(
        artifact,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for index, (normal, damaged) in enumerate(zip(clean, faulted, strict=True)):
            clean_delta = np.rad2deg(normal.joint_position_rad - initial_clean.joint_position_rad)
            fault_delta = np.rad2deg(damaged.joint_position_rad - initial_fault.joint_position_rad)
            joint_gap = abs(np.rad2deg(normal.joint_position_rad - damaged.joint_position_rad))
            eef_gap = 1000.0 * np.linalg.norm(
                normal.end_effector_position_m - damaged.end_effector_position_m
            )
            joint_gaps.append(float(joint_gap))
            eef_gaps.append(float(eef_gap))
            writer.append_data(
                compose_comparison_frame(
                    normal.image,
                    damaged.image,
                    title=f"RoboTwin {entry['id']} | affected arm: {arm}",
                    clean_label="NOMINAL SAPIEN SIMULATOR",
                    fault_label=f"FAULT SAPIEN SIMULATOR | {_fault_label(controller)}",
                    clean_telemetry={
                        "phase": normal.phase,
                        "moving arm": arm,
                        "commanded joint1 delta (deg)": normal.commanded_delta_deg,
                        "joint1 delta (deg)": float(clean_delta),
                    },
                    fault_telemetry={
                        "phase": damaged.phase,
                        "affected arm": arm,
                        "commanded joint1 delta (deg)": damaged.commanded_delta_deg,
                        "joint1 delta (deg)": float(fault_delta),
                        "joint1 gap (deg)": float(joint_gap),
                        "EEF gap (mm)": float(eef_gap),
                        "state": damaged.fault_state,
                    },
                    timestamp_seconds=index / fps,
                )
            )
    return {
        "arm": arm,
        "artifact": artifact.name,
        "num_frames": len(clean),
        "max_joint1_gap_deg": max(joint_gaps),
        "max_eef_gap_mm": max(eef_gaps),
        "fault": controller.metadata(),
    }


def _render_embodiment(
    entry: dict[str, Any],
    fault_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task = _create_task(entry)
    snapshot = _capture_snapshot(task)
    variants = []
    try:
        for arm in entry["arm_variants"]:
            arm = str(arm)
            _restore_snapshot(task, snapshot)
            clean = _run_arm_motion(task, entry, snapshot, arm)
            _restore_snapshot(task, snapshot)
            controller = install_fault_pipeline(
                task,
                _materialize_arm_config(fault_config, arm),
            )
            if controller is None:
                raise RuntimeError("Embodiment demonstration Fault pipeline is disabled.")
            try:
                faulted = _run_arm_motion(task, entry, snapshot, arm)
                variants.append(
                    _write_arm_video(
                        entry,
                        arm,
                        controller,
                        clean,
                        faulted,
                        output_dir / f"{arm}_arm",
                    )
                )
            finally:
                controller.detach()
        return {
            "kind": "embodiment",
            "camera": str(entry.get("embodiment_camera", "observer_camera")),
            "arm_variants": variants,
        }
    finally:
        task.close_env(clear_cache=False)
        del task
        gc.collect()


def main() -> None:
    args = parse_args()
    catalog_path = _project_path(args.catalog)
    catalog = _load_mapping(catalog_path)
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported RoboTwin demonstration catalog schema.")
    defaults = dict(catalog.get("defaults", {}))
    entries = [{**defaults, **dict(entry)} for entry in catalog.get("entries", [])]
    requested = set(args.faults)
    selected = entries if requested == {"all"} else [x for x in entries if x["id"] in requested]
    missing = requested - {str(entry["id"]) for entry in selected} - {"all"}
    if missing:
        raise ValueError(f"Unknown RoboTwin catalog Fault IDs: {sorted(missing)}")
    output_root = _project_path(args.output_root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "simulator": "robotwin_sapien",
        "catalog": str(catalog_path.relative_to(PROJECT_ROOT)),
        "entries": [],
    }
    previous_cwd = Path.cwd()
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda:0")
    try:
        os.chdir(ROBOTWIN_ROOT)
        for entry in selected:
            config_path = _project_path(entry["fault_config"])
            fault_config = _load_mapping(config_path)
            output_dir = output_root / str(entry["id"])
            if entry["kind"] == "visual":
                summary = _render_visual(entry, fault_config, output_dir)
            elif entry["kind"] == "embodiment":
                summary = _render_embodiment(entry, fault_config, output_dir)
            else:
                raise ValueError(f"Unsupported demonstration kind: {entry['kind']}")
            summary.update(
                {
                    "id": entry["id"],
                    "task_name": entry["task_name"],
                    "seed": int(entry["seed"]),
                    "fault_config": str(config_path.relative_to(PROJECT_ROOT)),
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["entries"].append(summary)
            print(f"[{entry['id']}] {output_dir}", flush=True)
    finally:
        os.chdir(previous_cwd)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_slug = (
        "all"
        if len(selected) == len(entries)
        else "__".join(str(entry["id"]) for entry in selected)
    )
    manifest_path = output_root / f"catalog_manifest__{selected_slug}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Catalog manifest: {manifest_path}")

    summary_paths = [output_root / str(entry["id"]) / "summary.json" for entry in entries]
    if all(path.is_file() for path in summary_paths):
        complete_manifest = {
            "schema_version": 1,
            "simulator": "robotwin_sapien",
            "catalog": str(catalog_path.relative_to(PROJECT_ROOT)),
            "entries": [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths],
        }
        complete_path = output_root / "catalog_manifest__all.json"
        complete_path.write_text(
            json.dumps(complete_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if complete_path != manifest_path:
            print(f"Refreshed complete catalog manifest: {complete_path}")


if __name__ == "__main__":
    main()
