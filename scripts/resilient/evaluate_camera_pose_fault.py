#!/usr/bin/env python3
"""Evaluate FastWAM under a deterministic LIBERO camera-pose fault."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resilient.camera_pose_fault import (  # noqa: E402
    SUPPORTED_CAMERAS,
    CameraPoseFaultApplier,
    CameraPoseFaultSpec,
)

DEFAULT_OUTPUT_DIR = Path("evaluate_results/camera_pose_faults")
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUPPORTED_SUITES = (*DEFAULT_SUITES, "libero_90")
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
    "libero_90": "LIBERO-90",
}


def _triplet(values: Sequence[float], option: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise argparse.ArgumentTypeError(f"{option} requires three finite values.")
    return result


def parse_gpu_ids(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of physical CUDA device IDs."""
    raw_parts = [part.strip() for part in raw_value.split(",")]
    if not raw_parts or any(not part for part in raw_parts):
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list such as 0,1,2,3.")
    try:
        gpu_ids = tuple(int(part) for part in raw_parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus accepts integer CUDA device IDs only.") from error
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("--gpus device IDs must be non-negative.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpus must not contain duplicate device IDs.")
    return gpu_ids


def build_parser() -> argparse.ArgumentParser:
    """Build the camera-fault evaluation argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the FastWAM LIBERO benchmark after applying a deterministic position and "
            "orientation offset to one camera."
        )
    )
    parser.add_argument("--camera", choices=SUPPORTED_CAMERAS, required=True)
    parser.add_argument(
        "--position-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Position offset in meters along the original local camera X/Y/Z axes.",
    )
    parser.add_argument(
        "--rotation-offset-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Right-handed local camera X/Y/Z rotations in degrees, applied in X-Y-Z order.",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        required=True,
        metavar="IDS",
        help="Physical CUDA device IDs, for example 0,1,2,3 or 0,1,2,3,4,5,6,7.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="FastWAM checkpoint path, relative to the repository or absolute.",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=None,
        help=(
            "Dataset-statistics JSON. By default, infer <checkpoint_stem>_dataset_stats.json "
            "or dataset_stats.json next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output root. A standardized camera/offset/model subdirectory is created below it "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUPPORTED_SUITES,
        default=list(DEFAULT_SUITES),
        help="LIBERO suites to evaluate (default: Spatial, Object, Goal, and Long).",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="Episodes per task.")
    parser.add_argument(
        "--preview-suite",
        choices=SUPPORTED_SUITES,
        default=None,
        help="Suite used for the comparison image (default: first entry in --suites).",
    )
    parser.add_argument(
        "--preview-task-id",
        type=int,
        default=0,
        help="Task ID used for the comparison image (default: 0).",
    )
    parser.add_argument(
        "--preview-init-state",
        type=int,
        default=0,
        help="Initial-state index used for the comparison image (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="LIBERO evaluation seed.")
    parser.add_argument(
        "--task-config",
        default="libero_uncond_2cam224_1e-4",
        help="Hydra task configuration used to instantiate the FastWAM model.",
    )
    parser.add_argument(
        "--sigma-shift",
        type=float,
        default=5.0,
        help="Diffusion sigma shift used for evaluation (default: 5.0).",
    )
    return parser


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def infer_dataset_stats(checkpoint: Path, explicit_path: Path | None) -> Path:
    """Resolve the dataset statistics associated with a checkpoint."""
    if explicit_path is not None:
        result = resolve_path(explicit_path)
        if not result.is_file():
            raise FileNotFoundError(f"Dataset statistics not found: {result}")
        return result

    candidates = (
        checkpoint.with_name(f"{checkpoint.stem}_dataset_stats.json"),
        checkpoint.parent / "dataset_stats.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not infer dataset statistics. Pass --dataset-stats explicitly. Tried:\n"
        f"{formatted}"
    )


def _encode_number(value: float, precision: int) -> str:
    threshold = 0.5 * 10 ** (-precision)
    normalized = 0.0 if abs(value) < threshold else value
    sign = "p" if normalized >= 0.0 else "m"
    magnitude = f"{abs(normalized):.{precision}f}".replace(".", "p")
    return f"{sign}{magnitude}"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized or "model"


def build_run_name(
    camera: str,
    position_offset: Sequence[float],
    rotation_offset_deg: Sequence[float],
    checkpoint: Path,
) -> str:
    """Build a deterministic, filesystem-safe camera-fault run name."""
    px, py, pz = _triplet(position_offset, "position_offset")
    rx, ry, rz = _triplet(rotation_offset_deg, "rotation_offset_deg")
    return (
        f"{camera}"
        f"-pos_x{_encode_number(px, 3)}_y{_encode_number(py, 3)}_z{_encode_number(pz, 3)}m"
        f"-rot_x{_encode_number(rx, 1)}_y{_encode_number(ry, 1)}_z{_encode_number(rz, 1)}deg"
        f"-model_{_slug(checkpoint.stem)}"
    )


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_or_validate_manifest(run_dir: Path, configuration: dict[str, Any]) -> Path:
    manifest_path = run_dir / "camera_fault_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise RuntimeError(
                f"Existing run has different parameters: {manifest_path}. "
                "Choose another --output-dir or remove the old ignored result directory."
            )
        print(f"Resuming matching run: {run_dir}")
        return manifest_path

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _format_fault_label(spec: CameraPoseFaultSpec) -> str:
    px, py, pz = spec.position_offset
    rx, ry, rz = spec.rotation_offset_deg
    return (
        f"Fault / {spec.camera}\n"
        "pos local xyz (m):\n"
        f"x={px:+.3f} y={py:+.3f} z={pz:+.3f}\n"
        "rot local xyz (deg):\n"
        f"x={rx:+.1f} y={ry:+.1f} z={rz:+.1f}"
    )


def _annotated_panel(frame: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    panel = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(panel)
    text_box = draw.multiline_textbbox((0, 0), label, spacing=2)
    padding = 4
    width = text_box[2] - text_box[0] + padding * 2
    height = text_box[3] - text_box[1] + padding * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.multiline_text((padding, padding), label, fill=(255, 255, 255), spacing=2)
    return panel


def render_comparison_image(
    *,
    output_path: Path,
    spec: CameraPoseFaultSpec,
    suite_name: str,
    task_id: int,
    init_state_index: int,
    seed: int,
) -> None:
    """Render original and faulted views from one deterministic initial state."""
    import torch
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    from experiments.libero.libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_env,
        get_libero_image,
    )

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    if not 0 <= task_id < int(task_suite.n_tasks):
        raise ValueError(
            f"--preview-task-id={task_id} is outside suite {suite_name} "
            f"range [0, {int(task_suite.n_tasks) - 1}]."
        )
    task = task_suite.get_task(task_id)
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    initial_states = torch.load(init_states_path, weights_only=False)
    if not 0 <= init_state_index < len(initial_states):
        raise ValueError(
            f"--preview-init-state={init_state_index} is outside [0, {len(initial_states) - 1}]."
        )

    env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    fault_applier = CameraPoseFaultApplier(spec)
    try:
        env.reset()
        original_observation = env.set_init_state(initial_states[init_state_index])
        original_images = get_libero_image(original_observation)

        env.reset()
        fault_applier.apply(env)
        fault_observation = env.set_init_state(initial_states[init_state_index])
        fault_images = get_libero_image(fault_observation)
    finally:
        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            close_fn()

    panel_size = LIBERO_ENV_RESOLUTION
    canvas = Image.new("RGB", (panel_size * 2, panel_size * 2), color=(0, 0, 0))
    panels = (
        _annotated_panel(original_images["image"], "Original / agentview"),
        _annotated_panel(original_images["wrist_image"], "Original / robot0_eye_in_hand"),
        _annotated_panel(
            fault_images["image"],
            _format_fault_label(spec) if spec.camera == "agentview" else "Fault / agentview",
        ),
        _annotated_panel(
            fault_images["wrist_image"],
            _format_fault_label(spec)
            if spec.camera == "robot0_eye_in_hand"
            else "Fault / robot0_eye_in_hand",
        ),
    )
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panel_size, 0))
    canvas.paste(panels[2], (0, panel_size))
    canvas.paste(panels[3], (panel_size, panel_size))
    canvas.save(output_path)


def build_markdown_summary(summary: dict[str, Any], suites: Sequence[str]) -> str:
    """Build the compact suite and overall success-rate table."""
    suite_stats = summary.get("suite_stats", {})
    lines = ["| Suite | Result |", "| --- | ---: |"]
    total_successes = 0
    total_trials = 0
    for suite in suites:
        if suite not in suite_stats:
            raise KeyError(f"Evaluation summary does not contain requested suite: {suite}")
        stats = suite_stats[suite]
        successes = int(stats["total_successes"])
        trials = int(stats["total_trials"])
        if trials <= 0:
            raise ValueError(f"Suite {suite} has no completed trials.")
        rate = successes / trials * 100.0
        lines.append(f"| {SUITE_LABELS[suite]} | {successes}/{trials}, {rate:.1f}% |")
        total_successes += successes
        total_trials += trials

    total_rate = total_successes / total_trials * 100.0
    lines.append(f"| **Total** | **{total_successes}/{total_trials}, {total_rate:.2f}%** |")
    return "\n".join(lines)


def main() -> int:
    """Validate arguments, launch the multi-GPU evaluation, and summarize it."""
    args = build_parser().parse_args()
    position_offset = _triplet(args.position_offset, "--position-offset")
    rotation_offset_deg = _triplet(args.rotation_offset_deg, "--rotation-offset-deg")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if not math.isfinite(args.sigma_shift):
        raise ValueError("--sigma-shift must be finite.")
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("--suites must not contain duplicates.")
    if args.preview_task_id < 0:
        raise ValueError("--preview-task-id must be non-negative.")
    if args.preview_init_state < 0:
        raise ValueError("--preview-init-state must be non-negative.")

    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    dataset_stats = infer_dataset_stats(checkpoint, args.dataset_stats)
    output_root = resolve_path(args.output_dir)
    run_name = build_run_name(
        args.camera,
        position_offset,
        rotation_offset_deg,
        checkpoint,
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fault_spec = CameraPoseFaultSpec(
        enabled=True,
        camera=args.camera,
        position_offset=position_offset,
        rotation_offset_deg=rotation_offset_deg,
    )
    preview_suite = args.preview_suite or args.suites[0]
    configuration = {
        "fault": fault_spec.to_dict(),
        "checkpoint": _portable_path(checkpoint),
        "dataset_stats": _portable_path(dataset_stats),
        "gpus": list(args.gpus),
        "suites": list(args.suites),
        "num_trials": args.num_trials,
        "task_config": args.task_config,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
        "preview": {
            "suite": preview_suite,
            "task_id": args.preview_task_id,
            "init_state": args.preview_init_state,
        },
    }
    manifest_path = _write_or_validate_manifest(run_dir, configuration)

    position_override = json.dumps(list(position_offset), separators=(",", ":"))
    rotation_override = json.dumps(list(rotation_offset_deg), separators=(",", ":"))
    suite_override = json.dumps(list(args.suites), separators=(",", ":"))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "libero" / "run_libero_manager.py"),
        f"task={args.task_config}",
        f"ckpt={checkpoint}",
        f"EVALUATION.dataset_stats_path={dataset_stats}",
        f"EVALUATION.sigma_shift={args.sigma_shift}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.output_dir={run_dir}",
        f"seed={args.seed}",
        f"MULTIRUN.task_suite_names={suite_override}",
        f"MULTIRUN.num_gpus={len(args.gpus)}",
        "EVALUATION.camera_pose_fault.enabled=true",
        f"EVALUATION.camera_pose_fault.camera={args.camera}",
        f"EVALUATION.camera_pose_fault.position_offset={position_override}",
        f"EVALUATION.camera_pose_fault.rotation_offset_deg={rotation_override}",
    ]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.gpus)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / "AILOG" / "libero"))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(PROJECT_ROOT / "checkpoints"))
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
    os.environ.setdefault("MUJOCO_GL", "egl")
    runtime_env = os.environ.copy()

    comparison_path = run_dir / "camera_pose_comparison.png"
    render_comparison_image(
        output_path=comparison_path,
        spec=fault_spec,
        suite_name=preview_suite,
        task_id=args.preview_task_id,
        init_state_index=args.preview_init_state,
        seed=args.seed,
    )

    print(f"Run name: {run_name}")
    print(f"Output directory: {run_dir}")
    print(f"GPUs: {runtime_env['CUDA_VISIBLE_DEVICES']}")
    print(f"Comparison image: {comparison_path}")
    print(f"Command: {shlex.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, env=runtime_env, check=True)

    upstream_summary_path = run_dir / "summary.json"
    if not upstream_summary_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce {upstream_summary_path}")
    upstream_summary = json.loads(upstream_summary_path.read_text(encoding="utf-8"))
    result_table = build_markdown_summary(upstream_summary, args.suites)
    compact_summary_path = run_dir / "camera_fault_summary.md"
    compact_summary_path.write_text(
        "# Camera-pose fault evaluation\n\n"
        f"Run: `{run_name}`\n\n"
        f"Manifest: `{manifest_path.name}`\n\n"
        f"{result_table}\n",
        encoding="utf-8",
    )

    print("\n=== Camera-pose fault result ===\n")
    print(result_table)
    print(f"\nCompact summary: {compact_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
