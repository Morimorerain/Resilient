#!/usr/bin/env python3
"""Evaluate FastWAM under a reusable, severity-aware fault configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.faults import build_fault_pipeline  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("evaluate_results/faults")
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUPPORTED_SUITES = (*DEFAULT_SUITES, "libero_90")
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
    "libero_90": "LIBERO-90",
}


def parse_gpu_ids(raw_value: str) -> tuple[int, ...]:
    """Parse unique, comma-separated physical CUDA device IDs."""
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--gpus expects a list such as 0,1,2,3.")
    try:
        result = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus accepts integer device IDs only.") from error
    if any(device < 0 for device in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--gpus requires unique non-negative device IDs.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fault-config",
        type=Path,
        required=True,
        help="Repository-relative path to one reusable fault pipeline YAML file.",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=None,
        help="Override severity.value when the pipeline contains exactly one fault.",
    )
    parser.add_argument("--gpus", type=parse_gpu_ids, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opsd-adapter",
        type=Path,
        default=None,
        help="Optional LoRA adapter.pt to apply after the base Fast-WAM checkpoint.",
    )
    parser.add_argument(
        "--opsd-config",
        type=Path,
        default=None,
        help="Resolved OPSD training YAML defining the adapter layout.",
    )
    parser.add_argument("--dataset-stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suites", nargs="+", choices=SUPPORTED_SUITES, default=list(DEFAULT_SUITES)
    )
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--preview-suite", choices=SUPPORTED_SUITES, default=None)
    parser.add_argument("--preview-task-id", type=int, default=0)
    parser.add_argument("--preview-init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state-bank-manifest",
        type=Path,
        default=None,
        help="Optional unseen-state validation manifest.",
    )
    parser.add_argument(
        "--training-state-manifest",
        type=Path,
        default=None,
        help="Training exposure manifest used by the mandatory leakage guard.",
    )
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Render the comparison and create worker task files without loading a model.",
    )
    return parser


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def load_fault_config(
    path: Path, severity: float | None, seed: int | None = None
) -> dict[str, Any]:
    """Load a resolved pipeline and optionally replace its single severity value."""
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Fault configuration not found: {config_path}")
    raw = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Fault configuration root must be a mapping.")
    pipeline = config.get("pipeline", config)
    if not isinstance(pipeline, dict):
        raise TypeError("Fault pipeline must be a mapping.")
    pipeline["seed"] = int(pipeline.get("seed", 0) if seed is None else seed)
    if severity is not None:
        if not math.isfinite(severity):
            raise ValueError("--severity must be finite.")
        faults = pipeline.get("faults", [])
        if len(faults) != 1:
            raise ValueError("--severity requires a pipeline containing exactly one fault.")
        faults[0]["severity"]["value"] = float(severity)
    build_fault_pipeline(config)
    return config


def infer_dataset_stats(checkpoint: Path, explicit_path: Path | None) -> Path:
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
    raise FileNotFoundError("Could not infer dataset statistics; pass --dataset-stats.")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"


def build_run_name(
    fault_config: Mapping[str, Any],
    checkpoint: Path,
    opsd_adapter: Path | None = None,
) -> str:
    pipeline = build_fault_pipeline(fault_config)
    name = f"{pipeline.slug()}__model-{_slug(checkpoint.stem)}"
    if opsd_adapter is not None:
        adapter_id = opsd_adapter.parent.name or opsd_adapter.stem
        name += f"__opsd-{_slug(adapter_id)}"
    return name


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_or_validate_manifest(run_dir: Path, configuration: dict[str, Any]) -> Path:
    path = run_dir / "fault_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise RuntimeError(f"Existing run uses different parameters: {path}")
        return path
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _annotated_panel(frame: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    panel = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(panel)
    box = draw.multiline_textbbox((0, 0), label, spacing=2)
    padding = 4
    draw.rectangle(
        (0, 0, box[2] - box[0] + 2 * padding, box[3] - box[1] + 2 * padding),
        fill=(0, 0, 0),
    )
    draw.multiline_text((padding, padding), label, fill=(255, 255, 255), spacing=2)
    return panel


def render_comparison_image(
    *,
    output_path: Path,
    fault_config: Mapping[str, Any],
    suite_name: str,
    task_id: int,
    init_state_index: int,
    seed: int,
    state_bank_manifest: Path | None = None,
) -> None:
    """Render nominal and faulted views from one unchanged initial state."""
    import torch
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    from experiments.libero.libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_env,
        get_libero_image,
    )

    suite = benchmark.get_benchmark_dict()[suite_name]()
    if not 0 <= task_id < int(suite.n_tasks):
        raise ValueError(f"Preview task {task_id} is outside suite {suite_name}.")
    task = suite.get_task(task_id)
    if state_bank_manifest is None:
        states_path = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        states = torch.load(states_path, weights_only=False)
    else:
        from resilient.state_banks import load_task_states

        manifest = json.loads(state_bank_manifest.read_text(encoding="utf-8"))
        count = sum(
            record["suite"] == suite_name and int(record["task_id"]) == task_id
            for record in manifest["states"]
        )
        states = load_task_states(
            state_bank_manifest,
            suite=suite_name,
            task_id=task_id,
            expected_count=count,
        )
    if not 0 <= init_state_index < len(states):
        raise ValueError(f"Preview state {init_state_index} is unavailable.")

    pipeline = build_fault_pipeline(fault_config)
    nominal_env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    try:
        nominal_env.reset()
        nominal_obs = nominal_env.set_init_state(states[init_state_index])
        nominal_images = get_libero_image(nominal_obs)
    finally:
        nominal_env.close()

    faulted_env, _ = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        seed,
        fault_pipeline=pipeline,
    )
    try:
        faulted_env.reset()
        faulted_obs = faulted_env.set_init_state(states[init_state_index])
        faulted_images = get_libero_image(faulted_obs)
    finally:
        faulted_env.close()

    label = json.dumps(pipeline.metadata()["faults"], ensure_ascii=True, separators=(",", ":"))
    if len(label) > 160:
        label = label[:157] + "..."
    size = LIBERO_ENV_RESOLUTION
    canvas = Image.new("RGB", (size * 2, size * 2), color=(0, 0, 0))
    panels = (
        _annotated_panel(nominal_images["image"], "Nominal / agentview"),
        _annotated_panel(nominal_images["wrist_image"], "Nominal / wrist"),
        _annotated_panel(faulted_images["image"], f"Fault / agentview\n{label}"),
        _annotated_panel(faulted_images["wrist_image"], f"Fault / wrist\n{label}"),
    )
    for panel, position in zip(panels, ((0, 0), (size, 0), (0, size), (size, size)), strict=True):
        canvas.paste(panel, position)
    canvas.save(output_path)


def build_markdown_summary(summary: dict[str, Any], suites: Sequence[str]) -> str:
    stats_by_suite = summary.get("suite_stats", {})
    lines = ["| Suite | Result |", "| --- | ---: |"]
    total_successes = 0
    total_trials = 0
    for suite in suites:
        stats = stats_by_suite[suite]
        successes = int(stats["total_successes"])
        trials = int(stats["total_trials"])
        percentage = successes / trials * 100
        lines.append(f"| {SUITE_LABELS[suite]} | {successes}/{trials}, {percentage:.1f}% |")
        total_successes += successes
        total_trials += trials
    lines.append(
        f"| **Total** | **{total_successes}/{total_trials}, "
        f"{total_successes / total_trials * 100:.2f}%** |"
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("--suites must not contain duplicates.")
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if (args.opsd_adapter is None) != (args.opsd_config is None):
        raise ValueError("--opsd-adapter and --opsd-config must be provided together.")
    if (args.state_bank_manifest is None) != (args.training_state_manifest is None):
        raise ValueError(
            "--state-bank-manifest and --training-state-manifest must be provided together."
        )
    opsd_adapter = resolve_path(args.opsd_adapter) if args.opsd_adapter else None
    opsd_config = resolve_path(args.opsd_config) if args.opsd_config else None
    if opsd_adapter is not None and not opsd_adapter.is_file():
        raise FileNotFoundError(f"OPSD adapter not found: {opsd_adapter}")
    if opsd_config is not None and not opsd_config.is_file():
        raise FileNotFoundError(f"OPSD config not found: {opsd_config}")
    state_bank_manifest = (
        resolve_path(args.state_bank_manifest) if args.state_bank_manifest else None
    )
    training_state_manifest = (
        resolve_path(args.training_state_manifest) if args.training_state_manifest else None
    )
    if state_bank_manifest is not None:
        from resilient.state_banks import assert_disjoint_manifests, load_manifest

        assert_disjoint_manifests(
            load_manifest(state_bank_manifest), load_manifest(training_state_manifest)
        )
    dataset_stats = infer_dataset_stats(checkpoint, args.dataset_stats)
    fault_config = load_fault_config(args.fault_config, args.severity, args.seed)
    pipeline = build_fault_pipeline(fault_config)
    output_root = resolve_path(args.output_dir)
    run_dir = output_root / build_run_name(fault_config, checkpoint, opsd_adapter)
    run_dir.mkdir(parents=True, exist_ok=True)
    preview_suite = args.preview_suite or args.suites[0]
    configuration = {
        "fault": pipeline.metadata(),
        "fault_config": _portable_path(resolve_path(args.fault_config)),
        "checkpoint": _portable_path(checkpoint),
        "opsd_adapter": _portable_path(opsd_adapter) if opsd_adapter else None,
        "opsd_config": _portable_path(opsd_config) if opsd_config else None,
        "dataset_stats": _portable_path(dataset_stats),
        "gpus": list(args.gpus),
        "suites": list(args.suites),
        "num_trials": args.num_trials,
        "task_config": args.task_config,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
        "create_only": args.create_only,
        "state_bank_manifest": (
            _portable_path(state_bank_manifest) if state_bank_manifest else None
        ),
        "training_state_manifest": (
            _portable_path(training_state_manifest) if training_state_manifest else None
        ),
        "preview": {
            "suite": preview_suite,
            "task_id": args.preview_task_id,
            "init_state": args.preview_init_state,
        },
    }
    manifest_path = _write_or_validate_manifest(run_dir, configuration)
    resolved_fault_path = run_dir / "resolved_fault.yaml"
    OmegaConf.save(OmegaConf.create(fault_config), resolved_fault_path, resolve=True)
    render_comparison_image(
        output_path=run_dir / "fault_comparison.png",
        fault_config=fault_config,
        suite_name=preview_suite,
        task_id=args.preview_task_id,
        init_state_index=args.preview_init_state,
        seed=args.seed,
        state_bank_manifest=state_bank_manifest,
    )

    suites_override = json.dumps(list(args.suites), separators=(",", ":"))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "libero" / "run_libero_manager.py"),
        f"task={args.task_config}",
        f"ckpt={checkpoint}",
        f"EVALUATION.dataset_stats_path={dataset_stats}",
        f"EVALUATION.sigma_shift={args.sigma_shift}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.output_dir={run_dir}",
        f"EVALUATION.fault_config_path={resolved_fault_path}",
        f"seed={args.seed}",
        f"MULTIRUN.task_suite_names={suites_override}",
        f"MULTIRUN.num_gpus={len(args.gpus)}",
        f"MULTIRUN.create_only={str(args.create_only).lower()}",
    ]
    if opsd_adapter is not None and opsd_config is not None:
        command.extend(
            [
                "EVALUATION.opsd_adapter.enabled=true",
                f"EVALUATION.opsd_adapter.checkpoint={opsd_adapter}",
                f"EVALUATION.opsd_adapter.training_config={opsd_config}",
            ]
        )
    if state_bank_manifest is not None and training_state_manifest is not None:
        command.extend(
            [
                "EVALUATION.initial_state_bank.enabled=true",
                f"EVALUATION.initial_state_bank.manifest={state_bank_manifest}",
                f"EVALUATION.initial_state_bank.training_manifest={training_state_manifest}",
            ]
        )
    runtime_env = os.environ.copy()
    runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.gpus)
    runtime_env.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / "AILOG" / "libero"))
    runtime_env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(PROJECT_ROOT / "checkpoints"))
    runtime_env.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
    runtime_env.setdefault("MUJOCO_GL", "egl")
    print(f"Output directory: {run_dir}")
    print(f"Command: {shlex.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, env=runtime_env, check=True)

    if args.create_only:
        print("Create-only validation completed; no result summary was expected.")
        return 0

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce {summary_path}")
    table = build_markdown_summary(
        json.loads(summary_path.read_text(encoding="utf-8")), args.suites
    )
    compact_path = run_dir / "fault_summary.md"
    compact_path.write_text(
        f"# Fault evaluation\n\nManifest: `{manifest_path.name}`\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"\n{table}\n\nSummary: {compact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
