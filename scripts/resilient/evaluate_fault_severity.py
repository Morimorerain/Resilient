#!/usr/bin/env python3
"""Run one LIBERO task across fault severities and plot latent-metric curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
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

from resilient.fault_detection import analyze_severity_study, available_pair_metrics  # noqa: E402
from resilient.faults import build_fault_pipeline  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("evaluate_results/fault_detection")
DEFAULT_CHECKPOINT = Path("checkpoints/fastwam_release/libero_uncond_2cam224.pt")
DEFAULT_METRICS_CONFIG = Path("configs/fault_detection/latent_v1.yaml")
SUPPORTED_METRICS = frozenset(available_pair_metrics()).union(
    {
        "rcs",
        "residual_matrix",
        "rtc",
        "spearman_rho",
        "fault_score",
        "roc_auc",
    }
)
SUPPORTED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def parse_gpu_ids(raw_value: str) -> tuple[int, ...]:
    """Parse unique physical CUDA IDs used for parallel severity runs."""
    try:
        values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus accepts comma-separated integers.") from error
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--gpus requires unique non-negative IDs, such as 0,1.")
    return values


def finite_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("Severity values must be finite.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-config", type=Path, required=True)
    parser.add_argument("--severities", type=finite_float, nargs="+", required=True)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--gpus", type=parse_gpu_ids, default=(0,))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", type=Path, default=None)
    parser.add_argument("--opsd-adapter", type=Path, default=None)
    parser.add_argument("--opsd-config", type=Path, default=None)
    parser.add_argument("--metrics-config", type=Path, default=DEFAULT_METRICS_CONFIG)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Override metrics-config, for example: --metrics lpe lcd rm rtc.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-bank-manifest", type=Path, default=None)
    parser.add_argument("--training-state-manifest", type=Path, default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--no-auto-clean",
        action="store_true",
        help="Do not add the configured clean severity automatically.",
    )
    return parser


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._") or "value"


def _number_slug(value: float) -> str:
    return ("p" if value >= 0 else "m") + f"{abs(value):g}".replace(".", "p")


def infer_dataset_stats(checkpoint: Path, explicit_path: Path | None) -> Path:
    candidates = []
    if explicit_path is not None:
        candidates.append(resolve_path(explicit_path))
    candidates.extend(
        [
            checkpoint.with_name(f"{checkpoint.stem}_dataset_stats.json"),
            checkpoint.parent / "dataset_stats.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Dataset statistics were not found; pass --dataset-stats.")


def load_metrics_config(path: Path, override: Sequence[str] | None) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Metrics configuration root must be a mapping.")
    metrics = list(config["metrics"] if override is None else override)
    metrics = list(dict.fromkeys(str(metric).strip().lower() for metric in metrics))
    unknown = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(
            f"Unknown metrics {unknown}; supported metrics: {sorted(SUPPORTED_METRICS)}"
        )
    config["metrics"] = metrics
    return config


def load_fault_at_severity(path: Path, severity: float, seed: int) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(resolve_path(path)), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Fault configuration root must be a mapping.")
    pipeline = config.get("pipeline", config)
    if not isinstance(pipeline, dict):
        raise TypeError("Fault pipeline must be a mapping.")
    faults = pipeline.get("faults", [])
    if len(faults) != 1:
        raise ValueError("A severity study requires a fault pipeline with exactly one fault.")
    faults[0]["severity"]["value"] = float(severity)
    faults[0]["severity"].pop("level", None)
    pipeline["seed"] = int(seed)
    build_fault_pipeline(config)
    return config


def build_study_name(
    fault_config: Mapping[str, Any], checkpoint: Path, suite: str, task_id: int
) -> str:
    metadata = build_fault_pipeline(fault_config).metadata()["faults"][0]
    severity = metadata["severity"]
    parts = [metadata["family"], metadata.get("target", "global"), severity["name"], "sweep"]
    fault_name = "-".join(_slug(part) for part in parts)
    return f"{fault_name}__model-{_slug(checkpoint.stem)}__task-{_slug(suite)}-{task_id}"


def _write_manifest(path: Path, configuration: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("configuration") != configuration:
            raise RuntimeError(f"Existing severity study uses different parameters: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_runtime_provenance(gpus: Sequence[int]) -> dict[str, Any]:
    """Capture code and hardware identity without making it part of source configuration."""

    def command_output(command: Sequence[str]) -> str | None:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        output = completed.stdout.strip()
        return output if completed.returncode == 0 and output else None

    return {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "physical_gpu_ids": list(gpus),
        "gpus": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def _run_severities(
    jobs: Sequence[dict[str, Any]], gpus: Sequence[int], runtime_env: Mapping[str, str]
) -> None:
    pending = list(jobs)
    active: dict[int, tuple[subprocess.Popen, Any, dict[str, Any]]] = {}
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                if job["result_path"].is_file() and job["residual_path"].is_file():
                    print(f"Reuse completed severity {job['severity']:g}: {job['output_dir']}")
                    continue
                log_handle = job["log_path"].open("w", encoding="utf-8")
                environment = dict(runtime_env)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    job["command"],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[gpu] = (process, log_handle, job)
                print(f"Started severity {job['severity']:g} on GPU {gpu}: {job['log_path']}")
            if not active:
                continue
            time.sleep(1.0)
            for gpu, (process, log_handle, job) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                del active[gpu]
                if return_code != 0:
                    raise RuntimeError(
                        f"Severity {job['severity']:g} failed on GPU {gpu}; "
                        f"inspect {job['log_path']}."
                    )
                if not job["result_path"].is_file() or not job["residual_path"].is_file():
                    raise FileNotFoundError(
                        f"Severity {job['severity']:g} did not produce the expected metric files."
                    )
                print(f"Completed severity {job['severity']:g}: {job['output_dir']}")
    finally:
        for process, log_handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            log_handle.close()


def main() -> int:
    args = build_parser().parse_args()
    if args.num_trials <= 1:
        raise ValueError("--num-trials must be at least 2 to estimate sample variance.")
    if args.task_id < 0:
        raise ValueError("--task-id must be non-negative.")
    if not math.isfinite(args.sigma_shift):
        raise ValueError("--sigma-shift must be finite.")
    if (args.opsd_adapter is None) != (args.opsd_config is None):
        raise ValueError("--opsd-adapter and --opsd-config must be supplied together.")
    if (args.state_bank_manifest is None) != (args.training_state_manifest is None):
        raise ValueError(
            "--state-bank-manifest and --training-state-manifest must be supplied together."
        )

    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    dataset_stats = infer_dataset_stats(checkpoint, args.dataset_stats)
    metrics_config = load_metrics_config(args.metrics_config, args.metrics)
    metrics = list(metrics_config["metrics"])
    error_bar = str(metrics_config.get("error_bar", "std"))
    if error_bar not in {"std", "sem"}:
        raise ValueError("metrics-config error_bar must be 'std' or 'sem'.")
    bootstrap_samples = int(metrics_config.get("bootstrap_samples", 1000))
    if bootstrap_samples <= 0:
        raise ValueError("metrics-config bootstrap_samples must be positive.")
    clean_severity = float(metrics_config.get("clean_severity", 0.0))
    severities = [float(value) for value in args.severities]
    if len(set(severities)) != len(severities):
        raise ValueError("--severities must not contain duplicates.")
    needs_clean = bool({"fault_score", "roc_auc"}.intersection(metrics))
    auto_clean = bool(metrics_config.get("auto_include_clean", True)) and not args.no_auto_clean
    if needs_clean and not any(
        math.isclose(value, clean_severity, abs_tol=1e-12) for value in severities
    ):
        if not auto_clean:
            raise ValueError(
                "FaultScore/ROC-AUC require the clean severity; include it or enable auto-clean."
            )
        severities.insert(0, clean_severity)
        print(f"Auto-added clean severity {clean_severity:g} for FaultScore and ROC-AUC.")
    if "spearman_rho" in metrics and len(severities) < 2:
        raise ValueError("Spearman rho requires at least two severity values.")

    fault_configs = {
        severity: load_fault_at_severity(args.fault_config, severity, args.seed)
        for severity in severities
    }
    first_pipeline = build_fault_pipeline(fault_configs[severities[0]])
    fault_metadata = first_pipeline.metadata()["faults"][0]
    severity_spec = fault_metadata["severity"]
    output_root = resolve_path(args.output_dir)
    study_dir = output_root / build_study_name(
        fault_configs[severities[0]], checkpoint, args.suite, args.task_id
    )
    study_dir.mkdir(parents=True, exist_ok=True)

    opsd_adapter = resolve_path(args.opsd_adapter) if args.opsd_adapter else None
    opsd_config = resolve_path(args.opsd_config) if args.opsd_config else None
    state_bank = resolve_path(args.state_bank_manifest) if args.state_bank_manifest else None
    training_manifest = (
        resolve_path(args.training_state_manifest) if args.training_state_manifest else None
    )
    for label, path in (
        ("OPSD adapter", opsd_adapter),
        ("OPSD config", opsd_config),
        ("state-bank manifest", state_bank),
        ("training-state manifest", training_manifest),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if state_bank is not None and training_manifest is not None:
        from resilient.state_banks import assert_disjoint_manifests, load_manifest

        assert_disjoint_manifests(load_manifest(state_bank), load_manifest(training_manifest))
    action_horizon = int(metrics_config.get("action_horizon", 32))
    if action_horizon <= 0:
        raise ValueError("metrics-config action_horizon must be positive.")
    configured_pair_metrics = [
        str(value) for value in metrics_config.get("pair_metrics", ["lpe", "lcd", "rm"])
    ]
    unknown_pair_metrics = sorted(set(configured_pair_metrics).difference(available_pair_metrics()))
    if unknown_pair_metrics:
        raise ValueError(
            f"Unknown pair metrics in metrics config: {unknown_pair_metrics}; "
            f"available: {available_pair_metrics()}."
        )
    required_pair_metrics = []
    if {"rm", "spearman_rho", "fault_score", "roc_auc"}.intersection(metrics):
        required_pair_metrics.append("rm")
    pair_metrics = list(
        dict.fromkeys(
            configured_pair_metrics
            + [metric for metric in metrics if metric in set(available_pair_metrics())]
            + required_pair_metrics
        )
    )
    configuration = {
        "fault_config": portable_path(resolve_path(args.fault_config)),
        "fault_family": fault_metadata["family"],
        "severity": {
            "name": severity_spec["name"],
            "unit": severity_spec["unit"],
            "values": severities,
        },
        "suite": args.suite,
        "task_id": args.task_id,
        "num_trials": args.num_trials,
        "gpus": list(args.gpus),
        "checkpoint": portable_path(checkpoint),
        "dataset_stats": portable_path(dataset_stats),
        "opsd_adapter": portable_path(opsd_adapter) if opsd_adapter else None,
        "opsd_config": portable_path(opsd_config) if opsd_config else None,
        "metrics_config": portable_path(resolve_path(args.metrics_config)),
        "metrics": metrics,
        "pair_metrics": pair_metrics,
        "action_horizon": action_horizon,
        "error_bar": error_bar,
        "bootstrap_samples": bootstrap_samples,
        "seed": args.seed,
        "sigma_shift": args.sigma_shift,
        "task_config": args.task_config,
        "save_videos": args.save_videos,
        "state_bank_manifest": portable_path(state_bank) if state_bank else None,
        "training_state_manifest": portable_path(training_manifest) if training_manifest else None,
    }
    jobs = []
    for severity in severities:
        severity_dir = (
            study_dir / f"severity-{_number_slug(severity)}{_slug(severity_spec['unit'])}"
        )
        severity_dir.mkdir(parents=True, exist_ok=True)
        resolved_fault_path = severity_dir / "resolved_fault.yaml"
        OmegaConf.save(OmegaConf.create(fault_configs[severity]), resolved_fault_path, resolve=True)
        command = [
            sys.executable,
            "experiments/libero/eval_libero_single.py",
            "--config-name",
            "sim_libero.yaml",
            f"task={args.task_config}",
            f"ckpt={checkpoint}",
            "gpu_id=0",
            f"EVALUATION.task_suite_name={args.suite}",
            f"EVALUATION.task_id={args.task_id}",
            f"EVALUATION.num_trials={args.num_trials}",
            f"EVALUATION.output_dir={severity_dir}",
            f"EVALUATION.dataset_stats_path={dataset_stats}",
            f"EVALUATION.sigma_shift={args.sigma_shift}",
            f"EVALUATION.fault_config_path={resolved_fault_path}",
            "EVALUATION.fault_detection.enabled=true",
            "EVALUATION.fault_detection.pair_metrics="
            + json.dumps(pair_metrics, separators=(",", ":")),
            "EVALUATION.fault_detection.exclude_conditioning_latent="
            + str(bool(metrics_config.get("exclude_conditioning_latent", True))).lower(),
            f"EVALUATION.action_horizon={action_horizon}",
            f"EVALUATION.replan_steps={action_horizon}",
            f"EVALUATION.save_rollout_videos={str(args.save_videos).lower()}",
            f"seed={args.seed}",
        ]
        if opsd_adapter is not None and opsd_config is not None:
            command.extend(
                [
                    "EVALUATION.opsd_adapter.enabled=true",
                    f"EVALUATION.opsd_adapter.checkpoint={opsd_adapter}",
                    f"EVALUATION.opsd_adapter.training_config={opsd_config}",
                ]
            )
        if state_bank is not None and training_manifest is not None:
            command.extend(
                [
                    "EVALUATION.initial_state_bank.enabled=true",
                    f"EVALUATION.initial_state_bank.manifest={state_bank}",
                    f"EVALUATION.initial_state_bank.training_manifest={training_manifest}",
                ]
            )
        suite_dir = severity_dir / args.suite
        jobs.append(
            {
                "severity": severity,
                "output_dir": severity_dir,
                "command": command,
                "log_path": severity_dir / "evaluation.log",
                "result_path": suite_dir / f"gpu0_task{args.task_id}_results.json",
                "residual_path": suite_dir / f"fault_detection_residuals_task{args.task_id}.npz",
            }
        )
        print(f"Severity {severity:g} command: {shlex.join(command)}")

    configuration["commands"] = [shlex.join(job["command"]) for job in jobs]
    configuration["runtime"] = collect_runtime_provenance(args.gpus)
    _write_manifest(study_dir / "study_manifest.json", configuration)

    runtime_env = os.environ.copy()
    runtime_env.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / "AILOG" / "libero"))
    runtime_env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(PROJECT_ROOT / "checkpoints"))
    runtime_env.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
    runtime_env.setdefault("MUJOCO_GL", "egl")
    _run_severities(jobs, args.gpus, runtime_env)

    analysis_runs = [
        {
            "severity": job["severity"],
            "result_path": job["result_path"],
            "residual_path": job["residual_path"],
        }
        for job in jobs
    ]
    analyze_severity_study(
        analysis_runs,
        output_dir=study_dir,
        metrics=metrics,
        clean_severity=clean_severity,
        error_bar=error_bar,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(metrics_config.get("bootstrap_seed", args.seed)),
        severity_name=str(severity_spec["name"]),
        severity_unit=str(severity_spec["unit"]),
    )
    print(f"Severity study completed: {study_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
