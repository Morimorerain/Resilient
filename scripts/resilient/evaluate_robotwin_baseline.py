#!/usr/bin/env python3
"""Launch the pinned FastWAM RoboTwin smoke, coverage, or paper protocol."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resilient.robotwin.preflight import run_preflight  # noqa: E402
from resilient.robotwin.reporting import (  # noqa: E402
    build_paper_comparison,
    write_paper_comparison,
)

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "fastwam_release" / "robotwin_uncond_3cam_384.pt"
DATASET_STATS = (
    PROJECT_ROOT
    / "checkpoints"
    / "fastwam_release"
    / "robotwin_uncond_3cam_384_dataset_stats.json"
)
REFERENCE = PROJECT_ROOT / "reproduce" / "fastwam_robotwin" / "paper_reference.csv"


def _parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be a unique comma-separated non-negative list"
        )
    return gpu_ids


def _busy_compute_processes(gpu_ids: list[int]) -> list[tuple[int, str, str]]:
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuid_to_index = {}
    for row in csv.reader(gpu_query.stdout.splitlines()):
        if len(row) >= 2:
            uuid_to_index[row[1].strip()] = int(row[0].strip())
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    busy = []
    for row in csv.reader(process_query.stdout.splitlines()):
        if len(row) < 3:
            continue
        index = uuid_to_index.get(row[0].strip())
        if index in gpu_ids:
            busy.append((index, row[1].strip(), row[2].strip()))
    return busy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "coverage", "paper"), default="smoke")
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, required=True)
    parser.add_argument("--task", default="click_alarmclock")
    parser.add_argument("--run-id")
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_tasks_per_gpu <= 0:
        parser.error("--max-tasks-per-gpu must be positive")
    preflight = run_preflight(PROJECT_ROOT, verify_hashes=False, require_runtime=True)
    if not preflight["passed"]:
        for check in preflight["checks"]:
            if not check["passed"]:
                print(f"ERROR {check['name']}: {check['detail']}", file=sys.stderr)
        return 1

    busy = _busy_compute_processes(args.gpu_ids)
    if busy and not args.allow_busy_gpus:
        for gpu_id, pid, memory in busy:
            print(f"ERROR GPU {gpu_id} has compute PID {pid} using {memory} MiB", file=sys.stderr)
        print(
            "Refusing to share a busy GPU; choose free IDs or pass --allow-busy-gpus.",
            file=sys.stderr,
        )
        return 1

    episodes = 100 if args.mode == "paper" else 1
    task_name = args.task if args.mode == "smoke" else None
    run_id = args.run_id or f"fastwam_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    gpu_override = "[" + ",".join(str(gpu_id) for gpu_id in args.gpu_ids) + "]"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "robotwin" / "run_robotwin_manager.py"),
        "task=robotwin_uncond_3cam_384_1e-4",
        f"ckpt={CHECKPOINT}",
        f"EVALUATION.dataset_stats_path={DATASET_STATS}",
        "EVALUATION.instruction_type=unseen",
        f"EVALUATION.eval_num_episodes={episodes}",
        "EVALUATION.action_horizon=32",
        "EVALUATION.replan_steps=24",
        "EVALUATION.num_inference_steps=10",
        "EVALUATION.sigma_shift=5.0",
        "EVALUATION.text_cfg_scale=1.0",
        "EVALUATION.rand_device=cpu",
        "EVALUATION.skip_get_obs_within_replan=true",
        f"EVALUATION.save_videos={str(args.save_videos).lower()}",
        f"EVALUATION.output_dir=evaluate_results/robotwin/{run_id}",
        f"MULTIRUN.num_gpus={len(args.gpu_ids)}",
        f"MULTIRUN.gpu_ids={gpu_override}",
        f"MULTIRUN.max_tasks_per_gpu={args.max_tasks_per_gpu}",
        "MULTIRUN.resume_completed=true",
    ]
    if task_name is not None:
        command.append(f"EVALUATION.task_name={task_name}")

    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    process = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if process.returncode != 0:
        return process.returncode

    if args.mode == "paper":
        output_dir = (
            PROJECT_ROOT
            / "evaluate_results"
            / "robotwin"
            / CHECKPOINT.stem
            / run_id
        )
        report = build_paper_comparison(
            output_dir / "summary.json",
            REFERENCE,
            episodes_per_condition=episodes,
        )
        write_paper_comparison(report, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
