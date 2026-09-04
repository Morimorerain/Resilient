#!/usr/bin/env python3
"""Generate an unseen LIBERO validation state bank for a completed OPSD run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_env,
)
from resilient.state_banks import (  # noqa: E402
    SCHEMA_VERSION,
    assert_disjoint_manifests,
    state_fingerprint,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def derive_seed(base_seed: int, *identity: object) -> int:
    encoded = ":".join([str(base_seed), *(str(value) for value in identity)]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states-per-task", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=104729)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_training_seeds(run_dir: Path) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for path in sorted(run_dir.glob("metrics_rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (record["suite"], int(record["task_id"]), int(record["initial_state_index"]))
            result[key] = int(record["environment_seed"])
    if not result:
        raise ValueError(f"No OPSD metrics with training state provenance found under {run_dir}")
    return result


def generate_task(payload: tuple) -> tuple[str, int, list[dict], list[dict], np.ndarray]:
    """Generate one task in an isolated process because MuJoCo resets are CPU-heavy."""
    suite_name, task_id, count, base_seed, training_seeds, excluded_seeds = payload
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    official_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    official_states = torch.load(official_path, map_location="cpu", weights_only=False)
    if len(official_states) < count:
        raise ValueError(f"{suite_name} task {task_id} has fewer than {count} official states.")
    training_records = []
    excluded_hashes = set()
    for state_index, state in enumerate(official_states[:count]):
        fingerprint = state_fingerprint(state)
        excluded_hashes.add(fingerprint)
        training_records.append(
            {
                "suite": suite_name,
                "task_id": task_id,
                "state_index": state_index,
                "generation_seed": training_seeds[(suite_name, task_id, state_index)],
                "state_sha256": fingerprint,
                "source": "libero_official",
            }
        )

    env, _ = get_libero_env(task, LIBERO_ENV_RESOLUTION, base_seed)
    generated: list[np.ndarray] = []
    generated_hashes: set[str] = set()
    validation_records: list[dict] = []
    try:
        candidate = 0
        while len(generated) < count:
            seed = derive_seed(base_seed, "validation", suite_name, task_id, candidate)
            candidate += 1
            if seed in excluded_seeds:
                continue
            env.seed(seed)
            env.reset()
            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            fingerprint = state_fingerprint(state)
            if fingerprint in excluded_hashes or fingerprint in generated_hashes:
                continue
            index = len(generated)
            generated.append(state)
            generated_hashes.add(fingerprint)
            validation_records.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "state_index": index,
                    "generation_seed": seed,
                    "state_sha256": fingerprint,
                    "source": "seeded_libero_reset",
                }
            )
    finally:
        env.close()
    return suite_name, task_id, training_records, validation_records, np.stack(generated)


def main() -> int:
    args = parse_args()
    if args.states_per_task <= 0:
        raise ValueError("--states-per-task must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    run_dir, output_dir = resolve(args.training_run), resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    training_seeds = load_training_seeds(run_dir)
    training_records: list[dict] = []
    validation_records: list[dict] = []
    excluded_seeds = set(training_seeds.values())
    payloads = [
        (
            suite_name,
            task_id,
            args.states_per_task,
            args.base_seed,
            training_seeds,
            excluded_seeds,
        )
        for suite_name in SUITES
        for task_id in range(10)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as executor:
        for suite_name, task_id, training, validation, generated in executor.map(
            generate_task, payloads
        ):
            relative = Path("validation") / suite_name / f"task_{task_id:02d}.pt"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(generated), destination)
            for record in validation:
                record["file"] = relative.as_posix()
            training_records.extend(training)
            validation_records.extend(validation)
            print(f"generated {suite_name} task {task_id}: {len(generated)} states", flush=True)

    training_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": "legacy_training_reference",
        "training_run": str(run_dir.relative_to(PROJECT_ROOT)),
        "states": training_records,
    }
    validation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": "validation",
        "base_seed": args.base_seed,
        "states_per_task": args.states_per_task,
        "generator": "seeded_libero_reset",
        "states": validation_records,
    }
    assert_disjoint_manifests(validation_manifest, training_manifest)
    validation_hashes = {record["state_sha256"] for record in validation_records}
    validation_seeds = {record["generation_seed"] for record in validation_records}
    if len(validation_hashes) != len(validation_records):
        raise ValueError("Generated validation state bank contains duplicate states.")
    if len(validation_seeds) != len(validation_records):
        raise ValueError("Generated validation state bank contains duplicate seeds.")
    (output_dir / "training_reference_manifest.json").write_text(
        json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(validation_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"state bank: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
