"""Aggregate RoboTwin manager outputs and compare them with FastWAM Table 3."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .protocol import PAPER_PROTOCOL, load_paper_references


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"Invalid binomial counts: {successes}/{trials}")
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return centre - margin, centre + margin


def _load_manager_summary(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {}
    for row in payload.get("per_task", []):
        task_name = str(row["task_name"])
        clean = row.get("clean_success_rate")
        randomized = row.get("random_success_rate")
        if clean is None or randomized is None:
            raise ValueError(f"Incomplete result for task {task_name} in {path}")
        result[task_name] = {"clean": float(clean), "randomized": float(randomized)}
    return result


def build_paper_comparison(
    summary_path: Path,
    reference_path: Path,
    *,
    episodes_per_condition: int = PAPER_PROTOCOL.episodes_per_condition_per_task,
) -> dict[str, Any]:
    """Create complete per-task and aggregate paper comparison data."""
    if episodes_per_condition <= 0:
        raise ValueError("episodes_per_condition must be positive")
    actual = _load_manager_summary(summary_path)
    references = load_paper_references(reference_path)
    reference_by_task = {item.task_name: item for item in references}
    if set(actual) != set(reference_by_task):
        raise ValueError(
            "Manager summary does not contain the complete paper task set: "
            f"missing={sorted(set(reference_by_task) - set(actual))}, "
            f"extra={sorted(set(actual) - set(reference_by_task))}"
        )

    per_task: list[dict[str, Any]] = []
    clean_successes = 0
    randomized_successes = 0
    for reference in references:
        rates = actual[reference.task_name]
        for condition in ("clean", "randomized"):
            rate = rates[condition]
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"Out-of-range {condition} rate for {reference.task_name}: {rate}")
        clean_count = round(rates["clean"] * episodes_per_condition)
        randomized_count = round(rates["randomized"] * episodes_per_condition)
        if abs(clean_count / episodes_per_condition - rates["clean"]) > 1e-9:
            raise ValueError(f"Clean rate is not an episode count for {reference.task_name}")
        if abs(randomized_count / episodes_per_condition - rates["randomized"]) > 1e-9:
            raise ValueError(f"Randomized rate is not an episode count for {reference.task_name}")
        clean_successes += clean_count
        randomized_successes += randomized_count
        per_task.append(
            {
                **asdict(reference),
                "actual_clean_percent": rates["clean"] * 100.0,
                "actual_randomized_percent": rates["randomized"] * 100.0,
                "clean_delta_pp": rates["clean"] * 100.0 - reference.clean_percent,
                "randomized_delta_pp": (
                    rates["randomized"] * 100.0 - reference.randomized_percent
                ),
            }
        )

    condition_trials = len(references) * episodes_per_condition
    total_successes = clean_successes + randomized_successes
    total_trials = condition_trials * 2

    def aggregate_row(
        name: str,
        successes: int,
        trials: int,
        paper_percent: float,
    ) -> dict[str, Any]:
        low, high = wilson_interval(successes, trials)
        actual_percent = successes / trials * 100.0
        return {
            "condition": name,
            "successes": successes,
            "trials": trials,
            "actual_percent": actual_percent,
            "paper_percent": paper_percent,
            "delta_pp": actual_percent - paper_percent,
            "wilson_95_low_percent": low * 100.0,
            "wilson_95_high_percent": high * 100.0,
        }

    aggregate = [
        aggregate_row(
            "clean",
            clean_successes,
            condition_trials,
            PAPER_PROTOCOL.clean_success_percent,
        ),
        aggregate_row(
            "randomized",
            randomized_successes,
            condition_trials,
            PAPER_PROTOCOL.randomized_success_percent,
        ),
        aggregate_row(
            "overall",
            total_successes,
            total_trials,
            PAPER_PROTOCOL.displayed_average_percent,
        ),
    ]
    return {
        "schema_version": 1,
        "episodes_per_condition_per_task": episodes_per_condition,
        "per_task": per_task,
        "aggregate": aggregate,
    }


def write_paper_comparison(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON, CSV, and Markdown forms of one comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aggregate = report["aggregate"]
    with (output_dir / "paper_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    per_task = report["per_task"]
    with (output_dir / "paper_comparison_per_task.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader()
        writer.writerows(per_task)

    lines = [
        "# FastWAM RoboTwin paper comparison",
        "",
        "| Condition | Reproduced | Paper | Delta | Wilson 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['condition']} | {row['successes']}/{row['trials']} "
            f"({row['actual_percent']:.2f}%) | {row['paper_percent']:.2f}% | "
            f"{row['delta_pp']:+.2f} pp | "
            f"[{row['wilson_95_low_percent']:.2f}%, "
            f"{row['wilson_95_high_percent']:.2f}%] |"
        )
    lines.extend(
        [
            "",
            "Paper source: Fast-WAM arXiv:2603.16666, Tables 1 and 3.",
            "",
        ]
    )
    (output_dir / "paper_comparison.md").write_text("\n".join(lines), encoding="utf-8")
