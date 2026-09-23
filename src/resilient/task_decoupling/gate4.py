"""Aggregation and predeclared decision rule for the Gate-4 reward audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from resilient.task_decoupling.reward import (
    group_ranking_metrics,
    within_group_motion_partial_correlation,
)

REWARD_KEYS = (
    "reward_full",
    "reward_counterfactual",
    "reward_margin",
    "reward_task_direction",
)


def summarize_gate4_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_indices: Sequence[int],
    minimum_reward_std: float,
    progress_tolerance: float,
) -> dict[str, Any]:
    """Summarize only complete, mask-valid same-state candidate groups."""
    requested_states = {int(value) for value in state_indices}
    selected = [row for row in rows if int(row["state_index"]) in requested_states]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    expected_size = max((int(row["group_size"]) for row in selected), default=0)
    valid_groups = {
        group_id: group_rows
        for group_id, group_rows in grouped.items()
        if len(group_rows) == expected_size
        and all(bool(row["mask_valid"]) for row in group_rows)
        and not any(bool(row["terminal_before_anchor"]) for row in group_rows)
    }
    valid_rows = [row for group_rows in valid_groups.values() for row in group_rows]
    summary: dict[str, Any] = {
        "requested_group_count": len(grouped),
        "valid_group_count": len(valid_groups),
        "valid_group_fraction": len(valid_groups) / len(grouped) if grouped else 0.0,
        "candidate_count": len(valid_rows),
        "rewards": {},
    }
    if not valid_rows:
        return summary
    group = np.asarray([row["group_id"] for row in valid_rows])
    progress = np.asarray([row["task_progress"] for row in valid_rows], dtype=np.float64)
    motion = np.asarray([row["joint1_motion"] for row in valid_rows], dtype=np.float64)
    for key in REWARD_KEYS:
        reward = np.asarray([row[key] for row in valid_rows], dtype=np.float64)
        metrics = group_ranking_metrics(
            reward,
            progress,
            group,
            progress_tolerance=progress_tolerance,
            minimum_reward_std=minimum_reward_std,
        )
        metrics["motion_partial_correlation"] = within_group_motion_partial_correlation(
            reward, progress, motion, group
        )
        metrics["mean"] = float(np.mean(reward))
        metrics["standard_deviation"] = float(np.std(reward))
        summary["rewards"][key] = metrics
    return summary


def evaluate_gate4(summary: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Apply the fixed held-out rule to the predeclared task-direction reward."""
    rewards = summary.get("rewards", {})
    task = rewards.get("reward_task_direction", {})
    full = rewards.get("reward_full", {})
    checks = {
        "valid_group_fraction": float(summary.get("valid_group_fraction", 0.0))
        >= float(thresholds["minimum_valid_group_fraction"]),
        "pairwise_accuracy": float(task.get("pairwise_accuracy", float("nan")))
        >= float(thresholds["minimum_pairwise_accuracy"]),
        "pairwise_improvement": (
            float(task.get("pairwise_accuracy", float("nan")))
            - float(full.get("pairwise_accuracy", float("nan")))
        )
        >= float(thresholds["minimum_pairwise_accuracy_improvement_over_full"]),
        "active_group_fraction": float(task.get("active_group_fraction", 0.0))
        >= float(thresholds["minimum_active_group_fraction"]),
        "motion_leakage": abs(
            float(task.get("motion_partial_correlation", float("nan")))
        )
        <= min(
            float(thresholds["maximum_abs_motion_partial_correlation"]),
            abs(float(full.get("motion_partial_correlation", float("nan")))),
        ),
    }
    return {
        "primary_reward": "reward_task_direction",
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }
