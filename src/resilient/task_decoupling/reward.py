"""Task-region outcome rewards and dependency-free Gate-4 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from resilient.outcome_fpo.outcome import OutcomeTarget, temporal_delta_features


@dataclass(frozen=True)
class TaskOutcomeRewards:
    """One-sided task-region decomposition of the nominal outcome reward."""

    full: torch.Tensor
    counterfactual: torch.Tensor
    margin: torch.Tensor
    task_direction: torch.Tensor


def task_outcome_rewards(
    full_hidden: torch.Tensor,
    counterfactual_hidden: torch.Tensor,
    target: OutcomeTarget,
) -> TaskOutcomeRewards:
    """Compare full and task-removed candidate features to one nominal target.

    The temporary box-prompted HQ-SAM backend cannot segment a decoded Teacher
    prediction. Therefore this deliberately performs a one-sided decomposition:
    the realized candidate is decomposed while the nominal Teacher stays intact.
    """
    target_features = temporal_delta_features(target.hidden, target.tokens_per_frame)
    full_features = temporal_delta_features(full_hidden, target.tokens_per_frame)
    counterfactual_features = temporal_delta_features(
        counterfactual_hidden, target.tokens_per_frame
    )
    if (
        target_features.shape != full_features.shape
        or full_features.shape != counterfactual_features.shape
    ):
        raise ValueError("Target, full, and counterfactual feature layouts must match.")
    full = F.cosine_similarity(full_features, target_features, dim=-1).mean()
    counterfactual = F.cosine_similarity(
        counterfactual_features, target_features, dim=-1
    ).mean()
    task_delta = full_features - counterfactual_features
    task_direction = F.cosine_similarity(task_delta, target_features, dim=-1).mean()
    return TaskOutcomeRewards(
        full=full,
        counterfactual=counterfactual,
        margin=full - counterfactual,
        task_direction=task_direction,
    )


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def group_ranking_metrics(
    reward: np.ndarray,
    progress: np.ndarray,
    group: np.ndarray,
    *,
    progress_tolerance: float = 1.0e-4,
    minimum_reward_std: float = 1.0e-4,
) -> dict[str, float | int]:
    """Measure within-state ranking without leaking comparisons across anchors."""
    reward = np.asarray(reward, dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64)
    group = np.asarray(group)
    if reward.shape != progress.shape or reward.shape != group.shape or reward.ndim != 1:
        raise ValueError("Reward, progress, and group must be aligned one-dimensional arrays.")
    correct = 0.0
    pair_count = 0
    correlations: list[float] = []
    top1_hits: list[float] = []
    regrets: list[float] = []
    active = 0
    unique_groups = np.unique(group)
    for group_id in unique_groups:
        selected = group == group_id
        r = reward[selected]
        p = progress[selected]
        if len(r) < 2:
            continue
        if np.std(r) >= minimum_reward_std:
            active += 1
        for left in range(len(r)):
            for right in range(left + 1, len(r)):
                progress_delta = p[left] - p[right]
                if abs(progress_delta) <= progress_tolerance:
                    continue
                reward_delta = r[left] - r[right]
                correct += float(np.sign(reward_delta) == np.sign(progress_delta))
                pair_count += 1
        correlation = _pearson(_rank_average(r), _rank_average(p))
        if np.isfinite(correlation):
            correlations.append(correlation)
        selected_index = int(np.argmax(r))
        best_progress = float(np.max(p))
        top1_hits.append(float(best_progress - float(p[selected_index]) <= progress_tolerance))
        regrets.append(best_progress - float(p[selected_index]))
    group_count = len(unique_groups)
    return {
        "group_count": int(group_count),
        "pair_count": int(pair_count),
        "pairwise_accuracy": float(correct / pair_count) if pair_count else float("nan"),
        "mean_group_spearman": float(np.mean(correlations)) if correlations else float("nan"),
        "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "mean_top1_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "active_group_fraction": float(active / group_count) if group_count else 0.0,
    }


def within_group_motion_partial_correlation(
    reward: np.ndarray,
    progress: np.ndarray,
    motion: np.ndarray,
    group: np.ndarray,
) -> float:
    """Correlate reward and motion after removing group means and progress."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (reward, progress, motion)]
    group = np.asarray(group)
    if any(value.shape != group.shape for value in arrays) or group.ndim != 1:
        raise ValueError("All partial-correlation inputs must be aligned vectors.")
    centered = []
    for value in arrays:
        output = np.empty_like(value)
        for group_id in np.unique(group):
            selected = group == group_id
            output[selected] = value[selected] - value[selected].mean()
        centered.append(output)
    centered_reward, centered_progress, centered_motion = centered
    denominator = float(np.dot(centered_progress, centered_progress))
    if denominator > 1e-12:
        reward_residual = centered_reward - centered_progress * (
            np.dot(centered_progress, centered_reward) / denominator
        )
        motion_residual = centered_motion - centered_progress * (
            np.dot(centered_progress, centered_motion) / denominator
        )
    else:
        reward_residual, motion_residual = centered_reward, centered_motion
    return _pearson(reward_residual, motion_residual)
