"""Gate-1 through Gate-3 metrics for task-region residual validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute visibility-aware binary segmentation metrics."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and oracle masks must have the same shape.")
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    pred_area = int(pred.sum())
    truth_area = int(truth.sum())
    precision = (
        1.0
        if pred_area == 0 and truth_area == 0
        else intersection / max(pred_area, 1)
    )
    return {
        "iou": 1.0 if union == 0 else intersection / union,
        "precision": precision,
        "recall": 1.0 if truth_area == 0 else intersection / truth_area,
        "prediction_area_ratio": pred_area / max(pred.size, 1),
        "target_area_ratio": truth_area / max(truth.size, 1),
    }


def temporal_mask_iou(masks: np.ndarray) -> float:
    """Mean adjacent-frame IoU, ignoring pairs where both frames are empty."""
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError("Temporal masks must have shape [T,H,W] with T >= 2.")
    values: list[float] = []
    for left, right in zip(array[:-1], array[1:], strict=True):
        union = np.logical_or(left, right).sum()
        if union:
            values.append(float(np.logical_and(left, right).sum() / union))
    return float(np.mean(values)) if values else 1.0


def robot_contamination(task_mask: np.ndarray, robot_mask: np.ndarray) -> float:
    """Fraction of predicted task pixels overlapping the robot oracle mask."""
    task = np.asarray(task_mask, dtype=bool)
    robot = np.asarray(robot_mask, dtype=bool)
    if task.shape != robot.shape:
        raise ValueError("Task and robot masks must have identical shapes.")
    return float(np.logical_and(task, robot).sum() / max(task.sum(), 1))


def residual_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine similarity between two candidate task residuals."""
    if left.shape != right.shape:
        raise ValueError("Residual tensors must have identical shapes.")
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0))


def effective_rank(features: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Entropy effective rank of a sample-by-feature matrix."""
    if features.ndim < 2 or features.shape[0] < 2:
        raise ValueError("Effective rank requires at least two feature samples.")
    matrix = features.float().reshape(features.shape[0], -1)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float(entropy.exp())


@dataclass(frozen=True)
class RidgeProbeResult:
    r2: float
    predictions: np.ndarray


def ridge_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    regularization: float = 1.0,
) -> RidgeProbeResult:
    """Fit a dependency-free ridge probe and report held-out R-squared."""
    x_train = np.asarray(train_features, dtype=np.float64)
    y_train = np.asarray(train_targets, dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(test_features, dtype=np.float64)
    y_test = np.asarray(test_targets, dtype=np.float64).reshape(-1, 1)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Probe features must be aligned two-dimensional matrices.")
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.concatenate([train, np.ones((train.shape[0], 1))], axis=1)
    test = np.concatenate([test, np.ones((test.shape[0], 1))], axis=1)
    identity = np.eye(train.shape[1], dtype=np.float64)
    identity[-1, -1] = 0.0
    if train.shape[1] <= train.shape[0]:
        weights = np.linalg.solve(
            train.T @ train + regularization * identity,
            train.T @ y_train,
        )
    else:
        # Dual ridge avoids a feature-by-feature inverse for latent probes.
        dual = np.linalg.solve(
            train @ train.T + regularization * np.eye(train.shape[0]),
            y_train,
        )
        weights = train.T @ dual
    predictions = test @ weights
    residual = float(np.square(y_test - predictions).sum())
    total = float(np.square(y_test - y_test.mean()).sum())
    r2 = 1.0 - residual / max(total, 1e-12)
    return RidgeProbeResult(r2=r2, predictions=predictions[:, 0])
