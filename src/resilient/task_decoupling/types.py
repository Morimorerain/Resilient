"""Validated data contracts for task-region decoupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EntitySpec:
    """One task-relevant entity and its fixed segmentation prompt."""

    entity_id: str
    role: str
    prompt: str
    oracle_instances: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entity_id or not self.prompt:
            raise ValueError("Entity id and prompt must be non-empty.")
        if self.role not in {"manipulated", "goal", "interaction"}:
            raise ValueError(f"Unsupported entity role: {self.role}")


@dataclass(frozen=True)
class BoxPromptBatch:
    """Audit-only per-frame box prompts aligned with cameras and entities."""

    camera_names: tuple[str, ...]
    entity_ids: tuple[str, ...]
    boxes_xyxy: np.ndarray
    valid: np.ndarray
    source: str

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes_xyxy)
        valid = np.asarray(self.valid)
        expected = (len(self.camera_names), len(self.entity_ids))
        if boxes.ndim != 4 or boxes.shape[:2] != expected or boxes.shape[-1] != 4:
            raise ValueError("Box prompts must have shape [camera, entity, time, 4].")
        if valid.shape != boxes.shape[:-1] or valid.dtype != np.bool_:
            raise ValueError("Box validity must be boolean [camera, entity, time].")
        if not np.issubdtype(boxes.dtype, np.floating):
            raise TypeError("Box prompts must use a floating-point dtype.")
        selected = boxes[valid]
        if selected.size and (
            not np.isfinite(selected).all()
            or np.any(selected[:, 2] <= selected[:, 0])
            or np.any(selected[:, 3] <= selected[:, 1])
        ):
            raise ValueError("Valid box prompts must be finite, non-empty xyxy boxes.")
        if not self.source:
            raise ValueError("Box prompt provenance must be non-empty.")


@dataclass(frozen=True)
class MaskBatch:
    """Per-camera, per-entity binary masks aligned with a short video."""

    camera_names: tuple[str, ...]
    entity_ids: tuple[str, ...]
    masks: np.ndarray
    scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.masks)
        if array.ndim != 5:
            raise ValueError("Masks must have shape [camera, entity, time, height, width].")
        expected = (len(self.camera_names), len(self.entity_ids))
        if array.shape[:2] != expected:
            raise ValueError(
                f"Mask leading dimensions {array.shape[:2]} do not match {expected}."
            )
        if array.dtype != np.bool_:
            raise TypeError("Masks must use boolean dtype.")
        if self.scores is not None:
            scores = np.asarray(self.scores)
            if scores.shape != array.shape[:3]:
                raise ValueError("Scores must have shape [camera, entity, time].")

    @property
    def union(self) -> np.ndarray:
        """Return per-camera task union masks shaped [camera, time, height, width]."""
        return self.masks.any(axis=1)


@dataclass(frozen=True)
class TaskResidual:
    """Frozen-VAE response to removing task-region high-frequency appearance."""

    full_latent: torch.Tensor
    counterfactual_latent: torch.Tensor
    delta_latent: torch.Tensor
    future_delta: torch.Tensor
    mask_batch: MaskBatch
    mask_statistics: dict[str, float]
    valid: bool
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.full_latent.shape != self.counterfactual_latent.shape:
            raise ValueError("Full and counterfactual latents must have identical shapes.")
        if self.delta_latent.shape != self.full_latent.shape:
            raise ValueError("Delta latent must preserve the encoded video shape.")
        if self.full_latent.ndim != 5 or self.full_latent.shape[2] < 2:
            raise ValueError("Expected video latents [B,C,T,H,W] with a future slice.")
        if self.future_delta.shape != self.delta_latent[:, :, 1:].shape:
            raise ValueError("future_delta must exclude exactly the first latent slice.")
