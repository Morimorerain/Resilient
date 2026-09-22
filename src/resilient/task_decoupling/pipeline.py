"""End-to-end task-region counterfactual feature pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .counterfactual import build_local_blur_counterfactual
from .postprocess import postprocess_masks
from .preprocessing import camera_videos_to_model_video
from .types import BoxPromptBatch, EntitySpec, TaskResidual
from .vae_residual import encode_task_residual


class TaskRegionDisentangler:
    """Extract a frozen-VAE residual induced by task-only local blur."""

    def __init__(
        self,
        *,
        mask_provider,
        entities: tuple[EntitySpec, ...],
        camera_order: Sequence[str],
        concat_mode: str,
        minimum_component_area_px: int = 16,
        fill_hole_area_px: int = 64,
        dilation_radius_px: int = 5,
        sigma_short_edge_ratio: float = 0.08,
    ) -> None:
        self.mask_provider = mask_provider
        self.entities = entities
        self.camera_order = tuple(camera_order)
        self.concat_mode = str(concat_mode)
        self.minimum_component_area_px = int(minimum_component_area_px)
        self.fill_hole_area_px = int(fill_hole_area_px)
        self.dilation_radius_px = int(dilation_radius_px)
        self.sigma_short_edge_ratio = float(sigma_short_edge_ratio)

    def extract(
        self,
        videos: Mapping[str, np.ndarray],
        model: Any,
        *,
        prompt_hints: BoxPromptBatch | None = None,
    ) -> TaskResidual:
        """Segment, build a counterfactual, and encode one candidate video."""
        selected = {name: np.asarray(videos[name]) for name in self.camera_order}
        masks = self.mask_provider.segment(selected, self.entities, prompt_hints)
        if masks.camera_names != self.camera_order:
            raise ValueError("Mask provider changed the configured camera order.")
        cleaned = postprocess_masks(
            masks,
            minimum_component_area_px=self.minimum_component_area_px,
            fill_hole_area_px=self.fill_hole_area_px,
            dilation_radius_px=self.dilation_radius_px,
        )
        union = cleaned.union
        counterfactual = build_local_blur_counterfactual(
            selected,
            union,
            camera_names=self.camera_order,
            sigma_short_edge_ratio=self.sigma_short_edge_ratio,
        )
        full_video = camera_videos_to_model_video(
            selected,
            camera_order=self.camera_order,
            mode=self.concat_mode,
        )
        counterfactual_video = camera_videos_to_model_video(
            counterfactual,
            camera_order=self.camera_order,
            mode=self.concat_mode,
        )
        full, minus_task, delta, future_delta = encode_task_residual(
            model, full_video, counterfactual_video
        )
        area_ratio = float(union.mean())
        visible_entity_fraction = float(cleaned.masks.any(axis=(-2, -1)).mean())
        valid = bool(union.any())
        return TaskResidual(
            full_latent=full,
            counterfactual_latent=minus_task,
            delta_latent=delta,
            future_delta=future_delta,
            mask_batch=cleaned,
            mask_statistics={
                "union_area_ratio": area_ratio,
                "visible_entity_frame_fraction": visible_entity_fraction,
                "future_delta_rms": float(future_delta.square().mean().sqrt()),
            },
            valid=valid,
            provenance={
                "method": "task_region_counterfactual_vae_residual",
                "sigma_short_edge_ratio": self.sigma_short_edge_ratio,
                "dilation_radius_px": self.dilation_radius_px,
            },
        )
