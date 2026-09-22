"""Optional task-region feature extraction at the Stage-II rollout boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from resilient.task_decoupling import (
    TaskRegionDisentangler,
    load_entity_specification,
    validate_task_decoupling_config,
)
from resilient.task_decoupling.preprocessing import stack_camera_videos
from resilient.task_decoupling.providers import FileServiceMaskProvider


def build_task_disentangler(
    cfg: DictConfig, *, project_root: Path
) -> TaskRegionDisentangler | None:
    """Build the configured audit feature extractor without changing Stage-II defaults."""
    section = cfg.get("task_decoupling")
    if section is None or not bool(section.get("enabled", False)):
        return None
    validate_task_decoupling_config(section)

    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else project_root / value

    entities = load_entity_specification(
        resolve(str(section.entities.config)),
        expected_suite=str(cfg.two_stage_opsd.task.suite),
        expected_task_id=int(cfg.two_stage_opsd.task.task_id),
    )
    provider = FileServiceMaskProvider(
        resolve(str(section.mask_provider.queue_dir)),
        model_identity=str(section.mask_provider.model_identity),
        output_probability_threshold=float(
            section.segmentation.output_probability_threshold
        ),
        timeout_seconds=float(section.mask_provider.timeout_seconds),
        poll_seconds=float(section.mask_provider.poll_seconds),
        cache_enabled=bool(section.mask_provider.cache_enabled),
    )
    return TaskRegionDisentangler(
        mask_provider=provider,
        entities=entities,
        camera_order=("image", "wrist_image"),
        concat_mode=str(cfg.data.train.get("concat_multi_camera", "horizontal")),
        minimum_component_area_px=int(section.segmentation.minimum_component_area_px),
        fill_hole_area_px=int(section.segmentation.fill_hole_area_px),
        dilation_radius_px=int(section.segmentation.dilation_radius_px),
        sigma_short_edge_ratio=float(section.counterfactual.sigma_short_edge_ratio),
    )


def extract_task_features(
    disentangler: TaskRegionDisentangler,
    *,
    observations: tuple[Any, ...],
    processor,
    model,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Extract one candidate residual and return its aligned camera videos."""
    videos = stack_camera_videos(observations, processor)
    residual = disentangler.extract(videos, model)
    return residual, videos
