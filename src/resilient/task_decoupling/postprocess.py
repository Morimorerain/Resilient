"""Simple deterministic mask cleanup for coarse task-region coverage."""

from __future__ import annotations

import cv2
import numpy as np

from .types import MaskBatch


def _clean_mask(
    mask: np.ndarray,
    *,
    minimum_component_area_px: int,
    fill_hole_area_px: int,
    dilation_radius_px: int,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component_area_px:
            cleaned[labels == component] = 1

    inverted = 1 - cleaned
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    height, width = cleaned.shape
    for component in range(1, hole_count):
        x = int(hole_stats[component, cv2.CC_STAT_LEFT])
        y = int(hole_stats[component, cv2.CC_STAT_TOP])
        w = int(hole_stats[component, cv2.CC_STAT_WIDTH])
        h = int(hole_stats[component, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border and int(hole_stats[component, cv2.CC_STAT_AREA]) <= fill_hole_area_px:
            cleaned[hole_labels == component] = 1

    if dilation_radius_px > 0:
        size = 2 * dilation_radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    return cleaned.astype(bool)


def postprocess_masks(
    batch: MaskBatch,
    *,
    minimum_component_area_px: int,
    fill_hole_area_px: int,
    dilation_radius_px: int,
) -> MaskBatch:
    """Clean each entity mask independently before task-union construction."""
    if min(minimum_component_area_px, fill_hole_area_px, dilation_radius_px) < 0:
        raise ValueError("Mask postprocessing sizes must be non-negative.")
    masks = np.empty_like(batch.masks)
    for camera in range(masks.shape[0]):
        for entity in range(masks.shape[1]):
            for frame in range(masks.shape[2]):
                masks[camera, entity, frame] = _clean_mask(
                    batch.masks[camera, entity, frame],
                    minimum_component_area_px=minimum_component_area_px,
                    fill_hole_area_px=fill_hole_area_px,
                    dilation_radius_px=dilation_radius_px,
                )
    return MaskBatch(
        camera_names=batch.camera_names,
        entity_ids=batch.entity_ids,
        masks=masks,
        scores=batch.scores,
        metadata={**batch.metadata, "postprocessed": True},
    )
