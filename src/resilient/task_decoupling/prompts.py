"""Prompt construction helpers kept separate from segmentation backends."""

from __future__ import annotations

import numpy as np

from .types import BoxPromptBatch


def boxes_from_binary_masks(
    masks: np.ndarray,
    *,
    camera_names: tuple[str, ...],
    entity_ids: tuple[str, ...],
    source: str,
    padding_px: int = 0,
) -> BoxPromptBatch:
    """Convert aligned oracle masks to clipped per-frame xyxy prompts."""
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 5:
        raise ValueError("Source masks must have shape [camera, entity, time, H, W].")
    if array.shape[:2] != (len(camera_names), len(entity_ids)):
        raise ValueError("Source mask camera/entity dimensions do not match metadata.")
    if padding_px < 0:
        raise ValueError("Box padding must be non-negative.")
    boxes = np.zeros((*array.shape[:3], 4), dtype=np.float32)
    valid = np.zeros(array.shape[:3], dtype=bool)
    height, width = array.shape[-2:]
    for camera_index, entity_index, frame_index in np.ndindex(array.shape[:3]):
        ys, xs = np.nonzero(array[camera_index, entity_index, frame_index])
        if xs.size == 0:
            continue
        x0 = max(0, int(xs.min()) - padding_px)
        y0 = max(0, int(ys.min()) - padding_px)
        x1 = min(width, int(xs.max()) + 1 + padding_px)
        y1 = min(height, int(ys.max()) + 1 + padding_px)
        boxes[camera_index, entity_index, frame_index] = (x0, y0, x1, y1)
        valid[camera_index, entity_index, frame_index] = True
    return BoxPromptBatch(
        camera_names=camera_names,
        entity_ids=entity_ids,
        boxes_xyxy=boxes,
        valid=valid,
        source=source,
    )
