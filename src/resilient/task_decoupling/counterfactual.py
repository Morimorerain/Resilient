"""Local-blur counterfactual construction."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def build_local_blur_counterfactual(
    videos: Mapping[str, np.ndarray],
    union_masks: np.ndarray,
    *,
    camera_names: tuple[str, ...],
    sigma_short_edge_ratio: float,
) -> dict[str, np.ndarray]:
    """Blur only selected task pixels while preserving every outside pixel."""
    if sigma_short_edge_ratio <= 0.0:
        raise ValueError("sigma_short_edge_ratio must be positive.")
    if union_masks.ndim != 4 or union_masks.shape[0] != len(camera_names):
        raise ValueError("Union masks must have shape [camera,time,height,width].")
    result: dict[str, np.ndarray] = {}
    for camera_index, name in enumerate(camera_names):
        video = np.asarray(videos[name])
        masks = union_masks[camera_index]
        if video.shape[:3] != masks.shape or video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Video and masks are misaligned for camera {name}.")
        sigma = sigma_short_edge_ratio * min(video.shape[1:3])
        output = video.copy()
        for frame_index in range(video.shape[0]):
            blurred = cv2.GaussianBlur(
                video[frame_index],
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
            mask = masks[frame_index]
            output[frame_index][mask] = blurred[mask]
            if not np.array_equal(output[frame_index][~mask], video[frame_index][~mask]):
                raise RuntimeError("Counterfactual changed pixels outside the task mask.")
        result[name] = np.ascontiguousarray(output)
    return result
