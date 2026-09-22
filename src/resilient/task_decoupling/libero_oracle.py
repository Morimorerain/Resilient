"""Audit-only LIBERO instance masks and NumPy-2 segmentation compatibility."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from PIL import Image

from .types import EntitySpec


def install_robosuite_numpy2_segmentation_compatibility() -> None:
    """Decode segmentation RGB through int32 in this process only.

    Robosuite 1.4 multiplies uint8 channels by 256 and 65536. NumPy 2 rejects
    those out-of-range scalar operations. This local audit patch preserves the
    original algorithm without modifying the installed package or RGB renderer.
    """
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    if getattr(MjRenderContextOffscreen.read_pixels, "_resilient_numpy2_safe", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_image = np.empty((height, width, 3), dtype=np.uint8)
        depth_image = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(
            rgb=rgb_image,
            depth=depth_image,
            viewport=viewport,
            con=self.con,
        )
        result: Any = rgb_image
        if segmentation:
            encoded = rgb_image.astype(np.int32)
            segmentation_image = (
                encoded[:, :, 0] + encoded[:, :, 1] * 256 + encoded[:, :, 2] * 65536
            )
            segmentation_image[segmentation_image >= self.scn.ngeom + 1] = 0
            segmentation_ids = np.full(
                (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
            )
            for index in range(self.scn.ngeom):
                geometry = self.scn.geoms[index]
                if geometry.segid != -1:
                    segmentation_ids[geometry.segid + 1, 0] = geometry.objtype
                    segmentation_ids[geometry.segid + 1, 1] = geometry.objid
            result = segmentation_ids[segmentation_image]
        return (result, depth_image) if depth else result

    read_pixels._resilient_numpy2_safe = True
    MjRenderContextOffscreen.read_pixels = read_pixels


def instance_masks_from_observation(
    env: Any,
    observation: dict[str, Any],
    *,
    camera_observation_keys: tuple[str, ...],
    instance_names: tuple[str, ...],
) -> np.ndarray:
    """Return raw-resolution masks [camera,entity,H,W] from a segmentation env."""
    masks: list[np.ndarray] = []
    for key in camera_observation_keys:
        segmentation = np.asarray(observation[key])
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise ValueError(f"Unexpected segmentation image shape for {key}: {segmentation.shape}")
        per_entity = []
        for name in instance_names:
            if name not in env.instance_to_id:
                raise KeyError(f"LIBERO instance is absent from segmentation mapping: {name}")
            per_entity.append(segmentation == int(env.instance_to_id[name]))
        masks.append(np.stack(per_entity, axis=0))
    return np.stack(masks, axis=0)


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = image.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.NEAREST,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)), dtype=np.uint8
    ).astype(bool)


def stack_preprocessed_oracle_masks(
    env: Any,
    observations: tuple[dict[str, Any], ...],
    entities: tuple[EntitySpec, ...],
    *,
    width: int = 224,
    height: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Build task masks [camera,entity,time,H,W] and robot masks [camera,time,H,W]."""
    camera_keys = (
        "agentview_segmentation_instance",
        "robot0_eye_in_hand_segmentation_instance",
    )
    robot_names = tuple(
        name
        for name in env.instance_to_id
        if any(token in name.lower() for token in ("panda", "gripper", "mount"))
    )
    task_frames: list[np.ndarray] = []
    robot_frames: list[np.ndarray] = []
    for observation in observations:
        per_camera_task: list[np.ndarray] = []
        per_camera_robot: list[np.ndarray] = []
        for key in camera_keys:
            segmentation = np.asarray(observation[key])
            if segmentation.ndim == 3:
                segmentation = segmentation[..., 0]
            segmentation = np.ascontiguousarray(segmentation[::-1, ::-1])
            entity_masks = []
            for entity in entities:
                instance_mask = np.zeros_like(segmentation, dtype=bool)
                for name in entity.oracle_instances:
                    instance_mask |= segmentation == int(env.instance_to_id[name])
                entity_masks.append(_resize_mask(instance_mask, width=width, height=height))
            robot_mask = np.zeros_like(segmentation, dtype=bool)
            for name in robot_names:
                robot_mask |= segmentation == int(env.instance_to_id[name])
            per_camera_task.append(np.stack(entity_masks, axis=0))
            per_camera_robot.append(_resize_mask(robot_mask, width=width, height=height))
        task_frames.append(np.stack(per_camera_task, axis=0))
        robot_frames.append(np.stack(per_camera_robot, axis=0))
    task = np.stack(task_frames, axis=2)
    robot = np.stack(robot_frames, axis=1)
    return task, robot
