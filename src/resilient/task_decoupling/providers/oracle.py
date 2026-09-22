"""Audit-only provider for already rendered simulator instance masks."""

from __future__ import annotations

import numpy as np

from ..types import EntitySpec, MaskBatch


class ArrayOracleMaskProvider:
    """Return caller-supplied oracle masks; never construct this in training."""

    audit_only = True

    def __init__(self, masks: np.ndarray, camera_names: tuple[str, ...]) -> None:
        self.masks = np.asarray(masks, dtype=bool)
        self.camera_names = tuple(camera_names)

    def segment(
        self,
        videos: dict[str, np.ndarray],
        entities: tuple[EntitySpec, ...],
    ) -> MaskBatch:
        if tuple(videos) != self.camera_names:
            raise ValueError("Oracle camera order differs from the requested video.")
        if self.masks.shape[:2] != (len(videos), len(entities)):
            raise ValueError("Oracle masks do not match the requested entities.")
        return MaskBatch(
            camera_names=self.camera_names,
            entity_ids=tuple(entity.entity_id for entity in entities),
            masks=self.masks.copy(),
            metadata={"provider": "libero_instance_oracle", "audit_only": True},
        )
