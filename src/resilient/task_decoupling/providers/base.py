"""Abstract task-mask provider boundary."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import EntitySpec, MaskBatch


class TaskMaskProvider(Protocol):
    """Return camera/entity masks without exposing model-specific details."""

    def segment(
        self,
        videos: dict[str, np.ndarray],
        entities: tuple[EntitySpec, ...],
    ) -> MaskBatch:
        """Segment uint8 camera videos shaped [T,H,W,3]."""
        ...
