"""Task-region counterfactual features for embodiment-Fault recovery."""

from .config import validate_task_decoupling_config
from .counterfactual import build_local_blur_counterfactual
from .entities import load_entity_specification
from .pipeline import TaskRegionDisentangler
from .types import EntitySpec, MaskBatch, TaskResidual

__all__ = [
    "EntitySpec",
    "MaskBatch",
    "TaskRegionDisentangler",
    "TaskResidual",
    "build_local_blur_counterfactual",
    "load_entity_specification",
    "validate_task_decoupling_config",
]
