"""On-policy realized-future privileged world-representation distillation."""

from .adapters import FastWAMWorldLoraConfig, inject_fastwam_world_lora
from .future import FutureRollout, collect_action_chunk_future, stack_observation_video
from .losses import future_flow_loss
from .teacher import FrozenFastWAMVAETeacher

__all__ = [
    "FastWAMWorldLoraConfig",
    "FrozenFastWAMVAETeacher",
    "FutureRollout",
    "collect_action_chunk_future",
    "future_flow_loss",
    "inject_fastwam_world_lora",
    "stack_observation_video",
]
