"""Composable robot fault injection primitives."""

from .base import FaultContext, FaultRuntime
from .pipeline import FaultPipeline
from .registry import build_fault_pipeline, register_fault
from .types import FaultTransition, SeveritySpec

__all__ = [
    "FaultContext",
    "FaultPipeline",
    "FaultRuntime",
    "FaultTransition",
    "SeveritySpec",
    "build_fault_pipeline",
    "register_fault",
]
