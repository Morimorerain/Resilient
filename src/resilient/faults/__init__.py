"""Composable robot fault injection primitives."""

from .base import FaultContext, FaultRuntime
from .environment import FaultedEnvironment, install_fault_pipeline
from .pipeline import FaultPipeline
from .registry import build_fault_pipeline, register_fault
from .types import FaultTransition, SeveritySpec

__all__ = [
    "FaultContext",
    "FaultedEnvironment",
    "FaultPipeline",
    "FaultRuntime",
    "FaultTransition",
    "SeveritySpec",
    "build_fault_pipeline",
    "install_fault_pipeline",
    "register_fault",
]
