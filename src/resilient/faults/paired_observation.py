"""Paired privileged/student observations captured without stepping the simulator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import FaultContext
from .pipeline import FaultPipeline


@dataclass(frozen=True)
class PairedObservation:
    """Student and privileged observations from one unchanged simulator state."""

    student: Any
    privileged: Any
    fault_metadata: dict[str, Any]


def capture_paired_observation(
    *,
    env: Any,
    pipeline: FaultPipeline,
    capture_raw_observation: Callable[[], Any],
    context: FaultContext,
) -> PairedObservation:
    """Capture faulted and suspended views without calling ``env.step``."""
    faulted_raw = capture_raw_observation()
    student = pipeline.transform_observation(faulted_raw, env, context)
    with pipeline.suspend(env, context=context):
        privileged = capture_raw_observation()
    return PairedObservation(
        student=student,
        privileged=privileged,
        fault_metadata=pipeline.metadata(),
    )
