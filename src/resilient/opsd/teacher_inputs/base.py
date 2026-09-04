"""Interface for privileged Teacher conditioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from resilient.faults.paired_observation import PairedObservation

from ..types import ActionConditioning, StudentTrajectory


@dataclass(frozen=True)
class TeacherInputContext:
    """All modalities available to a provider, not to the OPSD trainer."""

    paired_observation: PairedObservation
    student_conditioning: ActionConditioning
    student_trajectory: StudentTrajectory
    extras: dict[str, Any]


class TeacherInputProvider(ABC):
    """Build privileged conditioning without branching on fault types in Trainer."""

    @abstractmethod
    def build(self, context: TeacherInputContext) -> ActionConditioning:
        """Return Teacher conditioning for the existing Student trajectory."""
