"""Outcome-guided Flow Policy Optimization for embodied-fault recovery."""

from .advantage import leave_one_out_advantages
from .objective import compute_fpo_ratio, fpo_clipped_objective
from .types import ActionConditioning, CandidateSample, OutcomeGroup

__all__ = [
    "ActionConditioning",
    "CandidateSample",
    "OutcomeGroup",
    "compute_fpo_ratio",
    "fpo_clipped_objective",
    "leave_one_out_advantages",
]
