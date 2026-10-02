"""Explicit simulator-fault contracts for reproducible training runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_LEGACY_JOINT1_HALF = {
    "family": "structure.joint_motion",
    "targets": ["robot0_joint1"],
    "severity_name": "motion_retention",
    "severity_value": 0.5,
}


def validate_embodied_fault_contract(
    metadata: Mapping[str, Any], expectation: Mapping[str, Any] | None = None
) -> None:
    """Require one enabled structural Fault matching the run's declared contract.

    Older configurations have no explicit contract and retain their original
    joint-one, half-motion restriction for strict checkpoint compatibility.
    """
    expected = _LEGACY_JOINT1_HALF if expectation is None else expectation
    faults = metadata.get("faults", [])
    if not metadata.get("enabled") or len(faults) != 1:
        raise ValueError("Training requires exactly one enabled simulator Fault.")
    fault = faults[0]
    if fault.get("family") != expected["family"]:
        raise ValueError("Simulator Fault family does not match the run contract.")
    if fault.get("targets") != list(expected["targets"]):
        raise ValueError("Simulator Fault targets do not match the run contract.")
    severity = fault.get("severity", {})
    if severity.get("name") != expected["severity_name"] or not math.isclose(
        float(severity.get("value", float("nan"))),
        float(expected["severity_value"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Simulator Fault severity does not match the run contract.")
    for key, value in expected.get("parameters", {}).items():
        actual = fault.get("parameters", {}).get(key)
        if isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]
        if isinstance(value, float | int) and isinstance(actual, float | int):
            matches = math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-9)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"Simulator Fault parameter {key} does not match the run contract.")
