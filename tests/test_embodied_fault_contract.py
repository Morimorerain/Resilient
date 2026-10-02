"""Safety checks for pinned simulator faults in adaptation experiments."""

from __future__ import annotations

import pytest

from resilient.faults import build_fault_pipeline
from resilient.faults.validation import validate_embodied_fault_contract


def test_legacy_contract_preserves_joint_one_half_motion() -> None:
    fault = {
        "pipeline": {
            "id": "legacy_e1",
            "enabled": True,
            "faults": [
                {
                    "id": "joint1",
                    "family": "structure.joint_motion",
                    "enabled": True,
                    "targets": ["robot0_joint1"],
                    "operation": {"type": "proportional"},
                    "severity": {"name": "motion_retention", "value": 0.5, "unit": "ratio"},
                }
            ],
        }
    }
    validate_embodied_fault_contract(build_fault_pipeline(fault).metadata())


def test_e5_contract_checks_period_and_hold() -> None:
    fault = {
        "pipeline": {
            "id": "e5",
            "enabled": True,
            "faults": [
                {
                    "id": "periodic_joint1",
                    "family": "structure.periodic_joint_freeze",
                    "enabled": True,
                    "targets": ["robot0_joint1"],
                    "operation": {"type": "angular_periodic_stick"},
                    "severity": {"name": "freeze_duration", "value": 35, "unit": "control_steps"},
                    "parameters": {
                        "period_deg": 10.0,
                        "phase_deg": 0.0,
                        "release_fraction": 0.25,
                    },
                }
            ],
        }
    }
    metadata = build_fault_pipeline(fault).metadata()
    contract = {
        "family": "structure.periodic_joint_freeze",
        "targets": ["robot0_joint1"],
        "severity_name": "freeze_duration",
        "severity_value": 35,
        "parameters": {
            "period_deg": 10.0,
            "phase_deg": 0.0,
            "hold_steps": 35,
            "release_fraction": 0.25,
        },
    }
    validate_embodied_fault_contract(metadata, contract)
    with pytest.raises(ValueError, match="period_deg"):
        validate_embodied_fault_contract(
            metadata, {**contract, "parameters": {**contract["parameters"], "period_deg": 5.0}}
        )
    with pytest.raises(ValueError, match="severity"):
        validate_embodied_fault_contract(metadata, {**contract, "severity_value": 34})
