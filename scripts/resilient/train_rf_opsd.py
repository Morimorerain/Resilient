#!/usr/bin/env python3
"""Train Fast-WAM RF-OPSD world representations on LIBERO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.rf_opsd.runtime import (  # noqa: E402
    run_rf_opsd_training,
    validate_rf_opsd_config,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="rf_opsd/fastwam_libero_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Validate on CPU or launch distributed RF-OPSD training."""
    if bool(cfg.rf_opsd.get("validate_only", False)):
        print(json.dumps(validate_rf_opsd_config(cfg), indent=2))
        return
    run_rf_opsd_training(cfg)


if __name__ == "__main__":
    main()
