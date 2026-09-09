#!/usr/bin/env python3
"""Train a Fast-WAM Recovery LoRA with outcome-guided FPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.outcome_fpo.runtime import (  # noqa: E402
    run_outcome_fpo_training,
    validate_outcome_fpo_config,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="outcome_fpo/fastwam_libero_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Validate the resolved experiment or launch distributed training."""
    if bool(cfg.outcome_fpo.get("validate_only", False)):
        print(json.dumps(validate_outcome_fpo_config(cfg), indent=2))
        return
    run_outcome_fpo_training(cfg)


if __name__ == "__main__":
    main()
