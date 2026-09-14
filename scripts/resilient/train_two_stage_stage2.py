#!/usr/bin/env python3
"""Continue the Stage-I adapter with single-task anchored Outcome-FPO."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.outcome_fpo.runtime import run_outcome_fpo_training  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="two_stage_opsd/fastwam_libero10_task7_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Run Stage II from the explicit Stage-I adapter checkpoint."""
    if cfg.two_stage_opsd.stage2.initial_adapter in {None, "", "null"}:
        raise ValueError("Stage II requires two_stage_opsd.stage2.initial_adapter.")
    run_outcome_fpo_training(cfg)


if __name__ == "__main__":
    main()
