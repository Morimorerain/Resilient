#!/usr/bin/env python3
"""Train Stage-I native joint Fast-WAM LoRA on Fault rollouts."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.two_stage_opsd.stage1_trainer import run_stage1_training  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="two_stage_opsd/fastwam_libero10_task7_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Run Stage I with the pinned 4-GPU recipe."""
    print(run_stage1_training(cfg))


if __name__ == "__main__":
    main()
