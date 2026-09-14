#!/usr/bin/env python3
"""Collect Stage-I Fast-WAM rollouts under an embodiment Fault."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.two_stage_opsd.stage1_collector import collect_stage1_dataset  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="two_stage_opsd/fastwam_libero10_task7_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Run resumable distributed data collection."""
    print(collect_stage1_dataset(cfg))


if __name__ == "__main__":
    main()
