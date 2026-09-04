#!/usr/bin/env python3
"""Train Fast-WAM with severity-aware OPSD-Flow on LIBERO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from resilient.opsd.runtime import run_opsd_training, validate_opsd_config  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="opsd/fastwam_libero")
def main(cfg: DictConfig) -> None:
    """Validate without CUDA when requested, otherwise start distributed training."""
    validate_only = bool(cfg.opsd.get("validate_only", False))
    if validate_only:
        print(json.dumps(validate_opsd_config(cfg), indent=2))
        return
    run_opsd_training(cfg)


if __name__ == "__main__":
    main()
