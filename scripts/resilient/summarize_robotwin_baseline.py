#!/usr/bin/env python3
"""Compare a complete RoboTwin manager summary with FastWAM paper results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resilient.robotwin.reporting import (  # noqa: E402
    build_paper_comparison,
    write_paper_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to run_robotwin_manager summary.json")
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    summary = args.summary if args.summary.is_absolute() else PROJECT_ROOT / args.summary
    output_dir = args.output_dir or summary.parent
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    reference = PROJECT_ROOT / "reproduce" / "fastwam_robotwin" / "paper_reference.csv"
    report = build_paper_comparison(
        summary,
        reference,
        episodes_per_condition=args.episodes_per_condition,
    )
    write_paper_comparison(report, output_dir)
    print(f"Wrote paper comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
