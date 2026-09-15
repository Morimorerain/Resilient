#!/usr/bin/env python3
"""Run the pinned RoboTwin baseline preflight and optionally save JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resilient.robotwin.preflight import run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_preflight(
        PROJECT_ROOT,
        verify_hashes=args.verify_hashes,
        require_runtime=not args.skip_runtime,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
