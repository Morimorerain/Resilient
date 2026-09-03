"""Capture reproducibility metadata without machine-specific project paths."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str | None:
    """Return command output, or None when the command is unavailable."""
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def main() -> int:
    """Write a compact environment snapshot to a user-selected path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
