"""Command-line entry point for the project scaffold."""

from __future__ import annotations

import argparse

from resilient import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="resilient",
        description="Run the Resilient project scaffold.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main() -> int:
    """Run the command-line interface."""
    build_parser().parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
