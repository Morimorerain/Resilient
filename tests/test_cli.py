"""Tests for the minimal command-line interface."""

from __future__ import annotations

import subprocess
import sys
import unittest

from resilient import __version__


class CommandLineTests(unittest.TestCase):
    def test_version(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "resilient", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), f"resilient {__version__}")


if __name__ == "__main__":
    unittest.main()
