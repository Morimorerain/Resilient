"""Keep the task-direction launcher from racing HQ-SAM checkpoint loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.sam_hq import server


class ReadinessChecks(unittest.TestCase):
    def test_marker_is_written_after_worker_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready_file = root / "samhq.ready.json"

            def initialize_worker(_checkpoint, *, device):
                self.assertEqual(device, "cuda")
                self.assertFalse(ready_file.exists())
                return object()

            arguments = [
                "server.py",
                "--queue-dir",
                str(root / "queue"),
                "--checkpoint",
                str(root / "mock.pth"),
                "--ready-file",
                str(ready_file),
                "--once",
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(server, "SamHqWorker", side_effect=initialize_worker),
            ):
                server.main()
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["model_identity"], server.MODEL_IDENTITY)


if __name__ == "__main__":
    unittest.main()
