"""Keep the task-direction launcher from racing HQ-SAM checkpoint loading."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.sam_hq import server


class ReadinessChecks(unittest.TestCase):
    def test_requests_are_served_by_arrival_not_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            requests = queue / "requests"
            requests.mkdir()
            older = requests / "ffff-older.json"
            newer = requests / "0000-newer.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(server._next_request(queue), older)

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
