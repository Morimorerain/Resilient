"""Tests for the pinned RoboTwin paper protocol and reporting utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from resilient.robotwin.assets import (
    materialize_embodiment_configs,
    validate_materialized_planner_config,
)
from resilient.robotwin.protocol import (
    PAPER_PROTOCOL,
    load_paper_references,
    load_task_limits,
    validate_task_universe,
)
from resilient.robotwin.reporting import (
    build_paper_comparison,
    wilson_interval,
    write_paper_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT_ROOT / "reproduce" / "fastwam_robotwin" / "paper_reference.csv"
TASK_LIMITS = (
    PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
)


class RobotwinProtocolTests(unittest.TestCase):
    def test_materializes_machine_local_curobo_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            robotwin_root = Path(temp_dir) / "RoboTwin"
            embodiment = robotwin_root / "assets" / "embodiments" / "aloha-agilex"
            embodiment.mkdir(parents=True)
            template = embodiment / "curobo_left_tmp.yml"
            template.write_text(
                "urdf_path: ${ASSETS_PATH}/assets/embodiments/robot.urdf\n",
                encoding="utf-8",
            )

            rendered = materialize_embodiment_configs(robotwin_root)

            target = embodiment / "curobo_left.yml"
            self.assertEqual(rendered, [target])
            self.assertIn(str(robotwin_root.resolve()), target.read_text(encoding="utf-8"))
            self.assertIn("${ASSETS_PATH}", template.read_text(encoding="utf-8"))
            self.assertIsNone(validate_materialized_planner_config(target, robotwin_root))

    def test_rejects_planner_config_from_a_different_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "curobo.yml"
            config.write_text("urdf_path: /old/checkout/assets/robot.urdf\n", encoding="utf-8")
            self.assertEqual(
                validate_materialized_planner_config(config, root / "RoboTwin"),
                "was generated for a different RoboTwin checkout",
            )

    def test_external_asset_manifest_is_pinned(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "manifests" / "assets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["model_repository"]["revision"],
            "8eaceeb24c3cc92ff2a9c9a9d266a4941b836705",
        )
        simulator = manifest["robotwin_simulator_assets"]
        self.assertEqual(
            simulator["revision"],
            "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042",
        )
        self.assertIsNone(simulator["license"])
        self.assertEqual(
            {entry["expected_directory"] for entry in simulator["archives"]},
            {"background_texture", "embodiments", "objects"},
        )
        for entry in simulator["archives"]:
            self.assertGreater(entry["size"], 0)
            self.assertEqual(len(entry["sha256"]), 64)

    def test_task_universe_and_published_means_match(self) -> None:
        task_limits = load_task_limits(TASK_LIMITS)
        references = load_paper_references(REFERENCE)
        validate_task_universe(task_limits, references)
        self.assertEqual(len(task_limits), PAPER_PROTOCOL.task_count)
        self.assertEqual(PAPER_PROTOCOL.total_episodes, 10_000)

    def test_paper_comparison_preserves_episode_counts(self) -> None:
        references = load_paper_references(REFERENCE)
        summary = {
            "per_task": [
                {
                    "task_name": item.task_name,
                    "clean_success_rate": item.clean_percent / 100.0,
                    "random_success_rate": item.randomized_percent / 100.0,
                }
                for item in references
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report = build_paper_comparison(summary_path, REFERENCE)
            write_paper_comparison(report, root)

            aggregate = {row["condition"]: row for row in report["aggregate"]}
            self.assertEqual(aggregate["clean"]["successes"], 4594)
            self.assertEqual(aggregate["randomized"]["successes"], 4589)
            self.assertEqual(aggregate["overall"]["successes"], 9183)
            self.assertTrue((root / "paper_comparison.md").is_file())
            self.assertTrue((root / "paper_comparison_per_task.csv").is_file())

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(90, 100)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)


if __name__ == "__main__":
    unittest.main()
