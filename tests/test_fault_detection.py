"""CPU tests for latent fault-detection metrics and severity aggregation."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fastwam.models.wan22.fastwam import FastWAM
from resilient.fault_detection.metrics import EpisodeLatentAccumulator
from resilient.fault_detection.study import (
    analyze_severity_study,
    roc_auc,
    spearman_correlation,
)


class LatentMetricTests(unittest.TestCase):
    def test_pair_metrics_and_temporal_consistency(self) -> None:
        accumulator = EpisodeLatentAccumulator()
        predicted = torch.tensor([0.0, 1.0])
        real = torch.tensor([1.0, 3.0])
        accumulator.update(predicted, real)
        accumulator.update(predicted + 1.0, real + 1.0)
        metrics, prototype = accumulator.finalize()

        self.assertEqual(metrics["num_aligned_clips"], 2)
        self.assertAlmostEqual(float(metrics["lpe"]), 2.5)
        self.assertAlmostEqual(float(metrics["rm"]), 1.5)
        self.assertAlmostEqual(float(metrics["rtc"]), 1.0, places=6)
        np.testing.assert_allclose(prototype, np.asarray([1.0, 2.0], dtype=np.float32))

    def test_rank_statistics_handle_ties(self) -> None:
        self.assertAlmostEqual(spearman_correlation([0, 1, 2], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.0, 0.5, 0.5, 1.0]), 0.875)

    def test_video_latent_switches_preserve_upstream_defaults(self) -> None:
        parameters = inspect.signature(FastWAM.infer_joint).parameters
        self.assertIs(parameters["return_video_latents"].default, False)
        self.assertIs(parameters["decode_video"].default, True)


class SeverityStudyTests(unittest.TestCase):
    def test_analysis_exports_every_requested_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for severity in (0.0, 10.0, 20.0):
                run_dir = root / f"severity_{severity:g}"
                run_dir.mkdir()
                episodes = []
                prototypes = []
                for episode_index in range(4):
                    rm = 0.1 + severity / 100.0 + episode_index * 0.01
                    episodes.append(
                        {
                            "episode_index": episode_index,
                            "num_aligned_clips": 2,
                            "lpe": rm**2,
                            "lcd": rm / 2.0,
                            "rm": rm,
                            "rtc": 0.8 + episode_index * 0.01,
                        }
                    )
                    prototypes.append([1.0, severity / 10.0 + episode_index * 0.1])
                result_path = run_dir / "result.json"
                result_path.write_text(
                    json.dumps({"fault_detection": {"episodes": episodes}}),
                    encoding="utf-8",
                )
                residual_path = run_dir / "residuals.npz"
                np.savez_compressed(
                    residual_path,
                    episode_indices=np.arange(4),
                    prototypes=np.asarray(prototypes, dtype=np.float32),
                )
                runs.append(
                    {
                        "severity": severity,
                        "result_path": result_path,
                        "residual_path": residual_path,
                    }
                )

            output = root / "analysis"
            metrics = [
                "lpe",
                "lcd",
                "rm",
                "rcs",
                "residual_matrix",
                "rtc",
                "spearman_rho",
                "fault_score",
                "roc_auc",
            ]
            summary = analyze_severity_study(
                runs,
                output_dir=output,
                metrics=metrics,
                clean_severity=0.0,
                bootstrap_samples=20,
            )
            self.assertEqual(summary["episode_metrics"]["rm"][0]["count"], 4)
            self.assertTrue((output / "metrics_summary.json").is_file())
            self.assertTrue((output / "metrics_summary.csv").is_file())
            self.assertTrue((output / "metrics_summary.md").is_file())
            self.assertTrue((output / "residual_similarity_episodes_severity-p0.png").is_file())
            expected_plots = {
                "lpe_severity_errorbar.png",
                "lcd_severity_errorbar.png",
                "rm_severity_errorbar.png",
                "rcs_severity_errorbar.png",
                "residual_similarity_matrix.png",
                "rtc_severity_errorbar.png",
                "spearman_rho_errorbar.png",
                "fault_score_severity_errorbar.png",
                "roc_auc_severity_errorbar.png",
            }
            self.assertTrue(all((output / name).is_file() for name in expected_plots))


if __name__ == "__main__":
    unittest.main()
