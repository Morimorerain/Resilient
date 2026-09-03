#!/usr/bin/env bash
set -euo pipefail

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=./evaluate_results/reproduction/full \
  MULTIRUN.num_gpus="${NUM_GPUS:-8}"
