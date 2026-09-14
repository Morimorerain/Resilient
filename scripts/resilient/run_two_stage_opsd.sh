#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${1:-runs/two_stage_opsd/libero10_task7_joint1_retention_p0p5/manual_run}"
shift || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly four physical GPU ids." >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "Two-stage single-task training currently requires exactly four GPUs." >&2
  exit 2
fi

cd "${project_root}"

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  scripts/resilient/collect_two_stage_opsd.py \
  "two_stage_opsd.distributed.num_processes=4" \
  "$@"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 \
  scripts/resilient/train_two_stage_stage1.py \
  "output_dir=${run_dir}" \
  "two_stage_opsd.distributed.num_processes=4" \
  "$@"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes 4 \
  scripts/resilient/train_two_stage_stage2.py \
  "output_dir=${run_dir}/stage2" \
  "two_stage_opsd.distributed.num_processes=4" \
  "two_stage_opsd.stage2.initial_adapter=${run_dir}/stage1/final/joint_adapter.pt" \
  "learning_rate=1.0e-6" \
  "weight_decay=0.0" \
  "max_grad_norm=0.1" \
  "gradient_accumulation_steps=1" \
  "num_epochs=1" \
  "$@"
