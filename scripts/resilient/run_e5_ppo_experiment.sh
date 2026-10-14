#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_NAME="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_ppo_10ep"
RUN_DIR="runs/two_stage_opsd/libero10_task7_e5_periodic_freeze_seed42/direct_stage2_task_direction_ppo_10ep"
FASTWAM_ENV="${FASTWAM_ENV:-${PROJECT_ROOT}/.venv}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/AILOG/libero}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_ROOT}/checkpoints}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${PROJECT_ROOT}"
TRAIN_ENTRYPOINT=train_outcome_fpo.py \
CONFIG_NAME="${CONFIG_NAME}" \
FASTWAM_ENV="${FASTWAM_ENV}" \
TRAIN_PORT="${TRAIN_PORT:-29511}" \
QUEUE_DIR="${QUEUE_DIR:-${PROJECT_ROOT}/data/.cache/task_decoupling_rpc_samhq_e5_direct_stage2_ppo}" \
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/AILOG/processes/e5_direct_stage2_task_direction_ppo_10ep}" \
  bash scripts/resilient/run_task_direction_outcome_fpo.sh \
    "${TRAIN_GPUS}" "${TRAIN_GPU_ARRAY[0]}" \
    "two_stage_opsd.distributed.num_processes=${#TRAIN_GPU_ARRAY[@]}"

shopt -s nullglob
epoch_dirs=("${RUN_DIR}"/checkpoints/epochs/epoch_0010_step_*)
if [[ "${#epoch_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one completed PPO epoch-10 checkpoint in ${RUN_DIR}." >&2
  exit 1
fi
adapter="${epoch_dirs[0]}/recovery_adapter.pt"
if [[ ! -f "${adapter}" ]]; then
  echo "Missing trained PPO recovery adapter: ${adapter}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" "${FASTWAM_ENV}/bin/python" \
  experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  MULTIRUN.task_file=configs/evaluation_tasks/libero10_task7.txt \
  MULTIRUN.num_gpus=4 \
  EVALUATION.num_trials=50 \
  EVALUATION.replan_steps=10 \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.dataset_stats_path=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.save_rollout_videos=true \
  EVALUATION.fault_config_path=configs/fault/catalog/structure/periodic_joint_freeze.yaml \
  EVALUATION.initial_state_bank.enabled=true \
  EVALUATION.initial_state_bank.manifest=data/libero_state_banks/opsd_step500_unseen_v1/validation_manifest.json \
  EVALUATION.initial_state_bank.training_manifest=data/libero_state_banks/opsd_step500_unseen_v1/training_reference_manifest.json \
  EVALUATION.opsd_adapter.enabled=true \
  "EVALUATION.opsd_adapter.checkpoint=${adapter}" \
  "EVALUATION.opsd_adapter.training_config=${RUN_DIR}/resolved_config.yaml" \
  EVALUATION.output_dir=evaluate_results/two_stage_opsd/libero10_task7_e5_direct_stage2_ppo_epoch10_replan10
