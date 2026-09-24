#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_GPUS="${1:-0,1,2,3}"
SAM_GPU="${2:-0}"
SAM_ENV="${SAM_ENV:-${PROJECT_ROOT}/.venv-samhq}"
FASTWAM_ENV="${FASTWAM_ENV:-${PROJECT_ROOT}/AILOG/envs/resilient-fastwam-uv}"
QUEUE_DIR="${QUEUE_DIR:-${PROJECT_ROOT}/data/.cache/task_decoupling_rpc_samhq}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sam_hq/sam_hq_vit_h.pth}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/AILOG/processes/task_direction_outcome_fpo_epoch1}"
CONFIG_NAME="${CONFIG_NAME:-two_stage_opsd/fastwam_libero10_task7_joint1_half_task_direction}"

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "TRAIN_GPUS must contain exactly four comma-separated GPU indices." >&2
  exit 2
fi
if [[ ! ",${TRAIN_GPUS}," =~ ,${SAM_GPU}, ]]; then
  echo "For this four-GPU run, SAM_GPU must be one of TRAIN_GPUS." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${QUEUE_DIR}"
SAM_PID=""
TRAIN_PID=""

stop_group() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || true
  fi
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "${TRAIN_PID}"
  stop_group "${SAM_PID}"
  [[ -z "${TRAIN_PID}" ]] || wait "${TRAIN_PID}" 2>/dev/null || true
  [[ -z "${SAM_PID}" ]] || wait "${SAM_PID}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "${PROJECT_ROOT}"
setsid env CUDA_VISIBLE_DEVICES="${SAM_GPU}" PYTHONUNBUFFERED=1 \
  "${SAM_ENV}/bin/python" services/sam_hq/server.py \
  --queue-dir "${QUEUE_DIR}" --checkpoint "${SAM_CHECKPOINT}" --device cuda \
  > "${LOG_DIR}/samhq.log" 2>&1 &
SAM_PID=$!

setsid env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
  "${FASTWAM_ENV}/bin/accelerate" launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes 4 scripts/resilient/train_two_stage_stage2.py \
  --config-name "${CONFIG_NAME}" \
  > "${LOG_DIR}/training.log" 2>&1 &
TRAIN_PID=$!

echo "Task-direction Outcome-FPO PID=${TRAIN_PID}; HQ-SAM PID=${SAM_PID}."
echo "Config: ${CONFIG_NAME}"
echo "Logs: ${LOG_DIR}"
wait "${TRAIN_PID}"
