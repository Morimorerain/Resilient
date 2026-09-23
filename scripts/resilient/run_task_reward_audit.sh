#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAM_GPU="${1:-0}"
AUDIT_GPUS="${2:-1,2,3,4}"
SAM_ENV="${SAM_ENV:-${PROJECT_ROOT}/.venv-samhq}"
FASTWAM_ENV="${FASTWAM_ENV:-${PROJECT_ROOT}/AILOG/envs/resilient-fastwam-uv}"
QUEUE_DIR="${QUEUE_DIR:-${PROJECT_ROOT}/data/.cache/task_decoupling_rpc_samhq}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sam_hq/sam_hq_vit_h.pth}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/AILOG/processes/gate4_task_reward}"

IFS=',' read -r -a AUDIT_GPU_ARRAY <<< "${AUDIT_GPUS}"
if [[ "${#AUDIT_GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "AUDIT_GPUS must contain exactly four comma-separated GPU indices." >&2
  exit 2
fi
for gpu in "${AUDIT_GPU_ARRAY[@]}"; do
  if [[ "${gpu}" == "${SAM_GPU}" ]]; then
    echo "SAM_GPU must not also appear in AUDIT_GPUS." >&2
    exit 2
  fi
done
for required in \
  "${SAM_ENV}/bin/python" \
  "${FASTWAM_ENV}/bin/accelerate" \
  "${SAM_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required Gate-4 asset is missing: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}" "${QUEUE_DIR}"
SAM_PID=""
AUDIT_PID=""

stop_process_group() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_process_group "${AUDIT_PID}"
  stop_process_group "${SAM_PID}"
  [[ -z "${AUDIT_PID}" ]] || wait "${AUDIT_PID}" 2>/dev/null || true
  [[ -z "${SAM_PID}" ]] || wait "${SAM_PID}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "${PROJECT_ROOT}"
setsid env CUDA_VISIBLE_DEVICES="${SAM_GPU}" PYTHONUNBUFFERED=1 \
  "${SAM_ENV}/bin/python" services/sam_hq/server.py \
  --queue-dir "${QUEUE_DIR}" \
  --checkpoint "${SAM_CHECKPOINT}" \
  --device cuda > "${LOG_DIR}/samhq.log" 2>&1 &
SAM_PID=$!

setsid env CUDA_VISIBLE_DEVICES="${AUDIT_GPUS}" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
  "${FASTWAM_ENV}/bin/accelerate" launch \
  --config_file scripts/accelerate_configs/accelerate_inference_multi_gpu.yaml \
  --num_processes 4 scripts/resilient/audit_task_reward.py \
  > "${LOG_DIR}/audit.log" 2>&1 &
AUDIT_PID=$!

echo "Gate-4 audit PID=${AUDIT_PID}; HQ-SAM PID=${SAM_PID}."
echo "Logs: ${LOG_DIR}"
wait "${AUDIT_PID}"
