#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
num_processes="${1:-8}"
if [[ "${num_processes}" != "4" && "${num_processes}" != "8" ]]; then
  echo "Usage: $0 [4|8] [Hydra overrides ...]" >&2
  exit 2
fi
shift || true

cd "${project_root}"
exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes "${num_processes}" \
  scripts/resilient/train_rf_opsd.py \
  "rf_opsd.distributed.num_processes=${num_processes}" \
  "$@"
