#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PATH="${1:-${PROJECT_ROOT}/AILOG/envs/resilient-fastwam-robotwin}"
PYTHON_SELECTOR="${RESILIENT_PYTHON_PATH:-3.10.20}"
CUDA_TOOLKIT_ROOT="${RESILIENT_CUDA_HOME:-/usr/local/cuda-12.8}"

cd "${PROJECT_ROOT}"
uv python install 3.10.20
uv venv --python "${PYTHON_SELECTOR}" "${ENV_PATH}"
uv pip install --python "${ENV_PATH}/bin/python" -r requirements-robotwin-build.txt
uv pip install \
  --python "${ENV_PATH}/bin/python" \
  --index-strategy unsafe-best-match \
  -r requirements-robotwin.txt
CUDA_HOME="${CUDA_TOOLKIT_ROOT}" uv pip install \
  --python "${ENV_PATH}/bin/python" \
  --no-build-isolation \
  -r requirements-robotwin-curobo.txt
uv pip install --python "${ENV_PATH}/bin/python" --no-deps -e .
uv pip check --python "${ENV_PATH}/bin/python"
