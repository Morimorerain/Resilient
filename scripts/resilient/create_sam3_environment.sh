#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_path="${1:-${project_root}/.venv-sam3}"
cache_dir="${RESILIENT_SAM3_CACHE_DIR:-${project_root}/AILOG/caches/uv-sam3}"
temporary_dir="${RESILIENT_SAM3_TMPDIR:-${project_root}/AILOG/tmp/sam3-install}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install the version documented in environment/README.md." >&2
  exit 2
fi

mkdir -p "${cache_dir}" "${temporary_dir}"

UV_CACHE_DIR="${cache_dir}" TMPDIR="${temporary_dir}" uv python install 3.12.13
if [[ ! -x "${environment_path}/bin/python" ]]; then
  UV_CACHE_DIR="${cache_dir}" TMPDIR="${temporary_dir}" \
    uv venv --python 3.12.13 "${environment_path}"
fi
UV_CACHE_DIR="${cache_dir}" TMPDIR="${temporary_dir}" \
  uv pip install --python "${environment_path}/bin/python" \
  --index-strategy unsafe-best-match \
  -r "${project_root}/requirements-sam3.txt"

echo "SAM 3.1 environment created at ${environment_path}"
echo "uv cache: ${cache_dir}"
echo "Request checkpoint access before starting services/sam3/server.py."
