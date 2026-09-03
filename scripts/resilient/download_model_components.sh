#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

clone_sparse_model() {
  local repository_url="$1"
  local commit="$2"
  local target="$3"
  local lfs_include="$4"
  shift 4

  if [[ -d "$target/.git" ]]; then
    local actual_commit
    actual_commit="$(git -C "$target" rev-parse HEAD)"
    if [[ "$actual_commit" != "$commit" ]]; then
      echo "Commit mismatch at $target: expected $commit, found $actual_commit" >&2
      return 1
    fi
  else
    if [[ -e "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to replace non-empty path: $target" >&2
      return 1
    fi
    rmdir "$target" 2>/dev/null || true
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$repository_url" "$target"
    git -C "$target" sparse-checkout set --no-cone "$@"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$target" checkout --detach "$commit"
  fi

  git -C "$target" lfs pull --include="$lfs_include" --exclude=""
}

clone_sparse_model \
  "https://www.modelscope.cn/DiffSynth-Studio/Wan-Series-Converted-Safetensors.git" \
  "150f75d811d51f6c7760154aa7fec371dccda529" \
  "$PROJECT_ROOT/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
  "models_t5_umt5-xxl-enc-bf16.safetensors,Wan2.2_VAE.safetensors" \
  "models_t5_umt5-xxl-enc-bf16.safetensors" \
  "Wan2.2_VAE.safetensors"

clone_sparse_model \
  "https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-1.3B.git" \
  "020b9e59db399efa534e408fb295b14a75700daf" \
  "$PROJECT_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B" \
  "google/umt5-xxl/*" \
  "google/umt5-xxl/*"
