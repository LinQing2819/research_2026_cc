#!/usr/bin/env bash
# Source in each client shell; serve_vllm.sh also loads this before Python imports.
# All runtime cache/config directories live on the repository's data volume.
E2_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${E2_ENV_ROOT}/.cache/huggingface"
export TMPDIR="${E2_ENV_ROOT}/.cache/tmp"
export MPLCONFIGDIR="${E2_ENV_ROOT}/.cache/matplotlib"
export PIP_CACHE_DIR="${E2_ENV_ROOT}/.cache/pip"
export XDG_CACHE_HOME="${E2_ENV_ROOT}/.cache"
export XDG_CONFIG_HOME="${E2_ENV_ROOT}/.home/.config"
export XDG_DATA_HOME="${E2_ENV_ROOT}/.home/.local/share"
mkdir -p "${HF_HOME}" "${TMPDIR}" "${MPLCONFIGDIR}" "${PIP_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" || return 1
