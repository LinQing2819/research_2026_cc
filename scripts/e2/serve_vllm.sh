#!/usr/bin/env bash
# Single 4090: explicit 8192 total window, one active sequence, CPU client retrieval.
set -euo pipefail
E2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${E2_ROOT}/scripts/e2/env.sh"
E2_PYTHON="${E2_PYTHON:-${E2_ROOT}/.venv-vllm/bin/python}"
E2_MODEL_PATH="${E2_MODEL_PATH:-${E2_ROOT}/models/Qwen3-8B}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ! -x "${E2_PYTHON}" ]]; then
  printf 'Missing vLLM interpreter: %s; set E2_PYTHON to .venv-vllm/bin/python.\n' "${E2_PYTHON}" >&2
  exit 2
fi
exec "${E2_PYTHON}" "${E2_ROOT}/scripts/e2/server.py" \
  --model "${E2_MODEL_PATH}" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port "${E2_PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 8192 --enforce-eager \
  --gpu-memory-utilization 0.90 --generation-config vllm \
  --disable-log-requests
