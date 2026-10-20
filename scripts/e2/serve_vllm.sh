#!/usr/bin/env bash
# Single 4090: explicit 8192 total window, one active sequence, CPU client retrieval.
set -euo pipefail
E2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
E2_PYTHON="${E2_PYTHON:-${E2_ROOT}/.venv/bin/python}"
E2_MODEL_PATH="${E2_MODEL_PATH:-${E2_ROOT}/models/Qwen3-8B}"
export HF_HOME="${HF_HOME:-${E2_ROOT}/.cache/huggingface}"
export TMPDIR="${TMPDIR:-${E2_ROOT}/.cache/tmp}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${HF_HOME}" "${TMPDIR}"
exec "${E2_PYTHON}" "${E2_ROOT}/scripts/e2/server.py" \
  --model "${E2_MODEL_PATH}" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port "${E2_PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 8192 --enforce-eager \
  --gpu-memory-utilization 0.90 --generation-config vllm \
  --disable-log-requests
