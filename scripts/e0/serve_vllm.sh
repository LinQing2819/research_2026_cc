#!/usr/bin/env bash
# Serve the generation model behind an OpenAI-compatible endpoint for SeqMem-Eval.
# `--generation-config vllm` keeps vLLM's neutral sampling defaults so only the
# temperature passed by SeqMem-Eval applies (as in its in-process vLLM path).
# Weights are read from models/<name> when present (offline server), otherwise
# from the Hugging Face hub. The served name stays the hub id so run directories
# and reports do not depend on where the weights live.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ ! -w "${HOME:-/}" ]]; then
  export HOME="${ROOT}/.home"
  mkdir -p "${HOME}"
fi
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export TMPDIR="${TMPDIR:-${ROOT}/.cache/tmp}"
mkdir -p "${HF_HOME}" "${TMPDIR}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/${MODEL##*/}}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"

VLLM="${VLLM:-vllm}"
if [[ -x "${ROOT}/.venv-vllm/bin/vllm" && "${VLLM}" == vllm ]]; then
  VLLM="${ROOT}/.venv-vllm/bin/vllm"
fi

SOURCE="${MODEL}"
if [[ -d "${MODEL_PATH}" ]]; then
  SOURCE="${MODEL_PATH}"
  export HF_HUB_OFFLINE=1
fi

exec "${VLLM}" serve "${SOURCE}" \
  --port "${PORT}" \
  --served-model-name "${MODEL}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --generation-config vllm
