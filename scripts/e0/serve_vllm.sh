#!/usr/bin/env bash
# Serve the generation model behind an OpenAI-compatible endpoint for SeqMem-Eval.
# `--generation-config vllm` keeps vLLM's neutral sampling defaults so only the
# temperature passed by SeqMem-Eval applies (as in its in-process vLLM path).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.80}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name "${MODEL}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --generation-config vllm
