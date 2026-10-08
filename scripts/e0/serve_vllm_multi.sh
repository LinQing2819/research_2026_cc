#!/usr/bin/env bash
# Start one vLLM server per GPU. Defaults to GPUs 0-3 on ports 8000-8003.
# Logs: logs/vllm/gpu<id>_port<port>.log. Re-running stops the previous pids.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8000}"
LOG_DIR="${ROOT}/logs/vllm"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a ids <<< "${GPUS}"
ports=()
i=0
for gpu in "${ids[@]}"; do
  port=$((BASE_PORT + i))
  ports+=("${port}")
  pidfile="${LOG_DIR}/gpu${gpu}.pid"
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    kill "$(cat "${pidfile}")" || true
    sleep 1
  fi
  log="${LOG_DIR}/gpu${gpu}_port${port}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PORT="${port}" \
    nohup bash "${ROOT}/scripts/e0/serve_vllm.sh" > "${log}" 2>&1 &
  echo $! > "${pidfile}"
  echo "GPU ${gpu} -> http://127.0.0.1:${port}/v1  (pid $!, log ${log})"
  i=$((i + 1))
done

joined=$(IFS=,; echo "${ports[*]}")
echo "When every log shows 'Application startup complete':"
echo "  .venv-client/bin/python scripts/e0/run_matrix.py --ports ${joined}"
