#!/usr/bin/env bash
# Invoke this script on the server, after extracting the portable package.
set -euo pipefail
stage_a_root=/data/zlq/research_2026_cc
stage_a_package="$(realpath "$(dirname "$0")")"
source "${stage_a_root}/scripts/e2/env.sh"
export SCREENDIR="${stage_a_root}/.cache/screen"
stage_a_run="${stage_a_root}/runs/v3-curated-9q-20261104"
stage_a_client="${stage_a_root}/.venv-client/bin/python"
mkdir -p "${SCREENDIR}" "${stage_a_run}"
chmod 700 "${SCREENDIR}"
command -v screen >/dev/null
test -x "${stage_a_client}"
test -x "${stage_a_root}/.venv-vllm/bin/python"
# Python verifies all transfer hashes without relying on caller's cwd.
"${stage_a_client}" - "${stage_a_package}" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
for line in (p / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
PY
"${stage_a_client}" "${stage_a_package}/stage-a.pyz" verify
"${stage_a_client}" "${stage_a_package}/stage-a.pyz" init
if screen -ls | grep -Eq '[.]stage-a-client[[:space:]]'; then
  printf 'Stage A client already has a screen session; inspect its persistent log.\n'
  exit 0
fi
# Reuse a live service on 8001 only after full client identity/token validation.
# If no service is listening, refuse to compete with an existing GPU workload.
if ! "${stage_a_client}" -c 'import socket; s=socket.create_connection(("127.0.0.1",8001),3); s.close()' 2>/dev/null; then
  if ! screen -ls | grep -Eq '[.]stage-a-service[[:space:]]'; then
    stage_a_gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "${stage_a_gpu_processes//[[:space:]]/}" ]]; then
      printf 'GPU already has a compute process. Inspect nvidia-smi before starting the service.\n' >&2
      exit 2
    fi
    screen -L -Logfile "${stage_a_run}/service.screen.log" -dmS stage-a-service \
      bash -c 'set -euo pipefail; source "$1/scripts/e2/env.sh"; export E2_PORT=8001; exec bash "$1/scripts/e2/serve_vllm.sh"' \
      _ "${stage_a_root}"
  fi
fi
screen -L -Logfile "${stage_a_run}/client.screen.log" -dmS stage-a-client \
  bash -c 'set -euo pipefail
source "$1/scripts/e2/env.sh"
# Only readiness metadata is polled; no generation probes or retries here.
"$1/.venv-client/bin/python" - <<'"'"'PY'"'"'
import time, urllib.request
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=5) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    print("Waiting for model service readiness...", flush=True)
    time.sleep(5)
else:
    raise SystemExit("Service did not become ready; inspect service.screen.log")
PY
exec "$1/.venv-client/bin/python" "$2/stage-a.pyz" run --base-url http://127.0.0.1:8001' \
  _ "${stage_a_root}" "${stage_a_package}"
printf 'Service and client run in screen. Logs and exports: %s\n' "${stage_a_run}"
