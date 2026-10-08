#!/usr/bin/env bash
# Install on a server that can reach a PyPI mirror and hf-mirror.
# Everything (Miniconda, both virtualenvs, model weights, datasets, caches, temp
# files) is created inside this repo, so the repo itself must sit on a disk with
# >= 40 GB free. Do not run this from a home directory on a full system disk.
#
#   bash scripts/setup/install_on_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ ! -w "${HOME:-/}" ]]; then
  export HOME="${ROOT}/.home"
  mkdir -p "${HOME}"
  echo "Home directory is not writable; using ${HOME}"
fi
INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TRUSTED="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CACHE_DIR="${ROOT}/.cache/pip"
export HF_HOME="${ROOT}/.cache/huggingface"
export TMPDIR="${ROOT}/.cache/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"

if [[ ! -f "${ROOT}/third_party/SeqMem-Eval/main.py" ]]; then
  echo "third_party/SeqMem-Eval is missing. Copy the whole repo, including third_party/." >&2
  exit 1
fi

avail_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"
mountpoint="$(df -P "${ROOT}" | awk 'NR==2 {print $6}')"
if [[ "${avail_kb}" -lt 40000000 ]]; then
  echo "Only $((avail_kb / 1024 / 1024)) GB free on ${mountpoint}. Move the repo to a data disk." >&2
  exit 1
fi

if [[ ! -x "${ROOT}/.miniconda/bin/python3.11" ]]; then
  echo "== Miniconda (Python 3.11) -> ${ROOT}/.miniconda"
  curl -fL -o "${TMPDIR}/miniconda.sh" \
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_25.1.1-2-Linux-x86_64.sh"
  bash "${TMPDIR}/miniconda.sh" -b -p "${ROOT}/.miniconda"
fi
PY="${ROOT}/.miniconda/bin/python3.11"

echo "== client env (.venv-client)"
"${PY}" -m venv "${ROOT}/.venv-client"
"${ROOT}/.venv-client/bin/pip" install -U pip -i "${INDEX}" --trusted-host "${TRUSTED}"
grep -v '^torch' "${ROOT}/requirements/client.in" > "${TMPDIR}/client-req.txt"
"${ROOT}/.venv-client/bin/pip" install -i "${INDEX}" --trusted-host "${TRUSTED}" \
  torch huggingface_hub -r "${TMPDIR}/client-req.txt"

echo "== vLLM env (.venv-vllm)"
"${PY}" -m venv "${ROOT}/.venv-vllm"
"${ROOT}/.venv-vllm/bin/pip" install -U pip -i "${INDEX}" --trusted-host "${TRUSTED}"
"${ROOT}/.venv-vllm/bin/pip" install -i "${INDEX}" --trusted-host "${TRUSTED}" \
  -r "${ROOT}/requirements/vllm.in"

echo "== models + datasets"
"${ROOT}/.venv-client/bin/python" "${ROOT}/scripts/offline/fetch_models.py" --out "${ROOT}/models"
"${ROOT}/.venv-client/bin/python" "${ROOT}/scripts/setup/download_seqmem_data.py"

(cd "${ROOT}" && .venv-client/bin/python -m pytest -q)
echo "Ready. Next, inside screen: CUDA_VISIBLE_DEVICES=0 bash scripts/e0/serve_vllm.sh"
