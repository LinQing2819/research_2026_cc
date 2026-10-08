#!/usr/bin/env bash
# Install the offline bundle on the server (no internet needed).
#
#   tar -xzf ~/offline_bundle/research_2026_cc.tar.gz -C ~
#   cd ~/research_2026_cc && bash scripts/offline/install_server.sh ~/offline_bundle
#
# Creates .venv-client (experiments, CPU torch) and .venv-vllm (generation server)
# inside the repo and links the bundle's models/ into the repo. Env: PYTHON (an
# existing python3.11 to use instead of the bundled Miniconda), CONDA_PREFIX_DIR.
set -euo pipefail

BUNDLE="$(cd "${1:?usage: install_server.sh <bundle_dir>}" && pwd)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${HOME}/miniconda3-py311}"

echo "== 1/5 verify bundle checksums"
(cd "${BUNDLE}" && sha256sum --quiet -c SHA256SUMS)

echo "== 2/5 python 3.11"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ ! -x "${CONDA_PREFIX_DIR}/bin/python3.11" ]]; then
    bash "${BUNDLE}"/installers/Miniconda3-py311_*-Linux-x86_64.sh -b -p "${CONDA_PREFIX_DIR}"
  fi
  PYTHON="${CONDA_PREFIX_DIR}/bin/python3.11"
fi
"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
  || { echo "Python 3.11 required (wheels are built for cp311)" >&2; exit 1; }

echo "== 3/5 client env (.venv-client)"
"${PYTHON}" -m venv "${ROOT}/.venv-client"
"${ROOT}/.venv-client/bin/pip" install --no-index --find-links "${BUNDLE}/wheels/client" \
  -r "${ROOT}/requirements/client.lock.txt"
"${ROOT}/.venv-client/bin/python" -c "import torch, sentence_transformers, transformers, networkx, pandas, sklearn; print('client ok, torch', torch.__version__)"

echo "== 4/5 vLLM env (.venv-vllm)"
"${PYTHON}" -m venv "${ROOT}/.venv-vllm"
"${ROOT}/.venv-vllm/bin/pip" install --no-index --find-links "${BUNDLE}/wheels/vllm" \
  -r "${ROOT}/requirements/vllm.lock.txt"
"${ROOT}/.venv-vllm/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, '| cuda available:', torch.cuda.is_available())"

echo "== 5/5 models"
if [[ ! -e "${ROOT}/models" ]]; then
  ln -s "${BUNDLE}/models" "${ROOT}/models"
fi
ls "${ROOT}/models"

(cd "${ROOT}" && .venv-client/bin/python -m pytest -q)
echo "Done. Next: bash scripts/e0/serve_vllm.sh (see README, section 离线服务器)."
