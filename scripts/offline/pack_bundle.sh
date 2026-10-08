#!/usr/bin/env bash
# Build everything an offline Linux x86_64 server needs, on a machine WITH internet
# (macOS or Linux). Copy the resulting directory to the server, then run
# scripts/offline/install_server.sh there.
#
#   bash scripts/offline/pack_bundle.sh
#
# Env: OUT (bundle dir), PYTHON (interpreter with pip >= 23 and huggingface_hub),
#      HF_ENDPOINT (e.g. https://hf-mirror.com), SKIP_MODELS=1, SKIP_WHEELS=1.
# Re-running resumes: existing wheels, models and data files are kept.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-${ROOT}/offline_bundle}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
PY_VER="3.11"
MINICONDA="Miniconda3-py311_25.1.1-2-Linux-x86_64.sh"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Lock files are resolved for x86_64-manylinux_2_28, so accept every tag up to glibc 2.28.
PLATFORMS=(--platform manylinux1_x86_64 --platform manylinux2010_x86_64 --platform manylinux2014_x86_64)
for minor in $(seq 5 28); do PLATFORMS+=(--platform "manylinux_2_${minor}_x86_64"); done
PIP_TARGET=(--no-deps --only-binary=:all: --python-version "${PY_VER}" --implementation cp "${PLATFORMS[@]}")

mkdir -p "${OUT}/wheels/client" "${OUT}/wheels/vllm" "${OUT}/models" "${OUT}/installers"

echo "== 1/5 datasets"
"${PYTHON}" "${ROOT}/scripts/setup/download_seqmem_data.py"

if [[ "${SKIP_WHEELS:-0}" != 1 ]]; then
  echo "== 2/5 wheels: client"
  "${PYTHON}" -m pip download "${PIP_TARGET[@]}" --extra-index-url "${TORCH_CPU_INDEX}" \
    -r "${ROOT}/requirements/client.lock.txt" -d "${OUT}/wheels/client"
  echo "== 2/5 wheels: vllm"
  "${PYTHON}" -m pip download "${PIP_TARGET[@]}" \
    -r "${ROOT}/requirements/vllm.lock.txt" -d "${OUT}/wheels/vllm"
fi

if [[ "${SKIP_MODELS:-0}" != 1 ]]; then
  echo "== 3/5 models"
  "${PYTHON}" "${ROOT}/scripts/offline/fetch_models.py" --out "${OUT}/models"
fi

echo "== 4/5 python installer"
if [[ ! -s "${OUT}/installers/${MINICONDA}" ]]; then
  curl -fL -o "${OUT}/installers/${MINICONDA}.part" "https://repo.anaconda.com/miniconda/${MINICONDA}"
  mv "${OUT}/installers/${MINICONDA}.part" "${OUT}/installers/${MINICONDA}"
fi

echo "== 5/5 code + data archive"
# COPYFILE_DISABLE stops macOS tar from adding ._* metadata files.
COPYFILE_DISABLE=1 tar -czf "${OUT}/research_2026_cc.tar.gz" -C "$(dirname "${ROOT}")" \
  --exclude ".venv" --exclude ".venv-*" --exclude "offline_bundle" --exclude "models" \
  --exclude "runs" --exclude "reports" --exclude "logs" \
  --exclude "__pycache__" --exclude ".pytest_cache" --exclude "*.egg-info" --exclude ".DS_Store" \
  "$(basename "${ROOT}")"

(cd "${OUT}" && find . -type f ! -name SHA256SUMS ! -path "*/.cache/*" | sort | xargs shasum -a 256 > SHA256SUMS)
du -sh "${OUT}"/* | sed 's/^/   /'
echo "Bundle ready: ${OUT}"
echo "Copy it to the server, e.g.: rsync -av --progress ${OUT}/ user@server:~/offline_bundle/"
