#!/usr/bin/env bash
# Export historical assets without changing the completed E2 checkout.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'Usage: bash export_server.sh /absolute/repo /absolute/tool.pyz /absolute/new-output-dir\n' >&2
  exit 2
fi
E4_REPO=$1
E4_TOOL=$2
E4_OUTPUT_DIR=$3
for E4_PATH in "$E4_REPO" "$E4_TOOL" "$E4_OUTPUT_DIR"; do
  if [[ "$E4_PATH" != /* ]]; then
    printf 'All three paths must be absolute: %s\n' "$E4_PATH" >&2
    exit 2
  fi
done
cd "$E4_REPO"
source scripts/e2/env.sh
test -x .venv-client/bin/python
test -f "$E4_TOOL"
# An existing output directory is evidence, never a place to restart over.
mkdir "$E4_OUTPUT_DIR"
(
  set -e
  .venv-client/bin/python "$E4_TOOL" version
  .venv-client/bin/python "$E4_TOOL" export \
    --repo "$E4_REPO" --output "$E4_OUTPUT_DIR/assets.json.gz"
  .venv-client/bin/python "$E4_TOOL" inspect \
    --input "$E4_OUTPUT_DIR/assets.json.gz"
  printf 'E4_ASSET_EXPORT_COMPLETE: %s\n' "$E4_OUTPUT_DIR/assets.json.gz"
) 2>&1 | tee "$E4_OUTPUT_DIR/run.log"
