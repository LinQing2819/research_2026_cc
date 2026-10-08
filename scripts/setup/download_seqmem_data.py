"""Download the public datasets used by SeqMem-Eval into ``data/seqmem/``.

Layout matches what ``scripts/e0/run_matrix.py`` passes to SeqMem-Eval via
``--task-data-override``, so the pinned submodule stays untouched.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "seqmem"

HF = "https://huggingface.co/datasets"
GORILLA = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data"

FILES = {
    "MATH500/test-2.jsonl": f"{HF}/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
    "MMLU-Pro/test.parquet": f"{HF}/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
    "HumanEval/test.parquet": f"{HF}/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet",
    "APIBench/huggingface_eval.json": f"{GORILLA}/apibench/huggingface_eval.json",
    "APIBench/huggingface_api.jsonl": f"{GORILLA}/api/huggingface_api.jsonl",
    "APIBench/tensorflow_eval.json": f"{GORILLA}/apibench/tensorflow_eval.json",
    "APIBench/tensorflowhub_api.jsonl": f"{GORILLA}/api/tensorflowhub_api.jsonl",
    "APIBench/torchhub_eval.json": f"{GORILLA}/apibench/torchhub_eval.json",
    "APIBench/torchhub_api.jsonl": f"{GORILLA}/api/torchhub_api.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    failed = []
    for rel, url in FILES.items():
        dest = DATA_ROOT / rel
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            print(f"[skip] {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"[get ] {rel} <- {url}")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        except Exception as exc:  # noqa: BLE001 - report and continue with other files
            print(f"[fail] {rel}: {exc}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            failed.append(rel)
    if failed:
        print(f"{len(failed)} file(s) failed: {failed}", file=sys.stderr)
        return 1
    print(f"Datasets ready under {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
