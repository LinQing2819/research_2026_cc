"""Download the public datasets used by SeqMem-Eval into ``data/seqmem/``.

Layout matches what ``scripts/e0/run_matrix.py`` passes to SeqMem-Eval via
``--task-data-override``, so the pinned submodule stays untouched.
Honours ``HF_ENDPOINT`` (e.g. ``https://hf-mirror.com``) and ``GITHUB_RAW``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "seqmem"

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/datasets"
GITHUB_RAW = os.environ.get("GITHUB_RAW", "https://raw.githubusercontent.com").rstrip("/")
GORILLA = f"{GITHUB_RAW}/ShishirPatil/gorilla/main/data"

FILES = {
    "MATH500/test-2.jsonl": [
        f"{HF}/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500/resolve/master/test.jsonl",
    ],
    "MMLU-Pro/test.parquet": [f"{HF}/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"],
    "HumanEval/test.parquet": [f"{HF}/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"],
    "APIBench/huggingface_eval.json": [f"{GORILLA}/apibench/huggingface_eval.json"],
    "APIBench/huggingface_api.jsonl": [f"{GORILLA}/api/huggingface_api.jsonl"],
    "APIBench/tensorflow_eval.json": [f"{GORILLA}/apibench/tensorflow_eval.json"],
    "APIBench/tensorflowhub_api.jsonl": [f"{GORILLA}/api/tensorflowhub_api.jsonl"],
    "APIBench/torchhub_eval.json": [f"{GORILLA}/apibench/torchhub_eval.json"],
    "APIBench/torchhub_api.jsonl": [f"{GORILLA}/api/torchhub_api.jsonl"],
}
ATTEMPTS = 5


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``. A dropped connection raises and leaves no file."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            expected = int(response.headers.get("Content-Length") or 0)
            written = 0
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        if expected and written != expected:
            raise IOError(f"truncated: got {written} of {expected} bytes")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    failed = []
    for rel, urls in FILES.items():
        dest = DATA_ROOT / rel
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            print(f"[skip] {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error = "no url attempted"
        for attempt in range(1, ATTEMPTS + 1):
            url = urls[(attempt - 1) % len(urls)]
            print(f"[get ] {rel} <- {url} (try {attempt}/{ATTEMPTS})")
            try:
                download(url, dest)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror or attempt
                last_error = str(exc)
                print(f"[fail] {rel}: {exc}", file=sys.stderr)
                time.sleep(min(5 * attempt, 20))
        if last_error:
            failed.append(rel)
    if failed:
        print(f"{len(failed)} file(s) failed: {failed}", file=sys.stderr)
        return 1
    print(f"Datasets ready under {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
