"""Download the model weights for an offline server (run where internet is available).

Honours ``HF_ENDPOINT`` (e.g. ``HF_ENDPOINT=https://hf-mirror.com``). Downloads
resume when re-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Directory receiving one sub-folder per model.")
    parser.add_argument("--only", action="append", choices=sorted(MODELS), default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing: pip install -e '.[pack]'", file=sys.stderr)
        return 1

    for name, repo_id in MODELS.items():
        if args.only and name not in args.only:
            continue
        dest = args.out / name
        print(f"[get ] {repo_id} -> {dest}")
        snapshot_download(repo_id=repo_id, local_dir=dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
