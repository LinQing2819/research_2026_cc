"""Immutable inputs and protocol constants for the V4 frozen experiment."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("OO", "ON", "NO", "NN", "ZE", "RZ", "ZZ")
FACTORS = dict(zip(ARMS, (("old", "old"), ("old", "new"), ("new", "old"),
                         ("new", "new"), ("empty", "new"),
                         ("new", "empty"), ("empty", "empty"))))
TIE_ORDER = ("NN", "OO", "ON", "NO")
TASK = "MMLU-Pro-Engineering"
MODEL = "Qwen/Qwen3-8B"
LIMITS = dict(context=8192, output=2048, safety=256, input=5888,
              rules=1024, experiences=3072, logical=2240, probes=2,
              retries=44, attempts=2286, per_request=3)
PARAMETERS = dict(protocol="V4", task=TASK, model=MODEL, temperature=0.7,
                  thinking=False, repeats=2, top_k=3, split_seed=20261020,
                  dev=64, test=96, old_fraction=0.2, new_fraction=0.6,
                  embedding="Qwen/Qwen3-Embedding-0.6B", delta=0.03,
                  limits=LIMITS)


class Invalid(RuntimeError):
    """Fail closed: this run cannot proceed with the available evidence."""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(path):
    path = Path(path)
    require(path.is_dir(), f"Required local model directory is missing: {path}")
    # Ignore download locks/cache metadata, but hash all actual model resources.
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
             if p.is_file() and ".cache" not in p.relative_to(path).parts
             and not p.name.startswith(".")}
    require(bool(files), f"Empty model directory: {path}")
    return files


def versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value, immutable=True):
    """Atomic publication + fsync; existing immutable files must match."""
    path = Path(path)
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == data, f"Immutable artifact changed: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def seal(value):
    return {"value": value, "sha256": digest(value)}


def unseal(envelope):
    require(set(envelope) == {"value", "sha256"}, "Invalid sealed artifact schema")
    require(digest(envelope["value"]) == envelope["sha256"], "Artifact seal mismatch")
    return envelope["value"]


def stable_seed(stage, qid, arm, rep):
    return int(digest(["e2-v4-decode", stage, qid, arm, rep])[:8], 16) % (2**31)


def logical_key(stage, qid, arm, rep):
    return canonical([stage, qid, arm, rep]).decode()
