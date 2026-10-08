"""Readers for SeqMem-Eval per-sample outputs.

SeqMem-Eval writes ``{task}_{method}_eval_log_readable.json`` (a JSON array,
one entry per processed sample, in stream order) into ``--output-dir``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List

import pandas as pd

_MAX_TEXT_CHARS = 4000


def run_dir(runs_root: Path, model_tag: str, log_name: str, rep: int) -> Path:
    return Path(runs_root) / model_tag / log_name / f"rep{rep}"


def eval_log_path(run_directory: Path, task: str, log_name: str) -> Path:
    return Path(run_directory) / f"{task}_{log_name}_eval_log_readable.json"


def _iter_dicts(obj: Any) -> Iterator[dict]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _memory_payload(entry: dict) -> List[Any]:
    """Everything that went into the prompt as memory.

    ``retrieved_memory`` is empty for methods that handle the whole trial
    themselves; those expose prompt-side state through ``extra_context``.
    """
    parts = []
    if entry.get("retrieved_memory"):
        parts.append(entry["retrieved_memory"])
    if entry.get("extra_context"):
        parts.append(entry["extra_context"])
    return parts


def _n_retrieved(retrieved: Any) -> int:
    if retrieved is None or retrieved == "":
        return 0
    if isinstance(retrieved, list):
        return len(retrieved)
    return 1


def load_eval_log(path: str | Path) -> pd.DataFrame:
    """Load one eval log into a per-sample frame ordered by stream position."""
    with Path(path).open("r", encoding="utf-8") as f:
        entries = json.load(f)

    rows = []
    for step, entry in enumerate(entries):
        payload = _memory_payload(entry)
        records = [d for d in _iter_dicts(payload) if "question" in d]
        texts = [s[:_MAX_TEXT_CHARS] for s in _iter_strings(payload) if s.strip()]
        question = entry.get("question")
        if not isinstance(question, str):
            question = json.dumps(question, ensure_ascii=False, default=str) if question else ""
        score = entry.get("score")
        score = float(score) if score is not None else 0.0
        rows.append(
            {
                "step": step,
                "qid": str(entry.get("qid")),
                "score": score,
                "correct": int(score >= 1.0),
                "question": question,
                "n_retrieved": _n_retrieved(entry.get("retrieved_memory")),
                "retrieved_questions": [str(r.get("question", "")) for r in records],
                "retrieved_feedback": [str(r.get("feedback", "")) for r in records if "feedback" in r],
                "memory_text": "\n".join(texts),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and df["qid"].duplicated().any():
        raise ValueError(f"Duplicate qids in {path}; cannot pair samples safely.")
    return df
