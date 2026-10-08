import json
from pathlib import Path

import pytest


def write_eval_log(path: Path, outcomes, with_memory: bool) -> Path:
    """Write a minimal SeqMem-Eval-format eval log with the given 0/1 outcomes."""
    entries = []
    for i, y in enumerate(outcomes):
        retrieved = (
            [{"question": f"past question {i - 1} about topic {i % 3}", "feedback": "failure" if i % 2 else "success"}]
            if with_memory and i > 0
            else []
        )
        entries.append(
            {
                "qid": f"q{i}",
                "task": "T",
                "method": "M",
                "question": f"question {i} about topic {i % 3}",
                "retrieved_memory": retrieved,
                "model_output": "x",
                "score": float(y),
                "correct": bool(y),
            }
        )
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture
def make_log(tmp_path):
    def _make(name, outcomes, with_memory=False):
        return write_eval_log(tmp_path / name, outcomes, with_memory)

    return _make
