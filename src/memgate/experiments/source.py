"""Strict import of one historical trajectory; no online state restoration."""
from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from .common import PARAMETERS, TASK, digest, require
from .official import Official


def load_public(path):
    import pandas as pd
    path = Path(path)
    if path.suffix in (".json", ".jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = pd.read_parquet(path).to_dict("records")
    rows = [r for r in rows if r.get("category") == "engineering"]
    public = [Official.public(r) for r in rows]
    require(public and len({r["qid"] for r in public}) == len(public), "Duplicate/empty public IDs")
    gold = {}
    for row, item in zip(rows, public):
        require("question_id" in row, "Public question_id is required; no sequential fallback")
        letter = str(row.get("answer", "")).strip().upper()
        require(letter in "ABCDEFGHIJ"[:len(item["options"])] and len(letter) == 1,
                f"Invalid gold for {item['qid']}")
        if "answer_index" in row:
            require(int(row["answer_index"]) == ord(letter) - 65, "Gold fields disagree")
        gold[item["qid"]] = letter
    return public, gold


def validate_source(path, public, new_count):
    """The full validator only compares future public fields, not outcomes.

    Returns an exact prefix copy. The component constructor never sees suffix
    records, although the full file is hashed for integrity.
    """
    prefix, raw_prefix = [], []
    full_hash = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as stream:
        for raw in stream:
            full_hash.update(raw)
            require(bool(raw.strip()), "Blank/unknown source row")
            row = json.loads(raw)
            require(count < len(public), "Source has extra/duplicate rows")
            expected = public[count]
            require(isinstance(row, dict) and row.get("task") == TASK, "Wrong source task")
            require(str(row.get("qid")) == expected["qid"], f"Source order/ID mismatch at {count}")
            require(row.get("question") == expected["question"] and
                    row.get("options") == expected["options"], f"Source public text mismatch at {count}")
            if count < new_count:
                prefix.append(row)
                raw_prefix.append(raw.decode("utf-8"))
            count += 1
    require(count == len(public), f"Incomplete source: {count}/{len(public)}; .done is ignored")
    return prefix, "".join(raw_prefix), full_hash.hexdigest()


def rule_items(text):
    require(isinstance(text, str), "Rules must be source text")
    if text == "(empty)":
        return []
    lines = text.splitlines(keepends=True)
    require(bool(lines), "Unexpected empty rule schema")
    # Official normalization collapses whitespace; numbered source lines only.
    for index, line in enumerate(lines, 1):
        require(re.match(rf"^{index}\. \S", line) is not None, "Unknown numbered rule schema")
    return lines


def build_snapshots(prefix, old_count, new_count):
    require(len(prefix) == new_count and 0 < old_count < new_count, "Wrong prefix boundaries")
    pool, snapshots = [], {}
    before = "(empty)"
    successes = 0
    for i, row in enumerate(prefix, 1):
        meta, memory = row.get("memory_meta", {}), row.get("memory", {})
        require(isinstance(meta, dict) and isinstance(memory, dict), "Missing ExpeL schema")
        require(meta.get("method_variant") == "ExpeL-Online-ST" and
                meta.get("rollout_type") == "online_single_try", "Not the ST non-interactive path")
        require(meta.get("insights_before") == before, f"Broken rule chain at {i}")
        after = meta.get("insights_after")
        require(memory.get("insights") == after and
                meta.get("rules_count") == len(rule_items(after)), f"Rule mismatch at {i}")
        rollout = memory.get("rollout", {})
        require(isinstance(rollout, dict), f"Invalid rollout schema at {i}")
        attempts = rollout.get("attempts", [])
        require(isinstance(attempts, list) and len(attempts) == 1 and isinstance(attempts[0], dict),
                f"Invalid attempt schema at {i}")
        require(rollout.get("max_tries") == 1 and
                attempts[0].get("try") == 1 and rollout.get("reflections") == [],
                f"Invalid single-try rollout at {i}")
        trajectory = attempts[0].get("trajectory")
        require(isinstance(trajectory, str), f"Missing source trajectory at {i}")
        feedback = row.get("feedback")
        require(isinstance(feedback, str) and feedback.lower() in ("success", "failure"),
                f"Unknown feedback at {i}")
        success = feedback.lower() == "success"
        require(row.get("score") in (0, 1) and (row["score"] == 1) == success,
                f"Feedback/score disagreement at {i}")
        if success:
            successes += 1
            pool.append(dict(qid=str(row["qid"]), question=row["question"].strip(),
                             trajectory=trajectory))
        require(meta.get("experience_pool_size") == successes, f"Pool jump/missing seed at {i}")
        require(meta.get("recent_success_size") == successes % 8,
                f"Unexpected batch state at {i}; expected E0 batch_update_size=8")
        if i in (old_count, new_count):
            snapshots["old" if i == old_count else "new"] = {
                "completed": i, "rules": after, "pool": list(pool),
                "prefix_sha256": digest(prefix[:i]),
            }
        before = after
    return snapshots


def split_public(public):
    n = len(public)
    old, new = int(0.2 * n), int(0.6 * n)
    suffix = list(public[new:])
    random.Random(PARAMETERS["split_seed"]).shuffle(suffix)
    require(len(suffix) >= 160, "Insufficient fixed suffix for dev64/test96")
    return old, new, {"dev": suffix[:64], "test": suffix[64:160]}


def normalized(text):
    # Preserve numbers, punctuation and formula symbols. Only Unicode/space/case normalize.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def duplicate_audit(prefix_public, splits, threshold=0.92):
    """Freeze all cross-boundary candidates; any candidate stops preparation.

    Exact canonical content includes sorted options to detect reordered choices.
    A high character similarity is only a candidate, never a same-question claim.
    Unresolved candidates require a protocol review, not silent resampling.
    """
    groups = {"prefix": prefix_public, **splits}
    pairs, candidates = 0, []
    encoded = {name: [(r, normalized(r["question"]),
                       sorted(normalized(x) for x in r["options"])) for r in rows]
               for name, rows in groups.items()}
    for left, right in (("prefix", "dev"), ("prefix", "test"), ("dev", "test")):
        for a, qa, oa in encoded[left]:
            for b, qb, ob in encoded[right]:
                pairs += 1
                exact = qa == qb and oa == ob
                matcher = difflib.SequenceMatcher(None, qa, qb, autojunk=False)
                ratio = 0.0
                if exact or matcher.quick_ratio() >= threshold:
                    ratio = matcher.ratio()
                if exact or ratio >= threshold:
                    candidates.append(dict(left=[left, a["qid"]], right=[right, b["qid"]],
                                           exact=exact, question_similarity=ratio,
                                           same_unordered_options=oa == ob,
                                           status="confirmed_exact" if exact else "unresolved"))
    return dict(algorithm="NFKC-space-case; sorted options; SequenceMatcher(autojunk=False)",
                threshold=threshold, comparisons=pairs, candidates=candidates,
                public_only=True, qualified=not candidates)
