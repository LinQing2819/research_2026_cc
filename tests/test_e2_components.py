"""Offline V4 source/retrieval/prompt/packing tests using real official definitions."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from memgate.experiments.common import ARMS, ROOT, Invalid, digest
from memgate.experiments.official import Official, definitions
from memgate.experiments.prompts import (CASE_HEADER, RULE_HEADER, chat, checked_vectors,
                                        pack_cases, pack_rules, prepare_question, retrieve)
from memgate.experiments.source import (build_snapshots, duplicate_audit, split_public,
                                       validate_source)


class TinyTokenizer:
    """Deterministic fake BPE with a cross-boundary double-newline merge."""
    is_fast = True

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        i = 0
        while i < len(text):
            width = 2 if text[i:i+2] == "\n\n" else 1
            ids.append(900001 if width == 2 else ord(text[i])+100)
            offsets.append((i, i+width))
            i += width
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "enable_thinking": False}
        text = "<user>\n"+messages[0]["content"]+"\n</user>\n<assistant>\n<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


@pytest.fixture
def official():
    return Official(ROOT / "third_party/SeqMem-Eval")


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


def public_rows(n=969):
    return [dict(qid=str(i), question=digest(["question", i]),
                 options=[digest(["option", i, j])[:12] for j in range(4)]) for i in range(n)]


def source_rows(public):
    rows, count, before = [], 0, "(empty)"
    for i, p in enumerate(public, 1):
        success = i % 2 == 0
        count += success
        after = "(empty)" if count < 8 else ("1. Check physical units." if i < 416 else
                                            "1. Recheck dimensions.\n2. Verify the final answer.")
        row = dict(task="MMLU-Pro-Engineering", **p, score=int(success),
                   gold_letter="A", predicted_letter="A" if success else "B",
                   feedback="success" if success else "failure", model_output="WRONG fallback sentinel",
                   memory=dict(insights=after, rollout=dict(max_tries=1, reflections=[],
                               attempts=[{"try": 1, "trajectory": f"Historical solution {p['qid']}\n\\boxed{{A}}"}])),
                   memory_meta=dict(method_variant="ExpeL-Online-ST", rollout_type="online_single_try",
                                    insights_before=before, insights_after=after,
                                    rules_count=0 if after == "(empty)" else len(after.splitlines()),
                                    experience_pool_size=count, recent_success_size=count % 8))
        rows.append(row)
        before = after
    return rows


def write_source(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def test_t01_exact_prefix_and_full_source_identity(tmp_path):
    public = public_rows()
    rows = source_rows(public)
    # A future payload is present in the source but cannot reach the constructor.
    rows[581]["memory"] = {"forbidden_future": "FUTURE_ANSWER"}
    path = tmp_path / "source.jsonl"
    write_source(path, rows)
    old, new, split = split_public(public)
    assert (old, new, len(split["dev"]), len(split["test"])) == (193, 581, 64, 96)
    prefix, raw, _ = validate_source(path, public, new)
    assert len(prefix) == 581 and "FUTURE_ANSWER" not in raw
    snapshots = build_snapshots(prefix, old, new)
    assert snapshots["old"]["completed"] == 193
    assert snapshots["new"]["rules"] == rows[580]["memory_meta"]["insights_after"]
    assert snapshots["new"]["pool"][-1]["qid"] == "579"
    for changed in (rows[1:], rows[:-1], rows+[rows[-1]], [rows[1], rows[0]]+rows[2:]):
        write_source(path, changed)
        (tmp_path / ".done").touch()
        with pytest.raises(Invalid):
            validate_source(path, public, new)
    changed = copy.deepcopy(rows)
    changed[3]["question"] = "same qid, wrong question"
    write_source(path, changed)
    with pytest.raises(Invalid, match="text mismatch"):
        validate_source(path, public, new)


@pytest.mark.parametrize("mutation", [
    lambda r: r["memory"].pop("rollout"),
    lambda r: r.update(score=0),
    lambda r: r["memory_meta"].update(experience_pool_size=100),
    lambda r: r["memory_meta"].update(insights_before="broken"),
    lambda r: r["memory_meta"].update(rules_count=20),
    lambda r: r["memory_meta"].update(recent_success_size=7),
    lambda r: r["memory_meta"].update(rollout_type="online_multi_try"),
])
def test_t02_bad_frozen_state_is_rejected(mutation):
    rows = source_rows(public_rows(30))
    mutation(rows[1])
    with pytest.raises(Invalid):
        build_snapshots(rows, 5, 30)


def test_t02_trajectory_and_rule_text_preserved():
    rows = source_rows(public_rows(30))
    rows[1]["memory"]["rollout"]["attempts"][0]["trajectory"] = "  <trajectory> literal saved text </trajectory>  "
    snapshots = build_snapshots(rows, 20, 30)
    assert snapshots["old"]["pool"][0]["trajectory"] == "  <trajectory> literal saved text </trajectory>  "
    assert snapshots["new"]["rules"] == rows[-1]["memory"]["insights"]
    assert "WRONG fallback" not in json.dumps(snapshots)


def test_t03_real_official_retrieval_ties_and_short_pool(official):
    pool = [dict(qid=str(i), question=f"task{i}", trajectory=f"legal answer{i}") for i in range(4)]
    vectors = [[1, 0], [1, 0], [0, 1], [-1, 0]]
    actual = retrieve(official, pool, vectors, [1, 0], "task\nA. first\nB. second")
    assert [r["qid"] for r in actual] == ["0", "1", "2"]
    assert actual[0]["retrieval_score"] == 1.0
    assert retrieve(official, [], [], [1, 0], "query") == []
    assert len(retrieve(official, pool[:1], vectors[:1], [1, 0], "query")) == 1
    for bad in ([[float("nan"), 0]], [[0, 0]], [[2, 0]]):
        with pytest.raises(Invalid):
            checked_vectors(bad, 1)


def tiny_snapshots():
    old_pool = [dict(qid="10", question="Earlier question", trajectory="Solution\n\\boxed{A}")]
    return {"old": dict(rules="1. Old rule.", pool=old_pool),
            "new": dict(rules="1. New rule.\n2. Another rule.",
                        pool=old_pool+[dict(qid="11", question="More recent question", trajectory="Reason\n\\boxed{B}")])}


def test_t04_t05_real_builder_fidelity_and_public_projection(official, tokenizer):
    row = dict(qid="99", question="Unique question α?", options=["one", "two"],
               gold_letter="GOLD_SENTINEL", cot_content="COT_SENTINEL",
               model_output="FUTURE_SENTINEL")
    public = Official.public(row)
    snap = tiny_snapshots()
    result = prepare_question(official, tokenizer, public, snap,
                              {"old": [[1, 0]], "new": [[1, 0], [0, 1]]}, [0, 1])
    assert result["query"].endswith("A. one\nB. two")
    for arm, record in result["prompts"].items():
        assert not any(s in record["prompt"] for s in ("GOLD_SENTINEL", "COT_SENTINEL", "FUTURE_SENTINEL"))
        assert record["prompt"].endswith(official.problem(public)+"\n")
        assert record["unchanged_from_official_unpacked"]
    # Independently load and invoke the original function body, not a second new builder.
    funcs = definitions(official.root / "methods/expel_online_st.py",
                        ["_build_solver_prompt", "_build_fewshot_examples_block"], "ExpelOnlineST")
    class Ref:
        pass
    ref = Ref()
    ref._build_fewshot_examples_block = funcs["_build_fewshot_examples_block"].__get__(ref)
    for arm, version in (("OO", "old"), ("NN", "new")):
        expected = funcs["_build_solver_prompt"](ref, result["problem"], snap[version]["rules"],
                                                  result["ranked"][version])
        assert result["prompts"][arm]["prompt"].encode() == expected.encode()
    assert "(none)" in result["prompts"]["ZZ"]["prompt"]
    assert "(empty)" in result["prompts"]["ZE"]["prompt"]


def test_t05_duplicate_and_reordered_options_stop():
    a = dict(qid="1", question="What is the force of this circuit?", options=["12", "15"])
    b = {**a, "qid": "2", "options": ["15", "12"]}
    audit = duplicate_audit([a], {"dev": [b], "test": []})
    assert audit["candidates"][0]["exact"] and not audit["qualified"]
    c = {**b, "question": "What is the force of that circuit?"}
    audit = duplicate_audit([a], {"dev": [c], "test": []})
    assert audit["candidates"][0]["status"] == "unresolved"
    assert "12" in str(a["options"])  # numeric distinctions aren't normalized away


def test_t06_prefix_packing_keeps_complete_items_and_never_borrows(official, tokenizer):
    rules = "1. "+"x"*1500+"\n2. Short."
    assert pack_rules(tokenizer, rules)["text"] == "(empty)"
    short = "1. Small."
    exact = len(tokenizer.encode(RULE_HEADER+short))
    assert pack_rules(tokenizer, short, exact)["kept"] == 1
    assert pack_rules(tokenizer, short, exact-1)["kept"] == 0
    cases = [dict(qid="1", question="past", trajectory="x"*4000, retrieval_score=1.0),
             dict(qid="2", question="past2", trajectory="tiny", retrieval_score=.1)]
    assert pack_cases(tokenizer, official, cases)["items"] == []
    assert len(tokenizer.encode("\n"))*2 != len(tokenizer.encode("\n\n"))
    # No arbitrary punctuation split inside a numbered rule.
    assert pack_rules(tokenizer, "1. First. Still the first rule.")["kept"] == 1


def test_t07_final_limit_common_exclusion(official, tokenizer):
    row = dict(qid="7", question="x", options=["a", "b"])
    snap = {"old": {"rules": "(empty)", "pool": []}, "new": {"rules": "(empty)", "pool": []}}
    vectors = {"old": [], "new": []}
    baseline = prepare_question(official, tokenizer, row, snap, vectors, [1, 0])
    count = baseline["prompts"]["ZZ"]["prompt_tokens"]
    row["question"] += "x"*(5888-count)
    exact = prepare_question(official, tokenizer, row, snap, vectors, [1, 0])
    assert exact["eligible"] and exact["prompts"]["ZZ"]["prompt_tokens"] == 5888
    row["question"] += "x"
    over = prepare_question(official, tokenizer, row, snap, vectors, [1, 0])
    assert not over["eligible"] and set(over["exclusion"]["arms"]) == set(ARMS)


def test_t13_official_scoring_and_zero_contrasts(official):
    from memgate.experiments.inference import analyze_scores
    from memgate.experiments.study import score_function
    score = score_function(official, "C")
    assert score("Reasoning. \\boxed{C}")["score"] == 1
    assert score("The answer is C")["score"] == 1
    assert score("no parseable answer")["score"] == 0
    rows = [dict(qid="1", arm=a, rep=r, score=int(a == "ON")) for a in ARMS for r in range(2)]
    report = analyze_scores(["1"], rows, "ON", "OO")
    assert report["contrasts"]["D_ON"]["ci"] == [0, 0]
    assert report["contrasts"]["C1"]["query_variance"] is None
    assert report["contrasts"]["C1"]["half_width"] == pytest.approx(math.sqrt(2*math.log(240)/4))
    assert report["contrasts"]["C1"]["decode_variance"] == 0
    assert analyze_scores(["1"], rows, "NN", "NN")["contrasts"]["D_F"]["ci"] == [0, 0]
    for bad in ([], rows[:-1], rows+[rows[0]]):
        with pytest.raises(Invalid):
            analyze_scores(["1"], bad, "ON", "OO")
    with pytest.raises(Invalid):
        analyze_scores([], [], "ON", "OO")


def test_t13_variance_and_strict_decision_threshold():
    from memgate.experiments.inference import analyze_scores, status
    qids = [str(i) for i in range(96)]
    rows = [dict(qid=q, arm=a, rep=r, score=int(a == "ON" and r == 1))
            for q in qids for a in ARMS for r in range(2)]
    report = analyze_scores(qids, rows, "ON", "NN")
    c1 = report["contrasts"]["C1"]
    assert c1["estimate"] == .5
    assert c1["half_width"] == pytest.approx(.16895, abs=1e-5)
    assert c1["decode_variance"] == pytest.approx(1/(4*96))
    assert report["contrasts"]["I"]["half_width"] == pytest.approx(math.sqrt(2)*c1["half_width"])
    assert report["mechanism_status"] == "GO"
    assert status(.03, .9) == "INCONCLUSIVE"
    assert status(-.9, .03) == "INCONCLUSIVE"
    assert status(.030001, .9) == "GO"
    assert status(-.9, .029999) == "NO_GO"
