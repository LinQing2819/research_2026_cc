"""Local acceptance checks for the offline V3 asset-preparation boundary.

Real sealed assets and tokenizer files are optional local research inputs, not
redistributed test fixtures. These checks skip if those inputs are unavailable.
"""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from memgate.experiments.common import Invalid, digest, unseal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage_a", ROOT / "scripts/prepare_stage_a.py")
stage_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_a)


@pytest.fixture(scope="module")
def evidence():
    path = ROOT / ".cache/e4-assets-v1/assets.json.gz"
    if not path.exists():
        pytest.skip("Local sealed research assets unavailable")
    with gzip.open(path, "rt") as stream:
        bundle = unseal(json.load(stream))
    review = json.loads((ROOT / "configs/stage_a_review.json").read_text())
    return bundle, review, stage_a.public_projection(bundle)


@pytest.fixture(scope="module")
def rendering(evidence):
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    path = ROOT / ".cache/stage-a-tokenizer"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Local pinned tokenizer unavailable")
    bundle, _, _ = evidence
    tokenizer = stage_a.OfflineTokenizer(
        path, bundle["e2_table"]["manifest"]["fingerprint"]["model"])
    return stage_a.Official(ROOT / "third_party/SeqMem-Eval"), tokenizer


def test_current_gold_and_outcomes_cannot_change_public_projection(evidence):
    bundle, _, original = evidence
    changed = dict(bundle, development={"answers": "FORBIDDEN_CURRENT_OUTCOME"})
    changed["artifacts_utf8"] = dict(bundle["artifacts_utf8"],
                                    **{"scoring_gold.json": "FORBIDDEN_CURRENT_GOLD"})
    changed["e2_table"] = dict(bundle["e2_table"], rows="FORBIDDEN_CURRENT_ANSWERS")
    projected = stage_a.public_projection(changed)
    assert projected == original
    assert "FORBIDDEN_CURRENT" not in json.dumps(projected)


@pytest.mark.parametrize("mutation,match", [
    ("public", "Reviewed public inputs changed"),
    ("rules", "Reviewed rules changed"),
    ("rank", "Frozen top-one example changed"),
    ("quote", "Example evidence quote changed"),
])
def test_review_rejects_changed_inputs(evidence, mutation, match):
    _, review, original = evidence
    data = copy.deepcopy(original)
    if mutation == "public":
        data["public"][0]["question"] += "changed"
    elif mutation == "rules":
        data["rules"] += "changed"
    elif mutation == "rank":
        data["ranked"]["11899"].reverse()
    else:
        next(x for x in data["pool"] if x["qid"] == "11583")["trajectory"] = "changed"
        data["ranked"]["11899"][0]["trajectory"] = "changed"
    with pytest.raises(Invalid, match=match):
        stage_a.validate_review(data, review)


def test_four_prompts_preserve_factors_and_current_question(evidence, rendering):
    _, review, data = evidence
    official, tokenizer = rendering
    public = next(x for x in data["public"] if x["qid"] == "11899")
    rule = stage_a.rule_items(data["rules"])[8]
    example = data["ranked"]["11899"][0]
    prompts = stage_a.prepare_pair(official, tokenizer, public, rule, example)
    assert len({p["sha256"] for p in prompts.values()}) == 4
    for p in prompts.values():
        assert p["prompt"].endswith(official.problem(public) + "\n")
        assert tokenizer.encode(p["text"]) == p["ids"]
        for name in ("rule_span", "case_span"):
            left, right = p[name]["positions"]
            assert p["ids"][left:right] == p[name]["ids"]
    assert prompts["10"]["rule_span"]["ids"] == prompts["11"]["rule_span"]["ids"]
    assert prompts["01"]["case_span"]["ids"] == prompts["11"]["case_span"]["ids"]
    assert example["trajectory"] in prompts["11"]["prompt"]
    assert example["trajectory"] not in prompts["10"]["prompt"]
    assert rule.rstrip("\n") in prompts["10"]["prompt"]
    assert rule.rstrip("\n") not in prompts["01"]["prompt"]


@pytest.mark.parametrize("violation,match", [
    ("gold", "Only public question fields"),
    ("self_source", "Self-source example"),
    ("long_example", "Example block exceeds capacity"),
    ("long_rule", "Rule block exceeds capacity"),
])
def test_prompt_boundary_rejects_leakage_and_truncation(evidence, rendering, violation, match):
    _, _, data = evidence
    official, tokenizer = rendering
    public = dict(next(x for x in data["public"] if x["qid"] == "11899"))
    example = dict(data["ranked"]["11899"][0])
    rule = stage_a.rule_items(data["rules"])[8]
    if violation == "gold":
        public["gold"] = "FORBIDDEN_CURRENT_GOLD"
    elif violation == "self_source":
        example["qid"] = public["qid"]
    elif violation == "long_example":
        example["trajectory"] = " long" * 5000
    else:
        rule = " long" * 2000
    with pytest.raises(Invalid, match=match):
        stage_a.prepare_pair(official, tokenizer, public, rule, example)


def test_sealed_preparation_is_deterministic_and_non_executable(evidence, rendering):
    _, review, data = evidence
    official, tokenizer = rendering
    inventory, pairs = stage_a.build_review(data, review, official, tokenizer)
    identity = digest(review)
    schedule = stage_a.draft_schedule(pairs, identity)
    assert schedule == stage_a.draft_schedule(list(reversed(pairs)), identity)
    assert len(inventory) == 158
    assert len(pairs) == 8 and all(p["qualified"] is False for p in pairs)
    assert len(schedule) == len({x["logical_sha256"] for x in schedule}) == 64
    assert len({x["request_sha256"] for x in schedule}) == 64
    assert all(x["executable"] is False for x in schedule)
    assert any("exclude_template_overlap" in p["blockers"] for p in pairs)
    assert any("visible_example_reasoning_error" in p["blockers"] for p in pairs)
    assert all("full_semantic_source_qualification_pending" in p["blockers"] for p in pairs)


def test_tokenizer_identity_is_enforced(evidence, rendering):
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    bundle, _, _ = evidence
    expected = dict(bundle["e2_table"]["manifest"]["fingerprint"]["model"])
    expected["tokenizer.json"] = "0" * 64
    with pytest.raises(Invalid, match="Tokenizer files differ"):
        stage_a.OfflineTokenizer(ROOT / ".cache/stage-a-tokenizer", expected)
