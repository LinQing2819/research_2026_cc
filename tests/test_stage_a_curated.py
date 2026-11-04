"""Acceptance tests for the new, non-executable curated preparation boundary."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from memgate.experiments.common import Invalid, canonical, digest, seal, unseal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "curated", ROOT / "scripts/prepare_stage_a_curated.py")
curated = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curated)


@pytest.fixture(scope="module")
def evidence():
    path = ROOT / ".cache/e4-assets-v1/assets.json.gz"
    if not path.exists():
        pytest.skip("Local sealed research assets unavailable")
    with gzip.open(path, "rt") as stream:
        bundle = unseal(json.load(stream))
    library = json.loads((ROOT / "configs/stage_a_curated_library.json").read_text())
    review = json.loads((ROOT / "configs/stage_a_curated_review.json").read_text())
    data = curated.project_sources(bundle, [e["source"] for e in library["examples"]])
    return bundle, library, review, data


@pytest.fixture(scope="module")
def rendering(evidence):
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    path = ROOT / ".cache/stage-a-tokenizer"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Local pinned tokenizer unavailable")
    tokenizer = curated.OfflineTokenizer(
        path, evidence[0]["e2_table"]["manifest"]["fingerprint"]["model"])
    return curated.Official(ROOT / "third_party/SeqMem-Eval"), tokenizer


def test_projection_ignores_current_answers_and_old_model_writing(evidence):
    bundle, library, _, expected = evidence
    changed = dict(bundle, development="FORBIDDEN_CURRENT")
    changed["e2_table"] = dict(bundle["e2_table"], rows="FORBIDDEN_CURRENT",
                               snapshots="FORBIDDEN_OLD_MEMORY")
    changed["e2_table"]["questions"] = {
        q: dict(public=dict(row["public"], gold="FORBIDDEN_CURRENT"),
                ranked="FORBIDDEN_OLD_MEMORY", prompts="FORBIDDEN_CURRENT",
                gold="FORBIDDEN_CURRENT")
        for q, row in bundle["e2_table"]["questions"].items()}
    changed["artifacts_utf8"] = dict(bundle["artifacts_utf8"],
                                    **{"scoring_gold.json": "FORBIDDEN_CURRENT"})
    prefix = []
    for line in unseal(json.loads(bundle["artifacts_utf8"]["prefix_raw.json"])).splitlines():
        row = json.loads(line)
        row["model_output"] = row["memory_meta"] = "FORBIDDEN_OLD_WRITING"
        prefix.append(json.dumps(row))
    changed["artifacts_utf8"]["prefix_raw.json"] = json.dumps(seal("\n".join(prefix)))
    actual = curated.project_sources(changed, [e["source"] for e in library["examples"]])
    assert actual == expected
    assert "FORBIDDEN_" not in json.dumps(actual)
    assert set(actual) == {"public", "prefix_public", "build"}
    assert all(set(x) == {"public", "gold", "subject"} for x in actual["build"].values())


def test_projection_rejects_a_target_as_a_source(evidence):
    with pytest.raises(Invalid, match="build source, never a target"):
        curated.project_sources(evidence[0], ["11870"])


@pytest.mark.parametrize("mutation,match", [
    ("public", "Reviewed public inputs"),
    ("prefix", "Reviewed build inputs"),
    ("gold", "Build-source gold"),
    ("rule_source", "Invalid rule source"),
    ("candidate_lineage", "omits direct content sources"),
    ("quarantine", "Quarantined example"),
    ("authorization", "cannot authorize"),
    ("reserve", "reserve arithmetic"),
    ("request_count", "request arithmetic"),
])
def test_changes_to_reviewed_boundaries_fail_closed(evidence, mutation, match):
    _, original_library, original_review, original_data = evidence
    library, review, data = map(copy.deepcopy, (original_library, original_review, original_data))
    if mutation == "public":
        data["public"][0]["options"][0] += " altered"
    elif mutation == "prefix":
        data["prefix_public"][0]["question"] += " altered"
    elif mutation == "gold":
        library["examples"][0]["gold"] = "A"
    elif mutation == "rule_source":
        library["rules"][0]["sources"] = ["11870"]
    elif mutation == "candidate_lineage":
        review["shortlist"]["11931"]["sources"].remove("11356")
    elif mutation == "quarantine":
        library["quarantine"].append(dict(source=library["examples"][0]["source"], reason="test"))
    elif mutation == "authorization":
        review["generation_proposal"]["authorized_attempts"] = 80
    elif mutation == "reserve":
        review["generation_proposal"]["proposed_total_attempt_cap"] += 1
    else:
        review["generation_proposal"]["factorial_requests"] += 1
    with pytest.raises(Invalid, match=match):
        curated.validate_inputs(data, library, review)


def test_numerical_error_or_mismatched_calculation_prevents_content_build(evidence, rendering):
    _, original, _, data = evidence
    official, _ = rendering
    library = copy.deepcopy(original)
    library["examples"][0]["claims"]["power_W"] *= 10
    with pytest.raises(Invalid, match="Numerical check failed"):
        curated.build_content(data, library, official)
    library = copy.deepcopy(original)
    library["examples"][0]["check"] = "wave_amplitude"
    with pytest.raises(Invalid, match="does not match reviewed build source"):
        curated.build_content(data, library, official)
    library = copy.deepcopy(original)
    library["examples"][0]["trajectory"] = "The answer is (A)."
    with pytest.raises(Invalid, match="trajectory answer"):
        curated.build_content(data, library, official)


def test_library_keeps_source_text_lineage_and_conditional_math(evidence, rendering):
    _, library, _, data = evidence
    official, _ = rendering
    content = curated.build_content(data, library, official)
    prefix = {x["qid"]: x for x in data["prefix_public"]}
    assert len(content["examples"]) == 14
    assert len(content["rules"]) == 11
    assert len(content["dimensional_checks"]) == 9
    assert content["singularity_rule_checks"]
    assert "11778" not in {e["source"] for e in content["examples"]}
    assert len(content["quarantine"]) == 6
    for e in content["examples"]:
        assert e["source_record"]["public"] == prefix[e["source"]]
        assert e["rendering"]["question"] == official.query(prefix[e["source"]])
        assert e["rendering"]["retrieval_score"] == 0.0
        assert e["numerical_review"]["claim_checks"] and e["numerical_review"]["cross_checks"]
        assert e["input_format_passed"] is (not e["source_control_characters"])
    assert [e["source"] for e in content["examples"] if not e["input_format_passed"]] == ["11464"]
    for r in content["rules"]:
        assert {p["qid"] for p in r["derived_from"]} == set(r["sources"])
        assert all(p["public_sha256"] == digest(prefix[p["qid"]]) for p in r["derived_from"])
    divergence = next(r for r in content["rules"] if r["id"] == "R-divergence-v1")
    assert divergence["derived_from"][0]["source_answer_quarantined"] is True


def test_complete_draft_preserves_factors_and_cannot_execute(evidence, rendering):
    _, library, review, data = evidence
    official, tokenizer = rendering
    result = curated.assemble(data, library, review, official, tokenizer, {}, {})
    assert len(result["inventory"]) == 158 and len(result["build_public_sources"]) == 581
    assert len(result["pairs"]) == 9
    assert result["counts"]["draft_prompts"] == 36
    assert result["executable_schedule"] == []
    budget = result["budget"]
    assert budget["authorized_attempts"] == 0
    assert budget["factorial_requests"] == 72
    assert budget["factorial_requests"] + budget["shared_probe_retry_reserve"] == 80
    assert budget["proposed_total_attempt_cap"] == 80
    assert all(p["execution_authorized"] is False for p in result["pairs"])
    assert "12082" not in {p["qid"] for p in result["pairs"]}
    for pair in result["pairs"]:
        prompts = pair["prompts"]
        assert len({p["sha256"] for p in prompts.values()}) == 4
        assert prompts["10"]["rule_span"]["ids"] == prompts["11"]["rule_span"]["ids"]
        assert prompts["01"]["case_span"]["ids"] == prompts["11"]["case_span"]["ids"]
        for p in prompts.values():
            assert p["prompt"].endswith(official.problem(pair["public"]) + "\n")
            assert tokenizer.encode(p["text"]) == p["ids"]
    schedule = result["draft_schedule"]
    assert len(schedule) == len({x["logical_sha256"] for x in schedule}) == 72
    assert len({x["request_sha256"] for x in schedule}) == 72
    assert all(x["executable"] is False for x in schedule)
    assert schedule == curated.draft_schedule(list(reversed(result["pairs"])), result["identity"])
    assert budget["draft_input_tokens"] == sum(x["prompt_tokens"] for x in schedule)
    assert budget["draft_max_output_tokens"] == 72 * 2048
    # Sealing is lossless; full evidence and every token remain locally replayable.
    assert unseal(json.loads(gzip.decompress(gzip.compress(canonical(seal(result)), mtime=0)))) == result
    # Regenerating from serialized input must preserve the entire identity.
    # JSON key order must not change the order of per-claim audit records.
    serialized_data, serialized_library, serialized_review = json.loads(
        canonical([data, library, review]))
    repeated = curated.assemble(
        serialized_data, serialized_library, serialized_review, official, tokenizer, {}, {})
    assert canonical(repeated) == canonical(result)


def test_damaged_example_cannot_be_selected_even_when_numerics_pass(evidence, rendering):
    _, library, original_review, data = evidence
    official, tokenizer = rendering
    review = copy.deepcopy(original_review)
    review["shortlist"]["12076"]["example"] = "11464"
    with pytest.raises(Invalid, match="Selected example has damaged control characters"):
        curated.build_diagnostic(data, library, review, official, tokenizer)
