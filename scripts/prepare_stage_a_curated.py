#!/usr/bin/env python3
"""Prepare a separate, non-executable package for the curated V3 diagnostic.

No model, network, generation client, mutable ledger or training is constructed.
Historical answer bindings are checked for integrity, then only allowlisted
public inputs and selected BUILD-source gold enter the preparation pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

# Support direct CLI execution and importlib-based acceptance tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_stage_a import (
    ARMS, LIMITS, SAMPLE, OfflineTokenizer, draft_schedule, duplicate_candidates,
    lexical_neighbors, prepare_pair, verify_historical_prompts,
)
from stage_a_curated_checks import (
    dimensional_checks, singularity_checks, verify_example,
)
from memgate.e4_assets.assets import publish_bytes, validate_bundle
from memgate.e4_assets.provenance import MODULE_FILES
from memgate.experiments.common import canonical, digest, file_hash, require, seal, unseal
from memgate.experiments.official import Official

ROOT = Path(__file__).resolve().parents[1]
STATUS = "CURATED_LOCAL_CHECKS_PASSED_NOT_EXECUTION_AUTHORIZED"
SCHEMA = "V3_CURATED_PREPARATION_1"


def public_only(row):
    return dict(qid=row["qid"], question=row["question"], options=list(row["options"]))


def project_sources(bundle, selected):
    """No old responses, ranks, rule bodies, or current gold leave this boundary."""
    table = bundle["e2_table"]
    prefix = [public_only(row) for row in unseal(json.loads(
        bundle["artifacts_utf8"]["public.json"]))[:581]]
    sources = {row["qid"]: row for row in prefix}
    public = [public_only(table["questions"][q]["public"]) for q in sorted(table["questions"])]
    require(len(sources) == 581 and len(public) == 158, "Unexpected source/development counts")
    require(not set(sources) & {row["qid"] for row in public}, "Build/development overlap")
    require(set(selected) <= set(sources), "Example must be a build source, never a target")
    build = {}
    # JSON must be decoded to select records. Discard all nonallowlisted fields,
    # including model_output and memory_meta; never use them to author content.
    raw = unseal(json.loads(bundle["artifacts_utf8"]["prefix_raw.json"]))
    for line in raw.splitlines():
        row = json.loads(line)
        if row["qid"] not in selected:
            continue
        q = row["qid"]
        require(q not in build, "Duplicate selected build source")
        require(public_only(row) == sources[q], "Build-source public input changed")
        build[q] = dict(public=sources[q], subject=row["subject"], gold=row["gold_letter"])
    require(set(build) == set(selected), "Missing selected build-source gold")
    return dict(public=public, prefix_public=prefix, build=build)


def validate_inputs(data, library, review):
    require(library["schema"] == "V3_CURATED_LIBRARY_1"
            and review["schema"] == "V3_CURATED_REVIEW_1", "Unknown curated schema")
    require(digest(data["public"]) == review["public_sha256"], "Reviewed public inputs changed")
    require(digest(data["prefix_public"]) == library["prefix_public_sha256"],
            "Reviewed build inputs changed")
    targets = {row["qid"] for row in data["public"]}
    sources = {row["qid"] for row in data["prefix_public"]}
    require(not targets & sources, "Build/development overlap")
    rules = {r["id"]: r for r in library["rules"]}
    examples = {e["source"]: e for e in library["examples"]}
    quarantine = {e["source"] for e in library["quarantine"]}
    require(len(rules) == len(library["rules"]) and len(examples) == len(library["examples"]),
            "Duplicate content identity")
    require(set(examples) <= sources and set(data["build"]) == set(examples),
            "Example must be a build source, never a target")
    require(quarantine <= sources and not quarantine & set(examples),
            "Quarantined example entered library")
    for rule in rules.values():
        require(rule["text"].strip() and rule["sources"]
                and set(rule["sources"]) <= sources, "Invalid rule source")
    for q, example in examples.items():
        require(example["gold"] == data["build"][q]["gold"], "Build-source gold changed")
        require(len(example["gold"]) == 1
                and 0 <= ord(example["gold"])-65 < len(data["build"][q]["public"]["options"]),
                "Invalid build-source answer index")
        require(example["conditions"].strip() and example["trajectory"].strip(),
                "Missing source conditions or derivation")
    require(set(review["shortlist"]) <= targets, "Unknown reviewed target")
    allowed = {"candidate", "exclude_template", "exclude_coverage", "hold_input"}
    for q, item in review["shortlist"].items():
        require(item["status"] in allowed and item["reason"].strip()
                and set(item["sources"]) <= sources, "Invalid target assessment")
        if item["status"] == "candidate":
            require(item["rule"] in rules and item["example"] in examples,
                    "Candidate refers to unavailable content")
            require(set(rules[item["rule"]]["sources"]) | {item["example"]}
                    <= set(item["sources"]), f"Candidate review omits direct content sources: {q}")
    proposal = review["generation_proposal"]
    pairs = sum(item["status"] == "candidate" for item in review["shortlist"].values())
    require(pairs > 0 and proposal["planned_pairs"] == pairs
            and proposal["repetitions"] == 2
            and proposal["factorial_requests"] == pairs*len(ARMS)*2,
            "Diagnostic request arithmetic changed")
    require(proposal["shared_probe_retry_reserve"] == 8
            and proposal["proposed_total_attempt_cap"] == proposal["factorial_requests"]+8,
            "Shared reserve arithmetic changed")
    require(all(proposal[key] == 0 for key in (
        "authorized_attempts", "curator_calls", "embedding_calls", "training_runs")),
        "Offline preparation cannot authorize execution")


def control_characters(public):
    result = []
    for field, text in [("question", public["question"]),
                        *((f"option_{i}", s) for i, s in enumerate(public["options"]))]:
        for position, char in enumerate(text):
            if ord(char) < 32 and char not in "\n\r\t":
                result.append(dict(field=field, position=position, code=f"U+{ord(char):04X}"))
    return result


def build_content(data, library, official):
    sources = {row["qid"]: row for row in data["prefix_public"]}
    quarantine = {x["source"] for x in library["quarantine"]}

    def lineage(qids):
        return [dict(qid=q, public_sha256=digest(sources[q]),
                     source_answer_quarantined=q in quarantine) for q in sorted(qids)]

    examples = []
    for example in library["examples"]:
        q = example["source"]
        source = data["build"][q]
        check = verify_example(example)
        require(official.extract_letter(example["trajectory"]) == example["gold"],
                "Rewritten trajectory answer does not match build-source gold")
        # Exact original stem and ALL options, followed by a separately versioned
        # derivation. The official renderer's score is a constant, not retrieval.
        rendering = dict(qid=q, question=official.query(source["public"]),
                         trajectory=example["trajectory"], retrieval_score=0.0)
        controls = control_characters(source["public"])
        examples.append(dict(
            id=f"E-{q}-v1", version=library["version"], **example,
            source_record=source, source_sha256=digest(source),
            derived_from=lineage([q]), numerical_review=check,
            source_control_characters=controls,
            input_format_passed=not controls,
            rendering=rendering, content_sha256=digest(rendering)))
    rules = [
        dict(**rule, version=library["version"], derived_from=lineage(rule["sources"]),
             content_sha256=digest(rule["text"]),
             review="Manual conditional-law review; not an automatic proof of natural-language text")
        for rule in library["rules"]]
    return dict(
        rules=rules, examples=examples,
        quarantine=[dict(**row, public=sources[row["source"]],
                         public_sha256=digest(sources[row["source"]]))
                    for row in library["quarantine"]],
        dimensional_checks=dimensional_checks(),
        singularity_rule_checks=singularity_checks(),
        author=library["author"], exposure=library["exposure"], cost=library["cost"])


def build_diagnostic(data, library, review, official, tokenizer):
    validate_inputs(data, library, review)
    content = build_content(data, library, official)
    rules = {r["id"]: r for r in content["rules"]}
    examples = {e["source"]: e for e in content["examples"]}
    sources = {p["qid"]: p for p in data["prefix_public"]}
    inventory, pairs = [], []
    for public in data["public"]:
        q = public["qid"]
        assessment = review["shortlist"].get(q, dict(
            status="not_shortlisted", sources=[],
            reason="全体公开题面初筛后未进入32项便利样本清单；不代表穷尽性无覆盖证明。"))
        neighbors = lexical_neighbors(public, data["prefix_public"])
        duplicates = duplicate_candidates(public, data["prefix_public"])
        entry = dict(public=public, public_sha256=digest(public), assessment=assessment,
                     reviewed_source_public=[sources[s] for s in assessment["sources"]],
                     lexical_neighbors=neighbors, duplicate_candidates=duplicates,
                     control_characters=control_characters(public))
        inventory.append(entry)
        if assessment["status"] != "candidate":
            continue
        require(not duplicates, "Candidate has unreviewed near-verbatim overlap")
        require(not entry["control_characters"], "Candidate has damaged control characters")
        rule, example = rules[assessment["rule"]], examples[assessment["example"]]
        require(example["input_format_passed"], "Selected example has damaged control characters")
        prompts = prepare_pair(official, tokenizer, public, rule["text"], example["rendering"])
        pairs.append(dict(
            qid=q, public=public, family=assessment["family"], rationale=assessment["reason"],
            rule_id=rule["id"], rule_content_sha256=rule["content_sha256"],
            example_id=example["id"], example_content_sha256=example["content_sha256"],
            derived_from=sorted(set(rule["sources"]) | {example["source"]}),
            prompts=prompts, local_review_passed=True, execution_authorized=False))
    return content, inventory, pairs


def assemble(data, library, review, official, tokenizer, provenance, verification):
    content, inventory, pairs = build_diagnostic(data, library, review, official, tokenizer)
    # Include the public evidence, complete review, costs, scope, sources and code,
    # not only rendered bytes. Changing any of them creates a new review identity.
    identity = digest(dict(
        schema=SCHEMA, data_sha256=digest(data), library=library, review=review,
        provenance=provenance, limits=LIMITS, sampling=SAMPLE,
        content_sha256=digest(content),
        prompt_hashes={p["qid"]: {a: x["sha256"] for a, x in p["prompts"].items()} for p in pairs}))
    schedule = draft_schedule(pairs, identity)
    proposal = review["generation_proposal"]
    require(len(schedule) == proposal["factorial_requests"], "Draft request count mismatch")
    tokens = [p["prompt_tokens"] for pair in pairs for p in pair["prompts"].values()]
    reserve = proposal["shared_probe_retry_reserve"]
    budget = dict(
        **proposal, limits=LIMITS, draft_sampling=SAMPLE,
        draft_input_tokens=sum(x["prompt_tokens"] for x in schedule),
        draft_input_token_range=[min(tokens), max(tokens)],
        draft_max_output_tokens=len(schedule)*LIMITS["output"],
        reserve_input_token_upper_bound=reserve*LIMITS["input"],
        reserve_output_token_upper_bound=reserve*LIMITS["output"],
        old_results_automatically_reused=0,
        note="精确输入计数只针对正式草案；预留的探测/重试输入按5888上界。"
             "全部输出是上限。GPU时间/货币成本未实测，助手整理成本未单独计量。")
    counts = dict(
        public_development_questions=len(inventory), build_public_sources=len(data["prefix_public"]),
        shortlisted=len(review["shortlist"]),
        shortlist_status=dict(sorted(Counter(x["status"] for x in review["shortlist"].values()).items())),
        library_rules=len(content["rules"]), library_examples=len(content["examples"]),
        examples_with_format_holds=sum(not e["input_format_passed"] for e in content["examples"]),
        quarantined_sources=len(content["quarantine"]),
        checked_numerical_claims=sum(len(e["numerical_review"]["claim_checks"]) for e in content["examples"]),
        numerical_cross_checks=sum(len(e["numerical_review"]["cross_checks"]) for e in content["examples"]),
        dimensional_checks=len(content["dimensional_checks"]),
        draft_pairs=len(pairs), descriptive_families=len({p["family"] for p in pairs}),
        used_rules=len({p["rule_id"] for p in pairs}), used_examples=len({p["example_id"] for p in pairs}),
        draft_prompts=len(tokens), draft_requests=len(schedule), executable_requests=0,
        new_solver_calls=0, new_curator_calls=0, embedding_calls=0, training_runs=0)
    return dict(
        schema=SCHEMA, status=STATUS, identity=identity, provenance=provenance,
        asset_verification=verification, counts=counts, library_config=library, review=review,
        content=content, inventory=inventory, build_public_sources=data["prefix_public"],
        pairs=pairs, draft_schedule=schedule, executable_schedule=[], budget=budget,
        boundaries=[
            "No executable schedule, transport, new ledger, generation or training.",
            "Fixed hand-selected pairs only; not a test of learned retrieval, evolution or eviction.",
            "Same assistant's convenience sample and review; not independent expert validation.",
            "All 158 questions are development material; exposure prevents blind-confirmation claims.",
            "Family names are descriptive strata, not statistically independent histories.",
            "Numerical checks do not automatically certify prose, option uniqueness or applicability.",
            "Source text and options preserved verbatim; rewritten derivations have a separate version.",
            "All later reading baselines must share the same library; old/new content is a separate factor.",
            "New protocol, scoring-only gold binding, server preflight and budget authorization still needed.",
        ])


def prepare(bundle_path, library_path, review_path, tokenizer_dir, official_dir):
    library = json.loads(Path(library_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    with gzip.open(bundle_path, "rt", encoding="utf-8") as stream:
        bundle = unseal(json.load(stream))
    require(digest(bundle) == library["bundle_sha256"], "Reviewed asset bundle changed")
    verification = validate_bundle(bundle)
    print("Historical asset bindings verified; only allowlisted sources enter curation.", flush=True)
    fingerprint = bundle["e2_table"]["manifest"]["fingerprint"]
    for name, expected in fingerprint["official"]["files"].items():
        require(file_hash(Path(official_dir)/name) == expected, "Official source changed")
    official = Official(official_dir)
    tokenizer = OfflineTokenizer(tokenizer_dir, fingerprint["model"])
    verification["historical_prompts_retokenized"] = verify_historical_prompts(
        tokenizer, bundle["e2_table"])
    data = project_sources(bundle, [e["source"] for e in library["examples"]])
    code = {str(Path("src/memgate")/name): file_hash(ROOT/"src/memgate"/name)
            for name in MODULE_FILES}
    for name in ("scripts/prepare_stage_a.py", "scripts/prepare_stage_a_curated.py",
                 "scripts/stage_a_curated_checks.py", "src/memgate/experiments/official.py",
                 "src/memgate/experiments/prompts.py"):
        code[name] = file_hash(ROOT/name)
    provenance = dict(
        code=code, library_sha256=digest(library), review_sha256=digest(review),
        projected_data_sha256=digest(data), python=platform.python_version(),
        dependencies={n: importlib.metadata.version(n) for n in ("tokenizers", "jinja2", "numpy")},
        tokenizer_files=tokenizer.files, model_files_sha256=digest(fingerprint["model"]),
        official=fingerprint["official"], chat_thinking=False,
        selection="fixed manual pair; retrieval_score=0.0 is a rendering constant, not similarity")
    return assemble(data, library, review, official, tokenizer, provenance, verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT/".cache/e4-assets-v1/assets.json.gz")
    parser.add_argument("--library", type=Path, default=ROOT/"configs/stage_a_curated_library.json")
    parser.add_argument("--review", type=Path, default=ROOT/"configs/stage_a_curated_review.json")
    parser.add_argument("--tokenizer", type=Path, default=ROOT/".cache/stage-a-tokenizer")
    parser.add_argument("--official", type=Path, default=ROOT/"third_party/SeqMem-Eval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    require(output.parent.is_dir(), "Create an independent output directory first")
    for protected in (args.assets, args.library, args.review):
        require(output.resolve() != protected.resolve(), "Cannot overwrite preparation inputs")
    for protected in (ROOT/"runs", ROOT/"third_party", args.tokenizer):
        require(protected.resolve() not in output.resolve().parents, "Protected output directory")
    result = prepare(args.assets, args.library, args.review, args.tokenizer, args.official)
    publish_bytes(output, gzip.compress(canonical(seal(result)), mtime=0))
    print(json.dumps(dict(status=result["status"], identity=result["identity"],
                         counts=result["counts"], budget=result["budget"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
