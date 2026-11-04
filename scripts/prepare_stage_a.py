#!/usr/bin/env python3
"""Build an offline, non-executable V3 diagnostic review package.

No model weights, model inference, ledger construction, network requests or
training. Historical outcomes are inspected by the existing integrity validator;
candidate construction receives only public inputs and frozen memory content.
"""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import gzip
import json
import math
from pathlib import Path
import re

from memgate.e4_assets.assets import publish_bytes, validate_bundle
from memgate.experiments.common import (
    canonical, digest, file_hash, require, seal, unseal,
)
from memgate.experiments.official import Official
from memgate.experiments.prompts import (
    CASE_HEADER, RULE_HEADER, chat, token_span,
)
from memgate.experiments.source import normalized, rule_items

ROOT = Path(__file__).resolve().parents[1]
STATUS = "PREPARED_NOT_QUALIFIED_NOT_EXECUTION_AUTHORIZED"
ARMS = {"00": (False, False), "10": (True, False),
        "01": (False, True), "11": (True, True)}
# New proposal constants, deliberately not the historical E2 attempt budget.
LIMITS = dict(rules=1024, experiences=3072, input=5888, output=2048,
              context=8192, safety=256)
SAMPLE = dict(
    model="Qwen/Qwen3-8B", max_tokens=2048, temperature=0.7, top_p=1.0,
    top_k=-1, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
    frequency_penalty=0.0, n=1, best_of=1, stream=False, echo=False,
    stop=None, stop_token_ids=[], ignore_eos=False, skip_special_tokens=True,
    spaces_between_special_tokens=True, min_tokens=0,
    include_stop_str_in_output=False, add_special_tokens=False,
    use_beam_search=False,
)


class OfflineTokenizer:
    """Minimal adapter for the two exact, hash-pinned Qwen tokenizer files."""

    def __init__(self, directory, expected):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.files = {name: file_hash(directory / name)
                      for name in ("tokenizer.json", "tokenizer_config.json")}
        require(all(expected[k] == v for k, v in self.files.items()),
                "Tokenizer files differ from historical model identity")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.config = json.loads((directory / "tokenizer_config.json").read_text())
        self.template = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True).from_string(self.config["chat_template"])

    def encode(self, text, add_special_tokens=False):
        require(add_special_tokens is False, "Unexpected special-token policy")
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        require(add_special_tokens is False, "Unexpected special-token policy")
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return dict(input_ids=encoded.ids, offset_mapping=encoded.offsets)

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=True, enable_thinking=False):
        require(add_generation_prompt is True and enable_thinking is False,
                "Unexpected chat generation policy")
        text = self.template.render(
            messages=messages, add_generation_prompt=True, enable_thinking=False,
            bos_token=self.config.get("bos_token", ""),
            eos_token=self.config.get("eos_token", ""))
        return self.encode(text) if tokenize else text


def public_projection(bundle):
    """Explicit allowlist: exclude scoring gold and all current-question answers."""
    table = bundle["e2_table"]
    public = unseal(json.loads(bundle["artifacts_utf8"]["public.json"]))
    return dict(
        public=[table["questions"][q]["public"] for q in sorted(table["questions"])],
        ranked={q: table["questions"][q]["ranked"]["new"] for q in sorted(table["questions"])},
        rules=table["snapshots"]["new"]["rules"], pool=table["snapshots"]["new"]["pool"],
        prefix_public=public[:581],
        prefix_sha256=table["snapshots"]["new"]["prefix_sha256"],
    )


def validate_review(data, review):
    require(review["schema"] == "V3_STAGE_A_REVIEW_1", "Unknown review schema")
    require(digest(data["public"]) == review["public_sha256"], "Reviewed public inputs changed")
    require(digest(data["rules"]) == review["rules_sha256"], "Reviewed rules changed")
    qids = {q["qid"] for q in data["public"]}
    direct, adjacent = review["direct"], review["adjacent"]
    require(set(direct) <= qids and set(adjacent) <= qids
            and not set(direct) & set(adjacent), "Unknown/overlapping reviewed questions")
    require(set(review["family_review"]) == set(direct), "Missing family review")
    prefix_ids = {q["qid"] for q in data["prefix_public"]}
    rules = rule_items(data["rules"])
    pool = {x["qid"]: x for x in data["pool"]}
    for q, item in {**direct, **adjacent}.items():
        require(type(item["rule"]) is int and 1 <= item["rule"] <= len(rules),
                "Invalid rule reference")
        require(bool(item["reason"].strip()), "Missing applicability rationale")
        if q in direct:
            ranked = data["ranked"][q]
            require(ranked and ranked[0]["qid"] == item["example"],
                    "Frozen top-one example changed")
            require({k: ranked[0][k] for k in ("qid", "question", "trajectory")}
                    == pool[item["example"]], "Example source changed")
            require(q not in prefix_ids and q != item["example"],
                    "Current question is a memory source")
            require(set(review["family_review"][q]["sources"]) <= prefix_ids,
                    "Family review references unknown sources")
    require(set(review["example_review"]) == {x["example"] for x in direct.values()},
            "Missing/foreign example quality review")
    for q, item in review["example_review"].items():
        require(item["quote"] in pool[q]["trajectory"], "Example evidence quote changed")


def lexical_neighbors(public, prefix, count=5):
    """Public-only inspection aid; similarity is never a qualification decision."""
    stop = set("the and for from with that this are calculate determine find has its per "
               "can which when using into".split())
    words = lambda s: set(re.findall(r"[a-z]{3,}", s.lower())) - stop
    tokens = {p["qid"]: words(p["question"]) for p in prefix}
    df = Counter(w for ts in tokens.values() for w in ts)
    weights = {w: math.log((len(prefix) + 1) / (n + 1)) for w, n in df.items()}
    a = words(public["question"])
    scores = []
    for p in prefix:
        b = tokens[p["qid"]]
        score = (sum(weights.get(w, 0) for w in sorted(a & b))
                 / max(1, sum(weights.get(w, 0) for w in sorted(a | b))))
        scores.append(dict(public=p, public_sha256=digest(p), score=score))
    return sorted(scores, key=lambda x: (-x["score"], x["public"]["qid"]))[:count]


def duplicate_candidates(public, prefix, threshold=0.92):
    result = []
    left = normalized(public["question"])
    for p in prefix:
        right = normalized(p["question"])
        exact = left == right
        matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
        if exact or matcher.quick_ratio() >= threshold:
            ratio = matcher.ratio()
            if exact or ratio >= threshold:
                result.append(dict(source=p["qid"], question_similarity=ratio,
                                   exact_stem=exact, status="requires_review"))
    return result


def prepare_pair(official, tokenizer, public, rule, example):
    require(set(public) == {"qid", "question", "options"},
            "Only public question fields may enter prompt construction")
    require(set(example) == {"qid", "question", "trajectory", "retrieval_score"},
            "Unexpected example fields")
    require(public["qid"] != example["qid"], "Self-source example")
    # Original numbering and wording remain unchanged; no renumbering/truncation.
    rule = rule.rstrip("\n")
    problem = official.problem(public)
    prompts = {}
    for arm, (has_rule, has_case) in ARMS.items():
        rules = rule if has_rule else "(empty)"
        items = [example] if has_case else []
        cases = official.cases(items)
        require(len(tokenizer.encode(RULE_HEADER + rules)) <= LIMITS["rules"],
                "Rule block exceeds capacity; truncation forbidden")
        require(len(tokenizer.encode(CASE_HEADER + cases)) <= LIMITS["experiences"],
                "Example block exceeds capacity; truncation forbidden")
        prompt = official.solver(problem, rules, items)
        require(prompt.endswith(problem + "\n"), "Current question was altered")
        packed = chat(tokenizer, prompt)
        require(len(packed["ids"]) <= LIMITS["input"], "Full input exceeds capacity")
        require(len(packed["ids"]) + LIMITS["output"] + LIMITS["safety"] <= LIMITS["context"],
                "Context allocation exceeded")
        offset = packed["text"].index(prompt)
        start = offset + prompt.index(RULE_HEADER)
        end = start + len(RULE_HEADER + rules)
        case_start = end + 2
        case_end = case_start + len(CASE_HEADER + cases)
        require(packed["text"][case_start:case_end] == CASE_HEADER + cases,
                "Official scaffold changed")
        prompts[arm] = dict(
            prompt=prompt, **packed, prompt_tokens=len(packed["ids"]),
            sha256=digest(packed["ids"]), rule_present=has_rule, example_present=has_case,
            rule_span=token_span(tokenizer, packed["text"], start, end),
            case_span=token_span(tokenizer, packed["text"], case_start, case_end),
        )
    require(len({p["sha256"] for p in prompts.values()}) == 4,
            "Pair does not yield four distinct interventions")
    for flag, span in (("rule_present", "rule_span"), ("example_present", "case_span")):
        for present in (False, True):
            values = [p[span]["ids"] for p in prompts.values() if p[flag] == present]
            require(all(v == values[0] for v in values), "Factor token invariance failed")
    return prompts


def verify_historical_prompts(tokenizer, table):
    checked = 0
    for q in table["questions"].values():
        for p in q["prompts"].values():
            actual = chat(tokenizer, p["prompt"])
            require(all(actual[k] == p[k] for k in ("messages", "text", "ids")),
                    "Historical chat retokenization mismatch")
            checked += 1
    return checked


def build_review(data, review, official, tokenizer):
    validate_review(data, review)
    rules = rule_items(data["rules"])
    source = {p["qid"]: p for p in data["prefix_public"]}
    inventory, pairs = [], []
    for public in data["public"]:
        q = public["qid"]
        item = review["direct"].get(q) or review["adjacent"].get(q)
        assessment = ("direct_step_candidate" if q in review["direct"]
                      else "adjacent_not_direct" if item else "no_direct_rule_identified")
        inventory.append(dict(public=public, public_sha256=digest(public), assessment=assessment,
                              review=item or dict(reason=review["default_assessment"])))
        if q not in review["direct"]:
            continue
        example = data["ranked"][q][0]
        prompts = prepare_pair(official, tokenizer, public, rules[item["rule"] - 1], example)
        family = review["family_review"][q]
        quality = review["example_review"][example["qid"]]
        duplicates = duplicate_candidates(public, data["prefix_public"])
        blockers = ["full_semantic_source_qualification_pending",
                    "new_protocol_and_generation_budget_not_authorized"]
        if family["status"] != "screened_neighbors_only":
            blockers.append(family["status"])
        if quality["status"] == "visible_reasoning_error":
            blockers.append("visible_example_reasoning_error")
        if duplicates:
            blockers.append("lexical_duplicate_candidate")
        pairs.append(dict(
            qid=q, public=public, family=item["family"], rationale=item["reason"],
            rule_id=f'R{item["rule"]:02d}', rule_text=rules[item["rule"] - 1].rstrip("\n"),
            rule_content_sha256=digest(rules[item["rule"] - 1].rstrip("\n")),
            rule_provenance=dict(exact_derived_from=None,
                                 conservative_source="entire_581_question_prefix",
                                 prefix_sha256=data["prefix_sha256"]),
            example=example, example_content_sha256=digest(example),
            example_quality=quality, family_review=family,
            reviewed_source_public=[source[s] for s in family["sources"]],
            lexical_neighbors=lexical_neighbors(public, data["prefix_public"]),
            lexical_duplicate_candidates=duplicates,
            prompts=prompts, qualified=False, blockers=blockers,
        ))
    return inventory, pairs


def draft_schedule(pairs, identity):
    result = []
    for pair in pairs:
        for arm, p in pair["prompts"].items():
            for rep in range(2):
                key = [identity, pair["qid"], arm, rep]
                seed = int(digest(["stage-a-draft-decode", *key])[:8], 16) % (2**31)
                request = {**SAMPLE, "prompt": p["ids"], "seed": seed}
                result.append(dict(qid=pair["qid"], arm=arm, rep=rep, seed=seed,
                                   logical_sha256=digest(key),
                                   request_sha256=digest(request),
                                   prompt_sha256=p["sha256"], prompt_tokens=p["prompt_tokens"],
                                   executable=False))
    return sorted(result, key=lambda x: digest(["stage-a-draft-order", identity,
                                               x["qid"], x["arm"], x["rep"]]))


def prepare(bundle_path, review_path, tokenizer_dir, official_dir):
    with gzip.open(bundle_path, "rt", encoding="utf-8") as stream:
        bundle = unseal(json.load(stream))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(digest(bundle) == review["bundle_sha256"], "Reviewed asset bundle changed")
    verification = validate_bundle(bundle)
    print("Historical asset bindings verified.", flush=True)
    table = bundle["e2_table"]
    fingerprint = table["manifest"]["fingerprint"]
    for name, expected in fingerprint["official"]["files"].items():
        require(file_hash(Path(official_dir) / name) == expected, "Official source changed")
    official = Official(official_dir)
    tokenizer = OfflineTokenizer(tokenizer_dir, fingerprint["model"])
    checked = verify_historical_prompts(tokenizer, table)
    print(f"Exact historical chat/token checks: {checked}.", flush=True)
    data = public_projection(bundle)
    inventory, pairs = build_review(data, review, official, tokenizer)
    code_paths = [
        Path(__file__), ROOT / "src/memgate/experiments/official.py",
        ROOT / "src/memgate/experiments/prompts.py",
        ROOT / "src/memgate/experiments/source.py",
        ROOT / "src/memgate/experiments/common.py",
        ROOT / "src/memgate/e4_assets/assets.py",
    ]
    import importlib.metadata
    provenance = dict(
        review_sha256=digest(review), tokenizer_files=tokenizer.files,
        dependencies={name: importlib.metadata.version(name)
                      for name in ("tokenizers", "jinja2")},
        code={str(p.relative_to(ROOT)): file_hash(p) for p in code_paths},
        model_files_sha256=digest(fingerprint["model"]),
        official=fingerprint["official"], chat_thinking=False,
    )
    identity = digest(dict(provenance=provenance, review=review,
                           prompts={x["qid"]: {a: p["sha256"] for a, p in x["prompts"].items()}
                                    for x in pairs}, limits=LIMITS, sampling=SAMPLE))
    schedule = draft_schedule(pairs, identity)
    prefix = [json.loads(s) for s in unseal(json.loads(
        bundle["artifacts_utf8"]["prefix_raw.json"])).splitlines()]
    all_rule_bodies = {re.sub(r"^\d+\. ", "", line).strip()
                      for r in prefix for line in rule_items(r["memory_meta"]["insights_after"])}
    updates = sum(r["memory_meta"]["insights_before"] != r["memory_meta"]["insights_after"]
                  for r in prefix)
    tokens = [p["prompt_tokens"] for pair in pairs for p in pair["prompts"].values()]
    counts = dict(
        reviewed_questions=len(inventory), final_rules=len(rule_items(data["rules"])),
        successful_option_examples=len(data["pool"]),
        historical_rows=verification["counts"]["e2_rows"],
        historical_prompts_retokenized=checked,
        rule_batches=len(data["pool"]) // 8, changed_rule_batches=updates,
        distinct_historical_rule_bodies=len(all_rule_bodies),
        applicability=dict(sorted(Counter(x["assessment"] for x in inventory).items())),
        draft_pairs=len(pairs), draft_families=len({x["family"] for x in pairs}),
        draft_rules=len({x["rule_id"] for x in pairs}),
        draft_unique_examples=len(review["example_review"]),
        visibly_flawed_unique_examples=sum(
            x["status"] == "visible_reasoning_error" for x in review["example_review"].values()),
        pairs_with_flawed_examples=sum(
            x["example_quality"]["status"] == "visible_reasoning_error" for x in pairs),
        template_overlap_exclusions=sum(
            x["family_review"]["status"] == "exclude_template_overlap" for x in pairs),
        inverse_family_unresolved=sum(
            x["family_review"]["status"] == "unresolved_inverse_problem" for x in pairs),
        draft_prompts=len(tokens), draft_repeated_requests=len(schedule),
        qualified_pairs=0, executable_requests=0, new_model_calls=0, training_runs=0,
    )
    return dict(
        schema="V3_STAGE_A_PREPARATION_1", status=STATUS, identity=identity,
        counts=counts, provenance=provenance, asset_verification=verification,
        review=review, inventory=inventory,
        rules=[dict(rule_id=f"R{i:02d}", text=r.rstrip("\n"),
                    content_sha256=digest(r.rstrip("\n")),
                    concern=review["rule_concerns"].get(str(i)),
                    exact_derived_from=None, conservative_source_count=581)
               for i, r in enumerate(rule_items(data["rules"]), 1)],
        conservative_rule_source_public=data["prefix_public"],
        pairs=pairs, draft_schedule=schedule, executable_schedule=[],
        budget=dict(
            authorized_new_attempts=0, limits=LIMITS, draft_sampling=SAMPLE,
            draft_input_tokens=sum(x["prompt_tokens"] for x in schedule),
            draft_input_token_range=[min(tokens), max(tokens)],
            draft_max_output_tokens=len(schedule) * LIMITS["output"],
            original_320_planning_only=dict(pair_solver=256, curator=16, local_solver=32,
                                           probes_and_retries=16, total=320),
            old_results_automatically_reused=0,
            note="64个草案身份仅供审查，不能提交；旧ZZ虽同prompt，也不自动充当本轮新重复。"
                 "输入tokens精确、输出仅上界；GPU时间/货币成本尚无新实测。"),
        stop_conditions=[
            "Current package is non-executable; do not submit its draft schedule.",
            "Do not pad to 32 questions with adjacent or irrelevant rules.",
            "Rule provenance upper bound includes failed prefix questions too.",
            "Do not silently correct source text, replace rejected examples, or truncate.",
            "Resolve source families and content quality, then review a new protocol.",
            "Budget and model-run authorization must precede any submission.",
            "No negative algorithm or novelty conclusion follows from asset insufficiency.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/e4-assets-v1/assets.json.gz")
    parser.add_argument("--review", type=Path, default=ROOT / "configs/stage_a_review.json")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / ".cache/stage-a-tokenizer")
    parser.add_argument("--official", type=Path, default=ROOT / "third_party/SeqMem-Eval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    require(output.parent.is_dir(), "Create an independent output directory first")
    for protected in (args.assets.resolve(), args.review.resolve()):
        require(output.resolve() != protected, "Cannot overwrite preparation inputs")
    for protected in (ROOT / "runs", ROOT / "third_party", args.tokenizer.resolve()):
        require(protected.resolve() not in output.resolve().parents, "Protected output directory")
    result = prepare(args.assets, args.review, args.tokenizer, args.official)
    publish_bytes(output, gzip.compress(canonical(seal(result)), mtime=0))
    print(json.dumps(dict(status=result["status"], identity=result["identity"],
                          counts=result["counts"], budget=result["budget"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
