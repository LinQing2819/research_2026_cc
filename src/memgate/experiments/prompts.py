"""CPU retrieval, independent component packing and exact chat tokenization."""
from __future__ import annotations

import time

import numpy as np

from .common import ARMS, FACTORS, LIMITS, digest, require
from .source import rule_items

RULE_HEADER = "# RULEBOOK (EXISTING INSIGHTS)\n"
CASE_HEADER = "# FEWSHOT EXAMPLES (DYNAMIC, RETRIEVED FROM SUCCESS POOL)\n"


def checked_vectors(vectors, count):
    arr = np.asarray(vectors, dtype=np.float32)
    require(arr.ndim == 2 and arr.shape[0] == count and arr.shape[1] > 0,
            "Embedding count/dimension mismatch")
    require(np.isfinite(arr).all() and (np.linalg.norm(arr, axis=1) > 0).all(),
            "Non-finite/zero embedding")
    require(np.allclose(np.linalg.norm(arr, axis=1), 1, atol=1e-4),
            "Embedding vectors are not normalized")
    return arr


def retrieve(official, pool, vectors, query_vec, query):
    if not pool:
        return []
    vectors = checked_vectors(vectors, len(pool))
    qv = checked_vectors([query_vec], 1)[0]
    require(len(qv) == vectors.shape[1], "Embedding dimension changed")
    # Keep float32 scalar arithmetic identical to the official _encode/_cosine.
    vv, qv = [list(v) for v in vectors], list(qv)
    scores = [float(official.ref._cosine(qv, v)) for v in vv]
    indices = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)[:3]
    result = [{**pool[i], "retrieval_score": scores[i]} for i in indices]
    reference = official.reference_retrieval(pool, vv, qv, query)
    for actual, expected in zip(result, reference):
        require(all(actual[k] == expected[k] for k in ("question", "trajectory", "retrieval_score")),
                "Official retrieval fidelity failed")
    require(len(result) == len(reference), "Official retrieval count mismatch")
    return result


def encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def chat(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    ids = [int(x) for x in ids]
    require(ids == list(encode_text(tokenizer, text)), "Chat template adds unaccounted special tokens")
    return {"messages": messages, "text": text, "ids": ids}


def pack_rules(tokenizer, rules, limit=1024):
    items = rule_items(rules)
    for count in range(len(items), -1, -1):
        text = "".join(items[:count]).rstrip("\n") if count else "(empty)"
        if len(encode_text(tokenizer, RULE_HEADER + text)) <= limit:
            return dict(text=text, kept=count, total=len(items),
                        tokens=len(encode_text(tokenizer, RULE_HEADER + text)))
    raise RuntimeError("The empty rule marker exceeds the protocol budget")


def pack_cases(tokenizer, official, cases, limit=3072):
    for count in range(len(cases), -1, -1):
        text = official.cases(cases[:count])
        if len(encode_text(tokenizer, CASE_HEADER + text)) <= limit:
            return dict(text=text, items=cases[:count], kept=count, total=len(cases),
                        ids=[x["qid"] for x in cases[:count]],
                        tokens=len(encode_text(tokenizer, CASE_HEADER + text)))
    raise RuntimeError("The empty experience marker exceeds the protocol budget")


def token_span(tokenizer, full, start, end):
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
               if b > start and a < end]
    require(indices, "Tokenizer cannot expose an exact component span")
    return {"positions": [indices[0], indices[-1] + 1],
            "ids": [int(encoded["input_ids"][i]) for i in indices],
            "char_range": [start, end]}


def prepare_question(official, tokenizer, public, snapshots, pool_vectors, query_vector):
    started = time.perf_counter()
    query, problem = official.query(public), official.problem(public)
    ranked = {v: retrieve(official, snapshots[v]["pool"], pool_vectors[v], query_vector, query)
              for v in ("old", "new")}
    retrieval_seconds = time.perf_counter() - started
    packing_start = time.perf_counter()
    rules = {v: pack_rules(tokenizer, snapshots[v]["rules"], LIMITS["rules"])
             for v in ("old", "new")}
    rules["empty"] = pack_rules(tokenizer, "(empty)", LIMITS["rules"])
    cases = {v: pack_cases(tokenizer, official, ranked[v], LIMITS["experiences"])
             for v in ("old", "new")}
    cases["empty"] = pack_cases(tokenizer, official, [], LIMITS["experiences"])
    prompts = {}
    for arm in ARMS:
        r, e = FACTORS[arm]
        prompt = official.solver(problem, rules[r]["text"], cases[e]["items"])
        require(prompt.endswith(problem + "\n"), "Current problem was truncated/rewritten")
        packed = chat(tokenizer, prompt)
        # The offset is from the real full template, not a sum of fragment encodings.
        offset = packed["text"].index(prompt)
        rule_start = offset + prompt.index(RULE_HEADER)
        rule_end = rule_start + len(RULE_HEADER + rules[r]["text"])
        case_start = rule_end + 2
        require(packed["text"][case_start:case_start + len(CASE_HEADER)] == CASE_HEADER,
                "Unexpected official scaffold")
        case_end = case_start + len(CASE_HEADER + cases[e]["text"])
        raw_prompt = official.solver(problem, snapshots[r]["rules"] if r != "empty" else "(empty)",
                                     ranked[e] if e != "empty" else [])
        prompts[arm] = dict(prompt=prompt, **packed, prompt_tokens=len(packed["ids"]),
                            sha256=digest(packed["ids"]), rules_version=r, experiences_version=e,
                            rule_span=token_span(tokenizer, packed["text"], rule_start, rule_end),
                            case_span=token_span(tokenizer, packed["text"], case_start, case_end),
                            unpacked_sha256=digest(raw_prompt),
                            unpacked_tokens=len(chat(tokenizer, raw_prompt)["ids"]),
                            unchanged_from_official_unpacked=(prompt == raw_prompt))
    # Verify factor invariance in content and actual full-template token spans.
    for factor, key in (("rules_version", "rule_span"), ("experiences_version", "case_span")):
        for version in ("old", "new", "empty"):
            subset = [p[key]["ids"] for p in prompts.values() if p[factor] == version]
            require(all(x == subset[0] for x in subset), "Packing changed another factor's token span")
    oversize = {a: p["prompt_tokens"] for a, p in prompts.items()
                if p["prompt_tokens"] > LIMITS["input"]}
    return dict(public=public, query=query, problem=problem, ranked=ranked,
                rules=rules, cases=cases, prompts=prompts, eligible=not oversize,
                exclusion=({"reason": "final_input_exceeds_5888", "arms": oversize} if oversize else None),
                timings=dict(retrieval_seconds=retrieval_seconds,
                             packing_seconds=time.perf_counter() - packing_start),
                change=dict(source_rules=snapshots["old"]["rules"] != snapshots["new"]["rules"],
                            packed_rules=rules["old"]["text"] != rules["new"]["text"],
                            retrieved_experiences=ranked["old"] != ranked["new"],
                            packed_experiences=cases["old"]["text"] != cases["new"]["text"],
                            distinct_prompts=len({p["sha256"] for p in prompts.values()})))


def sentinel_prompts(tokenizer):
    short = "Whitespace:\t A\nUnicode: 数学 α + β\nOptions:\nA. red\nB. blue\nLiteral <|im_start|> marker."
    # Legal chat, close to the actual input boundary, without any evaluation content.
    unit = " neutral"
    lo, hi = 0, 6000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(chat(tokenizer, short + unit * mid)["ids"]) <= LIMITS["input"]:
            lo = mid
        else:
            hi = mid - 1
    boundary = short + unit * lo
    packed = [chat(tokenizer, p) for p in (short, boundary)]
    require(LIMITS["input"] - 16 <= len(packed[1]["ids"]) <= LIMITS["input"],
            "Could not build a legal near-boundary sentinel")
    return packed
