"""Sealed historical assets. Raw strings are JSON data, never executable source."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from memgate.experiments.common import (
    ARMS, FACTORS, LIMITS, TASK, canonical, digest, require, seal, unseal,
)
from memgate.experiments.ledger import payload
from memgate.experiments.source import build_snapshots, rule_items
from memgate.replay import export as e3
from . import MISSING_GATES, STATUS, VERSION
from .provenance import MODULE_FILES, implementation_identity

REQUIRED = ("prefix_raw.json", "old_prefix_raw.json", "public.json",
            "snapshots.json", "scoring_gold.json")
OPTIONAL = ("splits.json", "evaluation_splits.json", "data_qualification.json",
            "duplicate_audit.json")
HEADERS = ("# RULEBOOK (EXISTING INSIGHTS)\n",
           "# FEWSHOT EXAMPLES (DYNAMIC, RETRIEVED FROM SUCCESS POOL)\n")


def output_path(path):
    # Resolve parents, never the leaf: even dangling symlinks must not be replaced.
    path = Path(path).absolute()
    path = path.parent.resolve() / path.name
    require(not os.path.lexists(path), f"Output already exists: {path}")
    return path


def publish_bytes(path, data):
    """Atomic, exclusive publication; failures leave no partially written output."""
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=".e4-assets-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def artifact_names(manifest):
    for name in REQUIRED:
        require(name in manifest["files"], f"Missing required manifest source: {name}")
    return (*REQUIRED, *(n for n in OPTIONAL if n in manifest["files"]),
            *(f"questions/{q}.json" for p in ("dev", "test") for q in manifest["eligible"][p]))


def decode_artifact(name, text, manifest):
    require(isinstance(text, str), f"Artifact must contain exact UTF-8 file text: {name}")
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == manifest["files"][name],
            f"Manifest raw-byte checksum mismatch: {name}")
    return unseal(json.loads(text))


def public_index(rows):
    require(isinstance(rows, list) and rows, "Missing public inputs")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"qid", "question", "options"},
                "Public inputs must contain only qid, question and options")
        qid = row["qid"]
        require(isinstance(qid, str) and qid.isdigit() and qid not in result,
                "Invalid/duplicate public qid")
        require(isinstance(row["question"], str) and row["question"].strip()
                and isinstance(row["options"], list) and 2 <= len(row["options"]) <= 10
                and all(isinstance(x, str) for x in row["options"]), "Invalid public question")
        result[qid] = row
    return result


def validate_prefixes(artifacts, table):
    boundaries = table["manifest"]["boundaries"]
    old, new = boundaries["old"], boundaries["new"]
    require((old, new) == (193, 581), "Expected historical 193/581 endpoints")
    raw, old_raw = artifacts["prefix_raw.json"], artifacts["old_prefix_raw.json"]
    require(isinstance(raw, str) and isinstance(old_raw, str), "Raw prefixes must be JSONL strings")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == new and all(x.strip() for x in lines), "Wrong/blank raw prefix rows")
    require(old_raw == "".join(lines[:old]), "Old raw prefix is not the exact leading prefix")
    prefix = [json.loads(line) for line in lines]
    public = artifacts["public.json"]
    index = public_index(public)
    require(len(public) == boundaries["total"], "Public count/boundary mismatch")
    for row, expected in zip(prefix, public):
        require(isinstance(row, dict) and row.get("task") == TASK and
                str(row.get("qid")) == expected["qid"] and
                row.get("question") == expected["question"] and
                row.get("options") == expected["options"], "Prefix/public input binding mismatch")
    snapshots = build_snapshots(prefix, old, new)
    require(snapshots == artifacts["snapshots.json"] == table["snapshots"],
            "Raw prefix chain/snapshots mismatch")
    return index


def validate_prompt(question, arm):
    prompt = question["prompts"][arm]
    ids = prompt["ids"]
    require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
            and len(ids) == prompt["prompt_tokens"] <= LIMITS["input"]
            and digest(ids) == prompt["sha256"], "Prompt ID/count/hash mismatch")
    rv, ev = FACTORS[arm]
    require((prompt["rules_version"], prompt["experiences_version"]) == (rv, ev),
            "Prompt version binding mismatch")
    require(isinstance(prompt["prompt"], str) and isinstance(prompt["text"], str)
            and prompt["prompt"] in prompt["text"]
            and prompt["prompt"].endswith(question["problem"] + "\n")
            and prompt["messages"] == [{"role": "user", "content": prompt["prompt"]}],
            "Prompt text/message binding mismatch")
    for key, header, block in (
        ("rule_span", HEADERS[0], question["rules"][rv]["text"]),
        ("case_span", HEADERS[1], question["cases"][ev]["text"]),
    ):
        span = prompt[key]
        start, end = span["positions"]
        left, right = span["char_range"]
        require(all(type(i) is int for i in (start, end, left, right))
                and 0 <= start < end <= len(ids) and 0 <= left < right <= len(prompt["text"])
                and ids[start:end] == span["ids"]
                and prompt["text"][left:right] == header + block,
                "Actual packed input/span binding mismatch")


def validate_table(table, artifacts):
    manifest, analysis = table["manifest"], table["analysis"]
    require(table["protocol"] == e3.PROTOCOL and table["run_id"] == e3.RUN_ID
            and digest(manifest["fingerprint"]) == manifest["identity"] == e3.RUN_ID,
            "Wrong E2 table protocol/run identity")
    require(digest(manifest) == table["manifest_sha256"] == analysis["manifest"]
            and digest(analysis) == table["source_analysis_sha256"],
            "E2 table manifest/analysis binding mismatch")
    verification = table["verification"]
    require(all(verification[k] is True for k in (
        "original_scorer", "all_manifest_artifacts", "wal_visible_readonly_transaction"))
        and verification["attempt_counts"] == analysis["attempt_counts"],
        "Missing E2 export verification")
    splits = table["splits"]
    require(splits == manifest["eligible"] and set(splits) == {"dev", "test"}
            and len(splits["dev"]) == 64 and len(splits["test"]) == 94,
            "Expected complete 64/94 E2 table")
    qids = splits["dev"] + splits["test"]
    require(len(set(qids)) == 158 and set(table["questions"]) == set(qids),
            "Duplicate/missing E2 question IDs")
    public = validate_prefixes(artifacts, table)
    for qid, question in table["questions"].items():
        require(question == artifacts[f"questions/{qid}.json"] and question["eligible"] is True
                and question["public"] == public[qid], "Frozen question/input binding mismatch")
        require(set(question["prompts"]) == set(ARMS), "Missing/foreign prompt arms")
        for arm in ARMS:
            validate_prompt(question, arm)
        for version in ("old", "new"):
            rules, cases = question["rules"][version], question["cases"][version]
            source_rules = rule_items(table["snapshots"][version]["rules"])
            require(type(rules["kept"]) is int and 0 <= rules["kept"] <= len(source_rules)
                    and rules["total"] == len(source_rules)
                    and rules["text"] == ("".join(source_rules[:rules["kept"]]).rstrip("\n")
                                          if rules["kept"] else "(empty)"),
                    "Packed rules/snapshot binding mismatch")
            pool = {r["qid"]: r for r in table["snapshots"][version]["pool"]}
            require(type(cases["kept"]) is int and 0 <= cases["kept"] <= cases["total"]
                    and cases["total"] == len(question["ranked"][version])
                    and cases["items"] == question["ranked"][version][:cases["kept"]]
                    and cases["ids"] == [r["qid"] for r in cases["items"]],
                    "Packed experiences/order binding mismatch")
            for item in question["ranked"][version]:
                require({k: item[k] for k in ("qid", "question", "trajectory")} == pool[item["qid"]],
                        "Retrieved experience/snapshot binding mismatch")
    rows = {}
    for row in table["rows"]:
        key = (row["phase"], row["qid"], row["arm"], row["rep"])
        require(key not in rows and type(row["score"]) is int and row["score"] in (0, 1),
                "Duplicate/invalid answer row")
        rows[key] = row
    expected = {(p, q, a, r) for p in splits for q in splits[p] for a in ARMS for r in range(2)}
    require(set(rows) == expected, "Incomplete/foreign E2 answer rows")
    collected = {}
    for phase in ("dev", "test"):
        items = manifest["schedule"][phase]
        require(len(items) == len(splits[phase]) * 14 and
                {(x["phase"], x["qid"], x["arm"], x["rep"]) for x in items} ==
                {k for k in expected if k[0] == phase}, "Schedule/input binding mismatch")
        collected[phase] = []
        for item in items:
            row = rows[phase, item["qid"], item["arm"], item["rep"]]
            require(row["usage"]["prompt_tokens"] ==
                    table["questions"][item["qid"]]["prompts"][item["arm"]]["prompt_tokens"],
                    "Answer/prompt token binding mismatch")
            require(row["score"] == int(bool(row["predicted"]) and
                    row["predicted"] == artifacts["scoring_gold.json"][item["qid"]]),
                    "Answer/gold binding mismatch")
            collected[phase].append({k: v for k, v in row.items() if k not in ("phase", "predicted")})
    require(digest(collected["dev"]) == analysis["selection"]["dev_sha256"]
            and digest(analysis["selection"]) == analysis["selection_sha256"]
            and digest(collected["dev"] + collected["test"]) == analysis["result_sha256"],
            "E2 answer projection/analysis mismatch")
    return rows


def development_binding(table, rows):
    questions, manifest = table["questions"], table["manifest"]
    bindings = []
    for phase in ("dev", "test"):
        for item in manifest["schedule"][phase]:
            arm, qid = item["arm"], item["qid"]
            if arm not in ("OO", "NN"):
                continue
            q = questions[qid]
            p = q["prompts"][arm]
            version = "old" if arm == "OO" else "new"
            row = rows[phase, qid, arm, item["rep"]]
            bindings.append(dict(
                e2_phase=phase, qid=qid, arm=arm, rep=item["rep"], key=item["key"], seed=item["seed"],
                question_artifact_sha256=manifest["files"][f"questions/{qid}.json"],
                public_sha256=digest(q["public"]), prompt_ids_sha256=digest(p["ids"]),
                prompt_record_sha256=digest(p), request_sha256=digest(payload(p["ids"], item["seed"])),
                rules_sha256=digest(q["rules"][version]), cases_sha256=digest(q["cases"][version]),
                result_sha256=row["result_sha256"], row_sha256=digest(row), score=row["score"],
            ))
    require(len(bindings) == 632, "Expected 632 historical NN/OO answers")
    return dict(contrast="NN-OO", role="development_only_including_former_E2_test",
                qids=table["splits"]["dev"] + table["splits"]["test"], answers=bindings)


def validate_bundle(bundle):
    require(bundle["version"] == VERSION and bundle["status"] == STATUS,
            "Wrong asset version/status")
    table = bundle["e2_table"]
    require(digest(table) == bundle["e2_table_sha256"], "E2 table checksum mismatch")
    manifest = table["manifest"]
    require(set(bundle["artifacts_utf8"]) == set(artifact_names(manifest)),
            "Missing/foreign bundled source artifacts")
    artifacts = {name: decode_artifact(name, text, manifest)
                 for name, text in bundle["artifacts_utf8"].items()}
    rows = validate_table(table, artifacts)
    require(bundle["development"] == development_binding(table, rows),
            "Development answer/input binding mismatch")
    identity = bundle["exporter"]
    require(identity["version"] == VERSION and set(identity["files"]) == set(MODULE_FILES)
            and identity["source_sha256"] == digest(identity["files"]),
            "Exporter code identity mismatch")
    return dict(
        version=VERSION, status=STATUS, bundle_sha256=digest(bundle),
        e2_table_sha256=digest(table), manifest_sha256=digest(manifest),
        exporter_source_sha256=identity["source_sha256"],
        counts=dict(development_questions=158, old_answers=632, e2_rows=len(rows),
                    historical_roots=1, public_questions=len(artifacts["public.json"]),
                    old_prefix_rows=193, new_prefix_rows=581,
                    old_pool=len(table["snapshots"]["old"]["pool"]),
                    new_pool=len(table["snapshots"]["new"]["pool"])),
        artifact_sha256={name: manifest["files"][name] for name in bundle["artifacts_utf8"]},
        raw_jsonl_sha256={name: hashlib.sha256(artifacts[name].encode("utf-8")).hexdigest()
                         for name in ("prefix_raw.json", "old_prefix_raw.json")},
        development_binding_sha256=digest(bundle["development"]),
        missing_qualification_gates=list(MISSING_GATES), new_model_calls=0,
        inspection_scope="Offline seals, manifest bytes, prefix chain and input/score bindings; "
        "not a fresh ledger rescore, retokenization or signature/authenticity proof. "
        "Answer records are E2 scored projections and result hashes, not raw completion bodies.",
    )


def export_assets(repo, output):
    repo, output = Path(repo).resolve(), output_path(output)
    for protected in (repo / "runs", repo / "third_party"):
        require(protected.resolve() not in output.parents, "Output is inside a protected source tree")
    directory = repo / "runs/e2_component" / e3.RUN_ID
    # The E3 exporter is the only ledger reader. No Ledger constructor, restoration,
    # tokenizer, embedding model, network transport or online method is invoked here.
    with tempfile.TemporaryDirectory(prefix="e4-assets-") as temp:
        path = Path(temp) / "e2-table.json.gz"
        e3.export_table(repo, path)
        table = e3.read_export(path)
    raw = {}
    for name in artifact_names(table["manifest"]):
        source = (directory / name).resolve()
        require(directory in source.parents, "Unsafe source artifact path")
        raw[name] = source.read_bytes().decode("utf-8")
    bundle = dict(version=VERSION, status=STATUS, e2_table=table, e2_table_sha256=digest(table),
                  artifacts_utf8=raw, exporter=implementation_identity())
    artifacts = {n: decode_artifact(n, text, table["manifest"]) for n, text in raw.items()}
    bundle["development"] = development_binding(table, validate_table(table, artifacts))
    summary = validate_bundle(bundle)
    publish_bytes(output, gzip.compress(canonical(seal(bundle)), mtime=0))
    return summary


def inspect_assets(path, output=None):
    if output is not None:
        output = output_path(output)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        bundle = unseal(json.load(stream))
    summary = validate_bundle(bundle)
    if output is not None:
        publish_bytes(output, canonical(seal(summary)) + b"\n")
    return summary
