"""Bounded execution of the separately authorized, frozen nine-pair diagnostic.

The client uses only the standard library. No selection, curation or training.
The E2 implementation is reused unchanged; its experiment limits are overridden.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile

from .experiments.common import (
    Invalid, atomic_json, canonical, digest, file_hash, require, seal, unseal,
)
from .experiments.ledger import Ledger, Transport, complete_success, payload, validate_response
from .experiments.official import Official

SLOT = "v3-curated-9q-20261104"
PREPARATION_ID = "285636a7bb4c884c4581403fe9dd286f7f360ee65864367e4cbf20d700b70ce0"
PREPARATION_DIGEST = "9fdd8628c331447feddd649542e42661cab8ee68366384f3129afb4eb10560ff"
PREPARATION_FILE = "ea58096cd5dda94c6d7bc9e792aaa97e353e03083c7178f2cfc99d0fdd8af7b4"
SERVER_ROOT = Path("/data/zlq/research_2026_cc")
CAPS = dict(context=8192, output=2048, safety=256, input=5888,
            rules=1024, experiences=3072, logical=72, probes=2,
            retries=8, attempts=80, per_request=2)
ARMS = ("00", "10", "01", "11")


def read_sealed(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return unseal(json.load(stream))


def publish_gzip(path, value):
    """Publish immutable snapshots with the existing durable atomic writer."""
    data = gzip.compress(canonical(seal(value)), mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"Immutable export changed: {path}")
        return
    fd, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-clobber publication; interrupted exports can be regenerated.
        os.link(pending, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(pending)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def requests(data):
    pairs = {p["qid"]: p for p in data["preparation"]["pairs"]}
    result = {}
    for item in data["preparation"]["draft_schedule"]:
        request = payload(pairs[item["qid"]]["prompts"][item["arm"]]["ids"], item["seed"])
        require(digest(request) == item["request_sha256"], "Frozen request/seed changed")
        result[item["logical_sha256"]] = dict(request=request, probe=False, item=item)
    for index, sent in enumerate(data["sentinels"]):
        result[f"probe:{index}"] = dict(
            request=payload(sent["ids"], 2000000000 + index), probe=True, item=None)
    return result


def validate_package(data):
    require(data["schema"] == "V3_STAGE_A_EXECUTION_1" and data["slot"] == SLOT,
            "Unknown execution authorization")
    require(data["limits"] == CAPS and data["shared_probe_retry_cap"] == 8,
            "Authorization budget changed")
    prep = data["preparation"]
    require(digest(prep) == PREPARATION_DIGEST and prep["identity"] == PREPARATION_ID,
            "Frozen preparation changed")
    require(not prep["executable_schedule"] and prep["counts"]["new_solver_calls"] == 0,
            "Preparation is not the frozen zero-call artifact")
    require(len(data["sentinels"]) == 2, "Exactly two probes are authorized")
    for sent in data["sentinels"]:
        require(digest(sent["ids"]) == sent["sha256"], "Probe token IDs changed")
    qids = {p["qid"] for p in prep["pairs"]}
    require(len(qids) == 9 and set(data["scoring_gold"]) == qids, "Scoring-only gold mismatch")
    for pair in prep["pairs"]:
        gold = data["scoring_gold"][pair["qid"]]
        require(isinstance(gold, str) and len(gold) == 1
                and 0 <= ord(gold) - 65 < len(pair["public"]["options"]), "Invalid scoring gold")
    allowed = requests(data)
    require(len(allowed) == 74, "Duplicate or missing request identity")
    require(data["allowed_requests"] == {k: digest(v["request"]) for k, v in allowed.items()},
            "Authorized request allowlist changed")
    require(digest(data["model_files"]) == prep["provenance"]["model_files_sha256"],
            "Model identity differs from preparation")
    require(data["official_files"] == prep["provenance"]["official"]["files"],
            "Official scorer identity differs from preparation")
    return allowed


def load_package(directory, expected_hash, archive=None):
    directory = Path(directory)
    path = directory / "execution.json.gz"
    require(file_hash(path) == expected_hash, "Execution package differs from tool authorization")
    data = read_sealed(path)
    validate_package(data)
    require(file_hash(directory / "preparation-v2.json.gz") == PREPARATION_FILE,
            "Sealed preparation file changed")
    for name, expected in data["official_files"].items():
        require(file_hash(directory / "official" / name) == expected, "Official source changed")
    if archive is not None:
        with zipfile.ZipFile(archive) as zipped:
            for name, expected in data["runtime_code"].items():
                require(hashlib.sha256(zipped.read(name)).hexdigest() == expected,
                        f"Runtime source changed: {name}")
    return data


def score(extractor, gold):
    def transform(text):
        predicted = extractor(text)
        return dict(predicted=predicted, score=int(bool(predicted) and predicted == gold))
    return transform


def binding(data, meta, entry):
    common = dict(execution=digest(data), service=digest(meta["service"]))
    if not entry["probe"]:
        item = entry["item"]
        common.update(preflight=meta["preflight"],
                      **{k: item[k] for k in ("qid", "arm", "rep", "seed")})
    return common


class DiagnosticLedger(Ledger):
    """One fixed slot, checked before opening SQLite; no identity migration."""

    def __init__(self, root, data, *, initialize=False):
        root = Path(root).resolve()
        self.allowed = validate_package(data)
        identity = digest(data)
        guard_dir = root / ".stage-a-authorizations"
        ledger_dir = root / "runs" / ".stage-a-ledgers"
        guard_dir.mkdir(parents=True, exist_ok=True)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        marker = guard_dir / f"{SLOT}.json"
        path = ledger_dir / f"{SLOT}.sqlite"
        # Serialize first-use checks as well as the complete write session.
        self.guard = (guard_dir / f"{SLOT}.lock").open("a+b")
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            expected = dict(slot=SLOT, identity=identity, ledger=str(path))
            if marker.exists():
                require(json.loads(marker.read_text()) == expected, "Budget slot identity changed")
                require(path.is_file(), "Budget ledger disappeared; cannot recreate allowance")
                uri = path.as_uri() + "?mode=ro"
                with sqlite3.connect(uri, uri=True) as check:
                    row = check.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
                    require(row is not None and json.loads(row[0]) == identity,
                            "Existing ledger is empty or has another identity")
            else:
                require(initialize, "Initialize the single authorized budget slot first")
                require(not path.exists() and not path.with_suffix(".sqlite-wal").exists(),
                        "Orphaned budget ledger; do not reset or initialize again")
                # Marker becomes durable BEFORE the DB can ever dispatch.
                with marker.open("x", encoding="utf-8") as stream:
                    stream.write(canonical(expected).decode() + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                fsync_directory(guard_dir)
            super().__init__(path, identity, CAPS, data_binding=PREPARATION_ID)
            fsync_directory(ledger_dir)
        except BaseException:
            self.guard.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            self.guard.close()

    def reserve(self, key, request, probe=False):
        entry = self.allowed.get(key)
        require(entry is not None and entry["probe"] == probe
                and entry["request"] == request, "Request is outside the authorized allowlist")
        counts = self.counts()
        if probe or self.attempts(key):
            require(counts["probes"] + counts["retries"] < 8,
                    "Shared probe/retry reserve exhausted")
        return super().reserve(key, request, probe)


def check_service(identity, data):
    require(identity["actual_max_model_len"] == 8192, "Server context must be exactly 8192")
    require(identity["served_models"] == ["Qwen/Qwen3-8B"], "Server model mapping changed")
    require(identity["model_files"] == data["model_files"], "Server model files changed")
    required = dict(dtype="torch.bfloat16", tensor_parallel_size=1, max_num_seqs=1,
                    max_num_batched_tokens=8192, enforce_eager=True, generation_config="vllm")
    require(all(identity[k] == v for k, v in required.items()), "Server launch settings changed")
    require(len(identity["gpu"]) == 1, "Exactly one visible GPU is required")
    require(identity["runtime_versions"]["vllm"] == "0.8.5", "Unexpected vLLM version")
    require(identity["launcher_sha256"] == data["launcher_sha256"], "Server launcher changed")


def service_evidence(transport, data):
    identity = transport.json("/e2/identity")
    check_service(identity, data)
    require([m["id"] for m in transport.json("/v1/models")["data"]] == ["Qwen/Qwen3-8B"],
            "API model listing changed")
    prompts = data["sentinels"] + [
        p["prompts"][arm] for p in data["preparation"]["pairs"] for arm in ARMS]
    checked = []
    for p in prompts:
        chat = transport.json("/e2/chat_ids", dict(messages=p["messages"]))
        require(chat["ids"] == p["ids"] and chat["text"] == p["text"],
                "Live full chat template differs from frozen prompt")
        tokens = transport.json("/tokenize", dict(
            model="Qwen/Qwen3-8B", prompt=p["text"], add_special_tokens=False))
        require(tokens.get("tokens") == p["ids"] and tokens.get("max_model_len") == 8192,
                "Live token IDs or context differ")
        checked.append(dict(prompt=p["sha256"], chat=digest(chat), tokens=digest(tokens)))
    return dict(identity=identity, checked_prompts=checked)


def metadata(ledger):
    return {r["key"]: json.loads(r["value"]) for r in ledger.db.execute("SELECT * FROM meta")}


def preflight(data, ledger, transport):
    evidence = service_evidence(transport, data)
    ledger.bind("service", evidence)
    for i in range(2):
        key = f"probe:{i}"
        entry = ledger.allowed[key]
        print(f"Preflight {i+1}/2; all possible dispatches are charged.", flush=True)
        ledger.complete(key, entry["request"], transport, lambda text: {},
                        binding(data, metadata(ledger), entry), probe=True)
    ledger.bind("preflight", digest(dict(
        execution=digest(data), service=evidence,
        probes=[digest(ledger.result(f"probe:{i}")) for i in range(2)])))


def snapshot(ledger):
    # This is a logical, consistent export of live SQLite including its WAL.
    # The process lock is held for the entire run, including this transaction.
    with ledger.db:
        return dict(meta=metadata(ledger),
                    attempts=[dict(r) for r in ledger.db.execute("SELECT * FROM attempts ORDER BY id")],
                    results=[dict(r) for r in ledger.db.execute("SELECT * FROM results ORDER BY key")])


def readonly_snapshot(root, data):
    root = Path(root).resolve()
    marker = root / ".stage-a-authorizations" / f"{SLOT}.json"
    path = root / "runs" / ".stage-a-ledgers" / f"{SLOT}.sqlite"
    require(marker.is_file() and path.is_file(), "Existing budget marker/ledger required for export")
    require(json.loads(marker.read_text()) == dict(slot=SLOT, identity=digest(data), ledger=str(path)),
            "Read-only export identity changed")
    # The same guard excludes a writer; mode=ro reads the active WAL normally.
    with marker.with_suffix(".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN")
            meta = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT * FROM meta")}
            return dict(meta=meta,
                        attempts=[dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY id")],
                        results=[dict(r) for r in db.execute("SELECT * FROM results ORDER BY key")])
        finally:
            db.close()


def audit_snapshot(data, state, extractor):
    allowed = validate_package(data)
    meta = state["meta"]
    require(meta["identity"] == digest(data) and meta["limits"] == CAPS
            and meta["data_binding"] == PREPARATION_ID, "Export identity/budget differs")
    counts = dict(logical=0, probes=0, retries=0, attempts=0)
    prior = {}
    attempt_ids = set()
    for row in state["attempts"]:
        key = row["key"]
        require(row["id"] not in attempt_ids, "Duplicate attempt ID")
        attempt_ids.add(row["id"])
        entry = allowed.get(key)
        require(entry is not None, "Foreign attempt in diagnostic ledger")
        request = json.loads(row["payload"])
        require(request == entry["request"] and digest(request) == row["payload_hash"],
                "Attempt payload changed")
        rows = prior.setdefault(key, [])
        kind = "retries" if rows else ("probes" if entry["probe"] else "logical")
        require(row["kind"] == kind, "Incorrect attempt accounting kind")
        require(len(rows) < (1 if entry["probe"] else 2), "Per-request allowance exceeded")
        if rows:
            from .experiments.ledger import retryable
            earlier = rows[-1]
            require(earlier["raw"] is None or retryable(json.loads(earlier["raw"])),
                    "Retried a completed non-retryable response")
        if row["raw"] is not None:
            require(digest(json.loads(row["raw"])) == row["raw_hash"], "Raw response changed")
        counts[kind] += 1
        counts["attempts"] += 1
        rows.append(row)
    require(all(counts[k] <= CAPS[k] for k in counts)
            and counts["probes"] + counts["retries"] <= 8, "Export exceeds authorization")
    accepted = {}
    for row in state["results"]:
        key, value = row["key"], json.loads(row["value"])
        require(key not in accepted and digest(value) == row["hash"], "Result checksum/identity changed")
        require(key in prior, "Result has no reserved attempt")
        entry = allowed[key]
        successes = [r for r in prior[key] if r["raw"] is not None
                     and complete_success(json.loads(r["raw"]))]
        require(successes and successes[0]["id"] == value["attempt_id"],
                "Accepted result is not the first persisted successful response")
        raw = json.loads(successes[0]["raw"])
        parsed = validate_response(raw, entry["request"])
        scored = {} if entry["probe"] else score(extractor, data["scoring_gold"][
            entry["item"]["qid"]])(parsed["text"])
        expected = dict(binding=binding(data, meta, entry), attempt_id=successes[0]["id"],
                        request_hash=digest(entry["request"]), raw_hash=digest(raw), **parsed, **scored)
        require(value == expected, "Result differs from response, input binding or scorer")
        accepted[key] = value
    if "service" in meta:
        check_service(meta["service"]["identity"], data)
    if "preflight" in meta:
        require(all(f"probe:{i}" in accepted for i in range(2)), "Preflight probes missing")
        expected = digest(dict(execution=digest(data), service=meta["service"],
                               probes=[digest(accepted[f"probe:{i}"]) for i in range(2)]))
        require(meta["preflight"] == expected, "Preflight record changed")
    require(not any(not e["probe"] and k in prior for k, e in allowed.items())
            or "preflight" in meta, "Generation occurred without preflight")
    return counts, accepted


def analyze(data, state, extractor):
    counts, accepted = audit_snapshot(data, state, extractor)
    per_question = []
    schedule = data["preparation"]["draft_schedule"]
    for pair in data["preparation"]["pairs"]:
        qid = pair["qid"]
        arms = {}
        for arm in ARMS:
            rows = [accepted[x["logical_sha256"]] for x in schedule
                    if x["qid"] == qid and x["arm"] == arm and x["logical_sha256"] in accepted]
            arms[arm] = dict(n=len(rows), correct=sum(r["score"] for r in rows),
                             p=sum(r["score"] for r in rows) / 2 if len(rows) == 2 else None,
                             unparsed=sum(not r["predicted"] for r in rows),
                             length=sum(r["finish_reason"] == "length" for r in rows))
        contrasts = None
        if all(a["n"] == 2 for a in arms.values()):
            p = {a: arms[a]["p"] for a in ARMS}
            contrasts = dict(interaction=p["11"] - p["10"] - p["01"] + p["00"],
                             joint_vs_empty=p["11"] - p["00"],
                             joint_vs_rule=p["11"] - p["10"],
                             joint_vs_example=p["11"] - p["01"])
        per_question.append(dict(qid=qid, family=pair["family"], arms=arms, contrasts=contrasts))
    complete = all(q["contrasts"] is not None for q in per_question)
    formal = [r for k, r in accepted.items() if not k.startswith("probe:")]
    known = [json.loads(a["raw"]) for a in state["attempts"] if a["raw"] is not None]
    # Response usage is only known for valid completed responses. Every reserved
    # request is separately charged at its token upper bound, including failures.
    observed = []
    for a in state["attempts"]:
        if a["raw"] is not None and complete_success(json.loads(a["raw"])):
            try:
                observed.append(validate_response(json.loads(a["raw"]), json.loads(a["payload"]))["usage"])
            except Invalid:
                pass
    summary = dict(
        status="DESCRIPTIVE_COMPLETE" if complete else "INCOMPLETE_NO_EFFECT_SUMMARY",
        execution=digest(data), slot=SLOT, attempt_counts=counts,
        accepted_formal=len(formal), required_formal=72, questions=per_question,
        overall=None, cost=dict(
            reserved_input_tokens=sum(len(json.loads(a["payload"])["prompt"]) for a in state["attempts"]),
            reserved_output_upper_bound=counts["attempts"] * 2048,
            validated_response_count=len(observed),
            observed_input_tokens=sum(x["prompt_tokens"] for x in observed),
            observed_output_tokens=sum(x["completion_tokens"] for x in observed),
            missing_or_unvalidated_usage=counts["attempts"] - len(observed),
            raw_responses=len(known)),
        boundaries=[
            "Nine hand-selected development questions; descriptive estimates only.",
            "Two decodes are averaged within each question; not 18 independent questions.",
            "Interaction is on the accuracy scale and is not proof of a general mechanism.",
            "No learned selection, curator update, eviction or training was tested.",
            "No automatic expansion or effect-based stopping; historical budgets are untouched.",
        ])
    if complete:
        summary["overall"] = dict(
            accuracy={a: sum(q["arms"][a]["p"] for q in per_question) / 9 for a in ARMS},
            contrasts={k: sum(q["contrasts"][k] for q in per_question) / 9
                       for k in per_question[0]["contrasts"]},
            interaction_signs=dict(Counter(
                "positive" if q["contrasts"]["interaction"] > 0 else
                "negative" if q["contrasts"]["interaction"] < 0 else "zero"
                for q in per_question)))
    return summary


def markdown_report(report):
    lines = [
        "# V3 九题固定配对诊断", "",
        f"状态：`{report['status']}`；正式回答 {report['accepted_formal']}/72。",
        f"计费尝试：{json.dumps(report['attempt_counts'], ensure_ascii=False)}。", "",
        "| 题号 | 00 | 10（R） | 01（E） | 11（R+E） | I |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        values = [f"{q['arms'][a]['correct']}/{q['arms'][a]['n']}" for a in ARMS]
        interaction = "—" if q["contrasts"] is None else f"{q['contrasts']['interaction']:+.2f}"
        lines.append(f"| {q['qid']} | " + " | ".join(values) + f" | {interaction} |")
    if report["overall"] is not None:
        lines += ["", "题内两次平均后、九题等权的点估计：", "",
                  "```json", json.dumps(report["overall"], ensure_ascii=False, indent=2), "```"]
    lines += ["", "九道人工提名的开发题，只报告描述结果；不作显著性或独立确认声明。"
              "本轮没有验证自由选择、失败更新或淘汰，结果不触发自动扩量。", "",
              "成本含所有预留尝试；观测 token 仅覆盖通过校验的响应，失败的未知用量不记为零。",
              "", "```json", json.dumps(report["cost"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def export_run(data, state, extractor, output):
    report = analyze(data, state, extractor)
    envelope = dict(schema="V3_STAGE_A_EXPORT_1", execution=digest(data), state=state, report=report)
    tag = digest(envelope)[:16]
    output = Path(output)
    publish_gzip(output / f"export-{tag}.json.gz", envelope)
    atomic_json(output / f"analysis-{tag}.json", seal(report))
    text = markdown_report(report)
    path = output / f"report-{tag}.md"
    if path.exists():
        require(path.read_text() == text, "Immutable report changed")
    else:
        path.write_text(text, encoding="utf-8")
    return report


def run(data, ledger, transport, extractor, output):
    audit_snapshot(data, snapshot(ledger), extractor)
    preflight(data, ledger, transport)
    items = data["preparation"]["draft_schedule"]
    for index, item in enumerate(items):
        key = item["logical_sha256"]
        entry = ledger.allowed[key]
        ledger.complete(
            key, entry["request"], transport, score(extractor, data["scoring_gold"][item["qid"]]),
            binding(data, metadata(ledger), entry))
        print(f"Formal {index+1}/72 {item['qid']} {item['arm']} rep={item['rep']} "
              f"counts={ledger.counts()}", flush=True)
        if index == 7:
            # Smoke checks structural acceptance only; accuracy never controls continuation.
            audit_snapshot(data, snapshot(ledger), extractor)
            ledger.bind("smoke", digest([ledger.result(x["logical_sha256"]) for x in items[:8]]))
            print("First eight fixed requests accepted; continuing automatically.", flush=True)
    return export_run(data, snapshot(ledger), extractor, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "init", "run", "export", "analyze"))
    parser.add_argument("--package", type=Path, default=Path(sys.argv[0]).resolve().parent)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--export-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # The builder embeds the immutable input hash inside the standalone tool.
    from stage_a_authorization import INPUT_SHA256
    data = load_package(args.package, INPUT_SHA256, sys.argv[0])
    extractor = Official(args.package / "official").extract_letter
    if args.command == "verify":
        print(json.dumps(dict(execution=digest(data), slot=SLOT, formal=72, cap=80,
                              new_model_calls=0), indent=2))
        return
    if args.command == "analyze":
        require(args.export_file is not None and args.output is not None,
                "Analyze requires --export-file and --output")
        exported = read_sealed(args.export_file)
        require(exported["schema"] == "V3_STAGE_A_EXPORT_1" and exported["execution"] == digest(data),
                "Export belongs to another execution")
        report = analyze(data, exported["state"], extractor)
        require(report == exported["report"], "Export analysis does not replay exactly")
        export_run(data, exported["state"], extractor, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    require(SERVER_ROOT.is_dir(), "Generation/init/export require the fixed server repository")
    output = SERVER_ROOT / "runs" / SLOT / "exports"
    require(args.output is None or args.output.resolve() == output, "Execution output path is fixed")
    if args.command == "export":
        export_run(data, readonly_snapshot(SERVER_ROOT, data), extractor, output)
        return
    with DiagnosticLedger(SERVER_ROOT, data, initialize=args.command == "init") as ledger:
        if args.command == "init":
            audit_snapshot(data, snapshot(ledger), extractor)
            print(f"Budget slot ready: {ledger.counts()}", flush=True)
        else:
            try:
                run(data, ledger, Transport(args.base_url), extractor, output)
            except BaseException:
                # Preserve partial results without pretending partial arms are final effects.
                try:
                    export_run(data, snapshot(ledger), extractor, output)
                except Exception as export_error:
                    print(f"Partial export failed; retain original ledger/WAL: {export_error}",
                          file=sys.stderr, flush=True)
                raise


if __name__ == "__main__":
    main()
