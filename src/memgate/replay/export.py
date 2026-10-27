"""Verified read-only export of the completed E2 table, including committed WAL."""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path

from memgate.experiments.common import (ARMS, LIMITS, canonical, digest, file_hash,
                                       read_json, require, seal, unseal)
from memgate.experiments.ledger import payload, retryable, validate_response
from memgate.experiments.official import Official
from . import PROTOCOL, RUN_ID
from .provenance import implementation_identity


def publish_export(output, value):
    """Publish only a complete export; an existing path is never overwritten."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(canonical(seal(value)), mtime=0)
    fd, pending = tempfile.mkstemp(prefix=".e3-export-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, output)  # Atomic, exclusive publication on the same filesystem.
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def export_table(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    directory = repo / "runs/e2_component" / RUN_ID
    slot = repo / "runs/.e2-ledgers/component-v4.sqlite"
    require(slot.is_file(), f"Missing original ledger: {slot}")
    print("Verifying frozen E2 artifacts (read-only; no model calls)...", flush=True)
    manifest = unseal(read_json(directory / "manifest.json"))
    require(manifest["identity"] == RUN_ID == digest(manifest["fingerprint"]),
            "Unexpected run/fingerprint identity")
    artifacts = {}
    for name, expected in manifest["files"].items():
        target = (directory / name).resolve()
        require(directory in target.parents, "Unsafe manifest path")
        require(file_hash(target) == expected, f"Manifest artifact checksum mismatch: {name}")
        artifacts[name] = unseal(read_json(target))
    selection = unseal(read_json(directory / "selection.json"))
    preflight = unseal(read_json(directory / "preflight.json"))
    smoke = unseal(read_json(directory / "smoke.json"))
    analysis = unseal(read_json(directory / "analysis.json"))
    mh, ph, sh = digest(manifest), digest(preflight), digest(selection)
    require(all(x["manifest"] == mh for x in (preflight, selection, smoke, analysis)),
            "Artifact manifest binding mismatch")
    require(selection["preflight"] == ph == smoke["preflight"],
            "Artifact preflight binding mismatch")
    require(analysis["selection"] == selection and analysis["selection_sha256"] == sh,
            "Analysis selection mismatch")
    official_root = repo / "third_party/SeqMem-Eval"
    for name, expected in manifest["fingerprint"]["official"]["files"].items():
        require(file_hash(official_root / name) == expected, f"Official source changed: {name}")
    official = Official(official_root)
    gold = artifacts["scoring_gold.json"]
    questions = {qid: artifacts[f"questions/{qid}.json"]
                 for phase in ("dev", "test") for qid in manifest["eligible"][phase]}
    require(len(manifest["eligible"]["dev"]) == 64 and len(manifest["eligible"]["test"]) == 94,
            "E3-R1 requires the completed 64/94 table")
    require(len(questions) == 158 and not set(manifest["eligible"]["dev"]) &
            set(manifest["eligible"]["test"]), "Duplicate dev/test IDs")
    requests, bindings, records = {}, {}, {}
    for phase in ("dev", "test"):
        items = manifest["schedule"][phase]
        expected_cells = {(q, a, r) for q in manifest["eligible"][phase] for a in ARMS for r in range(2)}
        require(len(items) == len(expected_cells) and
                {(x["qid"], x["arm"], x["rep"]) for x in items} == expected_cells,
                "Schedule has missing/duplicate cells")
        for item in items:
            key = item["key"]
            require(key not in requests and item["phase"] == phase, "Invalid schedule key/phase")
            requests[key] = payload(questions[item["qid"]]["prompts"][item["arm"]]["ids"], item["seed"])
            bindings[key] = dict(manifest=mh, preflight=ph, phase=phase, qid=item["qid"],
                                 arm=item["arm"], rep=item["rep"], seed=item["seed"],
                                 selection=sh if phase == "test" else None)
            records[key] = item
    for index, sent in enumerate(artifacts["sentinels.json"]):
        key = f"probe:{index}"
        requests[key] = payload(sent["ids"], 2000000000 + index)
        bindings[key] = dict(manifest=mh, service=digest(preflight["service"]))
    require(len(requests) == 2214 and "probe:1" in requests, "Invalid probe/schedule size")

    print("Verifying ledger requests, raw responses and original scores...", flush=True)
    # URI mode=ro sees WAL; immutable=1 would silently ignore it.
    with closing(sqlite3.connect(slot.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        require([tuple(r) for r in db.execute("PRAGMA quick_check")] == [("ok",)], "Corrupt ledger")
        meta = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT * FROM meta")}
        require(meta.get("identity") == RUN_ID and meta.get("limits") == LIMITS,
                "Ledger run/limits changed")
        require(meta.get("manifest") == mh and meta.get("preflight") == ph and
                meta.get("selection") == sh and meta.get("service") == preflight["service"],
                "Ledger artifact binding mismatch")
        attempts = {}
        counts = Counter()
        for row in db.execute("SELECT * FROM attempts ORDER BY id"):
            key = row["key"]
            require(key in requests and row["kind"] in ("logical", "probes", "retries"),
                    "Foreign attempt/key/kind")
            req = json.loads(row["payload"])
            require(digest(req) == row["payload_hash"] == digest(requests[key]),
                    "Attempt payload/seed mismatch")
            raw = json.loads(row["raw"]) if row["raw"] is not None else None
            require(raw is None or digest(raw) == row["raw_hash"], "Raw response checksum mismatch")
            prior = attempts.setdefault(key, [])
            expected_kind = "retries" if prior else ("probes" if key.startswith("probe:") else "logical")
            require(row["kind"] == expected_kind, "Attempt category/order mismatch")
            prior.append(dict(id=row["id"], raw=raw))
            counts[row["kind"]] += 1
        results = {}
        for row in db.execute("SELECT * FROM results"):
            value = json.loads(row["value"])
            require(digest(value) == row["hash"], "Accepted result checksum mismatch")
            results[row["key"]] = value
        require(set(results) == set(requests) == set(attempts), "Missing/foreign results or attempts")
        checked = {}
        for key, request in requests.items():
            prior, value = attempts[key], results[key]
            require(len(prior) <= (1 if key.startswith("probe:") else LIMITS["per_request"]),
                    "Per-request/probe budget exceeded")
            accepted = next((r for r in prior if r["id"] == value["attempt_id"]), None)
            require(accepted is not None and accepted["raw"] is not None, "Missing accepted response")
            require(prior[-1]["id"] == accepted["id"], "Attempts after accepted response")
            for earlier in prior[:-1]:
                require(earlier["raw"] is None or retryable(earlier["raw"]),
                        "Accepted result follows a nonretryable/complete response")
            parsed = validate_response(accepted["raw"], request)
            scored = {}
            if key in records:
                item = records[key]
                prediction = official.extract_letter(parsed["text"])
                scored = dict(predicted=prediction,
                              score=int(bool(prediction) and prediction == gold[item["qid"]]))
            expected = dict(binding=bindings[key], attempt_id=accepted["id"],
                            request_hash=digest(request), raw_hash=digest(accepted["raw"]),
                            **parsed, **scored)
            require(value == expected, "Result differs from request, raw, binding or original scorer")
            checked[key] = value
        totals = {k: counts[k] for k in ("logical", "probes", "retries")}
        totals["attempts"] = sum(totals.values())
        require(totals == analysis["attempt_counts"] and all(totals[k] <= LIMITS[k] for k in totals),
                "Attempt counts differ from analysis/budget")
        require(preflight["probes"] == [digest(checked[f"probe:{i}"]) for i in range(2)],
                "Probe seal mismatch")
        smoke_items = [x for x in manifest["schedule"]["dev"]
                       if x["qid"] in set(manifest["eligible"]["dev"][:4])]
        require(smoke["qids"] == manifest["eligible"]["dev"][:4] and smoke["rows"] == 56
                and smoke["results"] == [digest(checked[x["key"]]) for x in smoke_items],
                "Smoke record mismatch")
        collected, projections = {}, []
        for phase in ("dev", "test"):
            collected[phase] = []
            for item in manifest["schedule"][phase]:
                v = checked[item["key"]]
                row = dict(qid=item["qid"], arm=item["arm"], rep=item["rep"], score=v["score"],
                           result_sha256=digest(v), usage=v["usage"], finish_reason=v["finish_reason"])
                collected[phase].append(row)
                projections.append(dict(phase=phase, predicted=v["predicted"], **row))
        require(digest(collected["dev"]) == selection["dev_sha256"]
                and digest(collected["dev"] + collected["test"]) == analysis["result_sha256"],
                "Export differs from analyzed rows")
        from memgate.experiments.inference import authorized_projection, select, analyze_scores
        for name, auth in manifest["authorization"].items():
            require(select(auth, authorized_projection(auth, collected["dev"])) ==
                    selection["selectors"][name], "Selector differs from authorized dev")
        recomputed = analyze_scores(manifest["eligible"]["test"], collected["test"],
                                    selection["selectors"]["S"]["arm"], selection["selectors"]["F"]["arm"])
        require(all(analysis[k] == v for k, v in recomputed.items()), "Analysis score recomputation differs")

    exported = dict(protocol=PROTOCOL, run_id=RUN_ID, manifest_sha256=mh,
                    manifest=manifest, exporter=implementation_identity(),
                    source_analysis_sha256=digest(analysis), analysis=analysis,
                    splits=manifest["eligible"], snapshots=artifacts["snapshots.json"],
                    questions=questions, rows=projections,
                    verification=dict(original_scorer=True, all_manifest_artifacts=True,
                                      wal_visible_readonly_transaction=True, attempt_counts=totals))
    publish_export(output, exported)
    print(f"Verified E3 export: {output}; rows={len(projections)}; sha256={digest(exported)}", flush=True)
    return exported


def read_export(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = unseal(json.load(stream))
    require(value["protocol"] == PROTOCOL and value["run_id"] == RUN_ID, "Wrong export protocol/run")
    require(digest(value["manifest"]) == value["manifest_sha256"] == value["analysis"]["manifest"]
            and digest(value["manifest"]["fingerprint"]) == value["manifest"]["identity"] == RUN_ID
            and digest(value["analysis"]) == value["source_analysis_sha256"],
            "Export manifest/analysis binding mismatch")
    return value
