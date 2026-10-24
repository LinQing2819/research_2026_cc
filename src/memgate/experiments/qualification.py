"""Public-only data qualification and read-only status for the one V4/V5 slot."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .common import (APPROVED_REVIEW, LEGACY_IDENTITY, LIMITS, PARAMETERS, REVIEW_PATH,
                     Invalid, atomic_json, digest, file_hash, read_json, require, seal, unseal)
from .source import (build_snapshots, duplicate_audit, load_public_inputs,
                     split_public, validate_source)


def qualify(public, splits, audit, review):
    """Every byte of the approved semantic decisions is bound, not just qids."""
    require(digest(review) == APPROVED_REVIEW, "Unapproved duplicate review content")
    require(review["schema"] == "e2-public-review-v1" and review["protocol"] == "V5",
            "Unknown review schema/protocol")
    require(digest(public) == review["public_sha256"], "Public data/order differs from review")
    require(digest(audit) == review["audit_sha256"], "Original duplicate audit differs from review")
    _, new, expected_splits = split_public(public)
    require(splits == expected_splits, "Original fixed splits changed")
    groups = {"prefix": public[:new], **splits}
    lookup = {(phase, r["qid"]): r for phase, rows in groups.items() for r in rows}
    decisions = review["decisions"]
    keys = [(tuple(r["left"]), tuple(r["right"])) for r in decisions]
    candidates = [(tuple(r["left"]), tuple(r["right"])) for r in audit["candidates"]]
    require(len(keys) == len(set(keys)) and set(keys) == set(candidates)
            and len(candidates) == len(keys), "Review must cover every candidate exactly once")
    exclusions = {"dev": [], "test": []}
    for record in decisions:
        require(record["left"][0] == "prefix" and record["right"][0] == "test",
                "Only approved prefix/test exceptions are allowed")
        for side in ("left", "right"):
            row = lookup.get(tuple(record[side]))
            require(row is not None and digest(row) == record[side + "_public_sha256"],
                    "Candidate endpoint/content differs from review")
        require(record["decision"] in ("duplicate", "distinct") and record["reason"].strip(),
                "Unresolved review decision")
        if record["decision"] == "duplicate":
            exclusions["test"].append(record["right"][1])
    # Keep original order, including partially identical interventions.
    evaluation = {p: [r for r in rows if r["qid"] not in exclusions[p]]
                  for p, rows in splits.items()}
    return dict(protocol="V5", qualified=True, public_sha256=digest(public),
                original_audit_sha256=digest(audit), review_sha256=digest(review),
                review=review, exclusions=exclusions,
                original_counts={p: len(rows) for p, rows in splits.items()},
                data_qualified_counts={p: len(rows) for p, rows in evaluation.items()},
                original_splits_sha256=digest(splits), evaluation_splits=evaluation)


def readonly_ledger(path):
    """Consistent SQLite read transaction, including committed WAL rows.

    Never use immutable=1 (it ignores WAL), create a DB, acquire a writer lock,
    instantiate Ledger, or modify meta. SQLite may manage its existing WAL/shm.
    """
    path = Path(path).resolve()
    require(path.is_file(), f"Existing budget ledger missing; zero is UNVERIFIED: {path}")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        require(db.execute("PRAGMA quick_check").fetchall() == [("ok",)], "Corrupt budget ledger")
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        counts = {k: 0 for k in ("logical", "probes", "retries")}
        for kind, n in db.execute("SELECT kind,count(*) FROM attempts GROUP BY kind"):
            require(kind in counts, "Unknown attempt kind in ledger")
            counts[kind] = n
        counts["attempts"] = sum(counts.values())
        results = db.execute("SELECT count(*) FROM results").fetchone()[0]
        require(isinstance(meta.get("identity"), str) and meta.get("limits") == LIMITS,
                "Unknown ledger identity/limits")
        stages = [p for p in ("manifest", "service", "preflight", "selection") if p in meta]
        return dict(identity=meta["identity"], counts=counts, results=results, stages=stages,
                    data_binding=meta.get("data_binding"), migration=meta.get("migration"),
                    zero_verified=counts["attempts"] == 0 and results == 0)


def check_slot(state, stable, prepared, runs_root):
    binding = state["data_binding"]
    if binding is not None:
        require(binding == digest(stable), "V5 data/source/review changed; cannot resume")
        return "V5_RESUME"
    require(state["identity"] == LEGACY_IDENTITY, "Unrecognized V4 migration origin")
    require(state["zero_verified"], "V4 migration requires zero attempts AND zero results")
    previous = Path(runs_root) / LEGACY_IDENTITY
    for name in ("public", "splits", "prefix_raw", "snapshots", "duplicate_audit"):
        require(unseal(read_json(previous / f"{name}.json")) == prepared[name],
                f"Legacy source/artifact provenance mismatch: {name}")
    return "V4_ZERO_MIGRATION"


def audit_data(cfg, ledger_path, runs_root):
    """Write one evidence entry even on failure; stable qualification has no clock/counts."""
    evidence = dict(protocol="V5", observed_at=time.time(), qualified=False,
                    next_step="STOP: inspect this evidence; do not start generation")
    prepared, stable, state = {}, None, None
    try:
        data_hash = file_hash(cfg["data"])
        public = load_public_inputs(cfg["data"])
        old, new, splits = split_public(public)
        audit = duplicate_audit(public[:new], splits)
        lookup = {r["qid"]: r for r in public}
        evidence.update(original_audit=audit, candidate_public=[
            {side: lookup[c[side][1]] for side in ("left", "right")}
            for c in audit["candidates"]])
        review = read_json(REVIEW_PATH)
        qualification = qualify(public, splits, audit, review)
        evidence["data_qualification"] = qualification
        prefix, raw_prefix, source_hash = validate_source(cfg["source"], public, new)
        snapshots = build_snapshots(prefix, old, new)
        require(file_hash(cfg["data"]) == data_hash and file_hash(cfg["source"]) == source_hash,
                "Inputs changed during data audit")
        prepared = dict(public=public, splits=splits, prefix_raw=raw_prefix,
                        snapshots=snapshots, duplicate_audit=audit)
        stable = dict(parameters=PARAMETERS, qualification=qualification,
                      source_sha256=source_hash, data_sha256=data_hash,
                      prefix_sha256=digest(raw_prefix), snapshots_sha256=digest(snapshots))
        evidence["stable"] = stable
        state = readonly_ledger(ledger_path)
        evidence["ledger"] = state
        evidence["mode"] = check_slot(state, stable, prepared, runs_root)
        evidence.update(qualified=True, next_step="prepare (no generation); then inspect intervention.json")
    except (Invalid, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    path = Path(runs_root) / "audit-data.json"
    atomic_json(path, seal(evidence), immutable=False)
    print(f"Data audit: {'PASS' if evidence['qualified'] else 'STOP'}; evidence: {path}", flush=True)
    require(evidence["qualified"], f"{evidence.get('error')}; evidence: {path}")
    return stable, prepared, state
