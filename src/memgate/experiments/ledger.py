"""Single-writer, identity-level durable attempts, responses and accepted rows."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import http.client
import json
import socket
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import Invalid, LIMITS, canonical, digest, require


class Transport:
    """No SDK and no implicit retry. Root URLs, including /tokenize, are explicit."""
    def __init__(self, base_url, timeout=600):
        self.base_url = base_url.rstrip("/")
        require(not self.base_url.endswith("/v1"), "Use the server root URL, without /v1")
        self.timeout = timeout

    def request(self, path, payload=None):
        data = canonical(payload) if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return self.read_response(response, response.status)
        except urllib.error.HTTPError as exc:
            with exc:
                return self.read_response(exc, exc.code)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            return {"status": None, "body": "", "transport_error": True,
                    "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def read_response(response, status):
        try:
            data = response.read()
        except (http.client.HTTPException, OSError) as exc:
            partial = getattr(exc, "partial", b"")
            return dict(status=status, body="", transport_error=True,
                        error=f"{type(exc).__name__}: {exc}",
                        partial_base64=base64.b64encode(partial).decode("ascii"),
                        partial_sha256=hashlib.sha256(partial).hexdigest(),
                        partial_bytes=len(partial), expected_remaining=getattr(exc, "expected", None))
        try:
            return dict(status=status, body=data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            # A complete invalid body is durable and never retried, including after restart.
            return dict(status=status, body="", invalid_body=True,
                        body_base64=base64.b64encode(data).decode("ascii"),
                        error=f"{type(exc).__name__}: {exc}")

    def json(self, path, payload=None):
        raw = self.request(path, payload)
        require(complete_success(raw), f"Metadata/tokenization failed at {path}: {raw}")
        return json.loads(raw["body"])


def complete_success(raw):
    return raw["status"] == 200 and not raw.get("transport_error") and not raw.get("invalid_body")


def retryable(raw):
    if raw.get("invalid_body"):
        return False
    status = raw["status"]
    return (status is None or status == 429 or status >= 500
            or (status == 200 and raw.get("transport_error", False)))


def payload(ids, seed):
    require(ids and all(type(i) is int and i >= 0 for i in ids), "Prompt must be integer token IDs")
    require(len(ids) <= LIMITS["input"], "Input exceeds 5888 before dispatch")
    return dict(model="Qwen/Qwen3-8B", prompt=ids, max_tokens=2048,
                temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                seed=seed, n=1, best_of=1, stream=False, echo=False,
                stop=None, stop_token_ids=[], ignore_eos=False,
                skip_special_tokens=True, spaces_between_special_tokens=True,
                min_tokens=0, include_stop_str_in_output=False,
                add_special_tokens=False, use_beam_search=False)


def validate_response(raw, request):
    require(complete_success(raw), "No complete successful HTTP response")
    try:
        body = json.loads(raw["body"])
        require(isinstance(body, dict), "Completion response must be an object")
        require(body.get("model") == request["model"], "Response served model differs")
        choices, usage = body["choices"], body["usage"]
        require(len(choices) == 1 and isinstance(choices[0]["text"], str), "Invalid completion choices")
        require(choices[0]["finish_reason"] in ("stop", "length"), "Unexpected finish reason")
        require(usage["prompt_tokens"] == len(request["prompt"]), "Backend prompt token count mismatch")
        require(type(usage["completion_tokens"]) is int and
                0 <= usage["completion_tokens"] <= request["max_tokens"], "Output token budget mismatch")
        require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
                "Invalid total token usage")
        return {"text": choices[0]["text"], "finish_reason": choices[0]["finish_reason"],
                "usage": usage}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise Invalid(f"Malformed successful response: {exc}") from exc


class Ledger:
    """Every ambiguous reservation counts; first valid persisted response wins.

    Hold a process lock through dispatch and commit. SQLite FULL synchronous
    transactions close the reserve/response/result crash windows. A copied run
    directory cannot reset this separate ledger in runs/.e2-ledgers.
    """
    def __init__(self, path, identity, limits=None, *, expected_previous=None, data_binding=None):
        self.path = Path(path)
        if expected_previous is not None:
            require(self.path.is_file(), "Existing budget slot disappeared before migration")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise Invalid("Another E2 writer holds this experiment identity") from exc
        if expected_previous is not None and not self.path.is_file():
            self.lock.close()
            raise Invalid("Existing budget slot disappeared under migration lock")
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY, key TEXT NOT NULL, kind TEXT NOT NULL,
              payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
              reserved REAL NOT NULL, raw TEXT, raw_hash TEXT);
            CREATE TABLE IF NOT EXISTS results (
              key TEXT PRIMARY KEY, value TEXT NOT NULL, hash TEXT NOT NULL);
        """)
        self.limits = {**LIMITS, **(limits or {})}
        try:
            previous_identity = self.meta("identity")
            if expected_previous is not None:
                require(previous_identity == expected_previous, "Ledger identity changed since audit")
            if previous_identity is not None and previous_identity != identity:
                require(self.counts()["attempts"] == 0 and not self.results(),
                        "Experiment identity changed after a submitted attempt; revision/review required")
                # Offline-only failed preparation may be corrected before any generation.
                # Old artifacts stay in their content-addressed directory.
                with self.db:
                    provenance = dict(from_identity=previous_identity, to_identity=identity,
                                      previous_meta={r["key"]: json.loads(r["value"])
                                                     for r in self.db.execute("SELECT * FROM meta")},
                                      attempts=0, results=0)
                    self.db.execute("DELETE FROM meta")
                    values = dict(migration=provenance, identity=identity, limits=self.limits)
                    if data_binding is not None:
                        values["data_binding"] = data_binding
                    self.db.executemany("INSERT INTO meta VALUES (?,?)",
                                        [(k, canonical(v).decode()) for k, v in values.items()])
            self.bind("identity", identity)
            self.bind("limits", self.limits)
            if data_binding is not None:
                self.bind("data_binding", data_binding)
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def bind(self, key, value):
        with self.db:
            existing = self.meta(key)
            require(existing is None or existing == value, f"Ledger binding changed: {key}")
            if existing is None:
                self.db.execute("INSERT INTO meta VALUES (?,?)", (key, canonical(value).decode()))

    def counts(self):
        result = {k: 0 for k in ("logical", "probes", "retries")}
        for row in self.db.execute("SELECT kind, count(*) AS n FROM attempts GROUP BY kind"):
            result[row["kind"]] = row["n"]
        result["attempts"] = sum(result.values())
        return result

    def attempts(self, key):
        rows = list(self.db.execute("SELECT * FROM attempts WHERE key=? ORDER BY id", (key,)))
        for row in rows:
            require(digest(json.loads(row["payload"])) == row["payload_hash"], "Attempt payload changed")
            if row["raw"] is not None:
                require(digest(json.loads(row["raw"])) == row["raw_hash"], "Raw response changed")
        return rows

    def reserve(self, key, request, probe=False):
        with self.db:
            prior = self.attempts(key)
            require(all(r["payload_hash"] == digest(request) for r in prior), "Retry payload/seed changed")
            require(len(prior) < self.limits["per_request"], f"Per-request attempts exhausted: {key}")
            require(not (probe and prior), "A probe may be submitted only once; no third probe on restart")
            kind = "retries" if prior else ("probes" if probe else "logical")
            counts = self.counts()
            require(counts[kind] < self.limits[kind] and counts["attempts"] < self.limits["attempts"],
                    f"Global attempt budget exhausted: {counts}")
            cursor = self.db.execute(
                "INSERT INTO attempts(key,kind,payload,payload_hash,reserved) VALUES (?,?,?,?,?)",
                (key, kind, canonical(request).decode(), digest(request), time.time()))
            return cursor.lastrowid

    def save_raw(self, attempt, raw):
        with self.db:
            cursor = self.db.execute("UPDATE attempts SET raw=?,raw_hash=? WHERE id=? AND raw IS NULL",
                                     (canonical(raw).decode(), digest(raw), attempt))
            require(cursor.rowcount == 1, "Attempt raw response already written")

    def result(self, key):
        row = self.db.execute("SELECT * FROM results WHERE key=?", (key,)).fetchone()
        if row:
            value = json.loads(row["value"])
            require(digest(value) == row["hash"], "Accepted result checksum mismatch")
            return value
        return None

    def results(self):
        return {r["key"]: self.result(r["key"]) for r in self.db.execute("SELECT key FROM results")}

    def commit(self, key, value):
        with self.db:
            previous = self.result(key)
            require(previous is None or previous == value, "Cannot replace an accepted result")
            if previous is None:
                self.db.execute("INSERT INTO results VALUES (?,?,?)",
                                (key, canonical(value).decode(), digest(value)))

    def complete(self, key, request, transport, transform, binding, probe=False):
        existing = self.result(key)
        if existing is not None:
            self.verify_result(key, request, transform, binding)
            return existing
        while True:
            prior = self.attempts(key)
            for row in prior:
                require(row["payload_hash"] == digest(request), "Request changed during resume")
                if row["raw"] is None:
                    continue  # Unknown dispatch counts and consumes a retry on recovery.
                raw = json.loads(row["raw"])
                if complete_success(raw):
                    parsed = validate_response(raw, request)
                    value = dict(binding=binding, attempt_id=row["id"],
                                 request_hash=digest(request), raw_hash=digest(raw),
                                 **parsed, **transform(parsed["text"]))
                    self.commit(key, value)
                    return value
                require(retryable(raw),
                        f"Non-retryable HTTP {raw['status']}: {raw.get('error', raw['body'])[:500]}")
            attempt = self.reserve(key, request, probe)
            raw = transport.request("/v1/completions", request)
            self.save_raw(attempt, raw)
            # Loop validates or retries from persisted state. No correctness-based selection.

    def verify_result(self, key, request, transform, binding):
        value = self.result(key)
        require(value is not None, f"Missing result: {key}")
        rows = self.attempts(key)
        accepted = next((r for r in rows if r["id"] == value["attempt_id"]), None)
        require(accepted is not None and accepted["raw"] is not None, "Missing accepted raw response")
        raw = json.loads(accepted["raw"])
        parsed = validate_response(raw, request)
        expected = dict(binding=binding, attempt_id=accepted["id"], request_hash=digest(request),
                        raw_hash=digest(raw), **parsed, **transform(parsed["text"]))
        require(expected == value and accepted["payload_hash"] == digest(request),
                "Result differs from source response, scorer or input binding")
        earlier = [r for r in rows if r["id"] < accepted["id"] and r["raw"] is not None]
        require(not any(complete_success(json.loads(r["raw"])) for r in earlier),
                "Did not accept the first successful response")
        return value
