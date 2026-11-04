"""Offline acceptance tests: real HTTP client, isolated ledgers, no model calls."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memgate.experiments.common import Invalid, canonical, digest, file_hash, seal
from memgate.experiments.ledger import Transport
from memgate.experiments.official import Official
from memgate.stage_a import (
    DiagnosticLedger, SLOT, analyze, audit_snapshot, binding, check_service, export_run,
    load_package, metadata, preflight, read_sealed, readonly_snapshot, requests, run,
    snapshot, validate_package,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    if not (ROOT / ".cache/stage-a-curated-v1/preparation-v2.json.gz").is_file():
        pytest.skip("Frozen local research assets unavailable")
    spec = importlib.util.spec_from_file_location("build_stage_a", ROOT / "scripts/build_stage_a_execution.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = tmp_path_factory.mktemp("stage-a-package") / "bundle"
    builder.build(output)
    data = load_package(output, file_hash(output / "execution.json.gz"), output / "stage-a.pyz")
    return data, output, Official(output / "official").extract_letter


def good_response(request, text="The answer is (A)."):
    return dict(status=200, body=json.dumps(dict(
        model=request["model"], choices=[dict(text=text, finish_reason="stop")],
        usage=dict(prompt_tokens=len(request["prompt"]), completion_tokens=9,
                   total_tokens=len(request["prompt"]) + 9))))


class StubTransport:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), []

    def request(self, path, request):
        self.calls.append(copy.deepcopy(request))
        return self.replies.pop(0) if self.replies else good_response(request)


def fake_identity(data):
    return dict(
        actual_max_model_len=8192, served_models=["Qwen/Qwen3-8B"],
        model_files=data["model_files"], dtype="torch.bfloat16", tensor_parallel_size=1,
        max_num_seqs=1, max_num_batched_tokens=8192, enforce_eager=True,
        generation_config="vllm", gpu=[dict(name="TEST_ONLY_FAKE_GPU")],
        runtime_versions=dict(vllm="0.8.5"), launcher_sha256=data["launcher_sha256"])


@pytest.fixture
def fake_http(package):
    data = package[0]
    prompts = data["sentinels"] + [p for pair in data["preparation"]["pairs"] for p in pair["prompts"].values()]
    by_messages = {digest(p["messages"]): p for p in prompts}
    by_text = {p["text"]: p for p in prompts}
    by_request = {digest(e["request"]): e for e in requests(data).values()}
    state = dict(calls=[], metadata_calls=[], identity=fake_identity(data), bad_tokens=False, replies=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, body):
            status = 200
            if self.path == "/v1/completions":
                state["calls"].append(body)
                e = by_request[digest(body)]
                text = "I cannot determine an answer."
                if not e["probe"] and e["item"]["arm"] == "11":
                    text = "The answer is (" + data["scoring_gold"][e["item"]["qid"]] + ")."
                raw = state["replies"].pop(0) if state["replies"] else good_response(body, text)
                status, output = raw["status"], raw["body"]
            else:
                state["metadata_calls"].append(self.path)
                if self.path == "/e2/identity":
                    result = state["identity"]
                elif self.path == "/v1/models":
                    result = dict(data=[dict(id="Qwen/Qwen3-8B")])
                elif self.path == "/e2/chat_ids":
                    p = by_messages[digest(body["messages"])]
                    result = dict(ids=p["ids"], text=p["text"])
                elif self.path == "/tokenize":
                    result = dict(tokens=by_text[body["prompt"]]["ids"], max_model_len=8192)
                    if state["bad_tokens"]:
                        result["tokens"] = [0]
                else:
                    raise AssertionError(self.path)
                output = json.dumps(result)
            encoded = output.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Transport(f"http://127.0.0.1:{server.server_port}", timeout=3), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_real_http_full_smoke_resume_and_local_replay(package, fake_http, tmp_path):
    data, directory, extractor = package
    transport, state = fake_http
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        report = run(data, ledger, transport, extractor, tmp_path / "exports")
        assert report["attempt_counts"] == dict(logical=72, probes=2, retries=0, attempts=74)
        assert report["accepted_formal"] == 72
        assert report["overall"]["accuracy"] == {"00": 0, "10": 0, "01": 0, "11": 1}
        assert report["overall"]["contrasts"]["interaction"] == 1
        assert report["cost"]["validated_response_count"] == 74
        assert len(state["calls"]) == 74
        assert len(state["metadata_calls"]) == 78  # model metadata + 38 * two token endpoints
        assert ledger.meta("smoke") is not None
    # A fresh process/session identity reuses every accepted response, including probes.
    with DiagnosticLedger(tmp_path, data) as ledger:
        resumed = run(data, ledger, transport, extractor, tmp_path / "exports")
        assert resumed == report and len(state["calls"]) == 74
    readonly = readonly_snapshot(tmp_path, data)
    assert analyze(data, readonly, extractor) == report
    exported = next((tmp_path / "exports").glob("export-*.json.gz"))
    cli = subprocess.run([sys.executable, str(directory / "stage-a.pyz"), "analyze",
                          "--export-file", str(exported), "--output", str(tmp_path / "replay")],
                         capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert (tmp_path / "replay" / exported.name).read_bytes() == exported.read_bytes()


@pytest.mark.parametrize("change", ["model", "context", "tokens"])
def test_bad_service_prevents_all_generation(package, fake_http, tmp_path, change):
    data = package[0]
    transport, state = fake_http
    if change == "model":
        state["identity"] = dict(state["identity"], model_files={})
    elif change == "context":
        state["identity"]["actual_max_model_len"] = 4096
    else:
        state["bad_tokens"] = True
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        with pytest.raises(Invalid):
            preflight(data, ledger, transport)
        assert ledger.counts()["attempts"] == 0 and not state["calls"]


def test_fixed_shared_reserve_and_global_cap(package, tmp_path):
    data = package[0]
    allowed = requests(data)
    formal = [(k, e) for k, e in allowed.items() if not e["probe"]]
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        for k, e in allowed.items():
            ledger.reserve(k, e["request"], probe=e["probe"])
        for k, e in formal[:6]:
            ledger.reserve(k, e["request"])
        assert ledger.counts() == dict(logical=72, probes=2, retries=6, attempts=80)
        with pytest.raises(Invalid, match="Shared"):
            ledger.reserve(formal[6][0], formal[6][1]["request"])
    # Shared cap also blocks retries long before the total cap is reached.
    with DiagnosticLedger(tmp_path / "separate-test", data, initialize=True) as ledger:
        for i in range(2):
            ledger.reserve(f"probe:{i}", allowed[f"probe:{i}"]["request"], probe=True)
        for k, e in formal[:6]:
            ledger.reserve(k, e["request"])
            ledger.reserve(k, e["request"])
        ledger.reserve(formal[6][0], formal[6][1]["request"])
        with pytest.raises(Invalid, match="Shared"):
            ledger.reserve(formal[6][0], formal[6][1]["request"])
        assert ledger.counts()["attempts"] == 15


def test_foreign_seed_and_probe_relabel_are_rejected(package, tmp_path):
    data = package[0]
    k, e = next(iter(requests(data).items()))
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        for key, request, probe in (
            ("foreign", e["request"], False),
            (k, dict(e["request"], seed=e["request"]["seed"] + 1), False),
            (k, e["request"], True),
        ):
            with pytest.raises(Invalid, match="allowlist"):
                ledger.reserve(key, request, probe)
        assert ledger.counts()["attempts"] == 0


@pytest.mark.parametrize("failure", [
    dict(status=500, body="server error"),
    dict(status=None, body="", transport_error=True, error="lost connection"),
])
def test_probes_cannot_retry_even_after_restart(package, tmp_path, failure):
    data = package[0]
    entry = requests(data)["probe:0"]
    transport = StubTransport([failure])
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        with pytest.raises(Invalid, match="only once"):
            ledger.complete("probe:0", entry["request"], transport, lambda t: {}, "test", probe=True)
    with DiagnosticLedger(tmp_path, data) as ledger:
        with pytest.raises(Invalid, match="only once"):
            ledger.complete("probe:0", entry["request"], transport, lambda t: {}, "test", probe=True)
        assert ledger.counts()["attempts"] == 1 and len(transport.calls) == 1


@pytest.mark.parametrize("raw", [
    dict(status=400, body="bad request"),
    dict(status=200, body='{"malformed":true}'),
    dict(status=200, body="", invalid_body=True, body_base64="/w=="),
])
def test_invalid_completed_response_never_retried(package, tmp_path, raw):
    data = package[0]
    key, entry = next(iter(requests(data).items()))
    transport = StubTransport([raw])
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        with pytest.raises(Invalid):
            ledger.complete(key, entry["request"], transport, lambda t: {}, "test")
        assert len(transport.calls) == 1 and ledger.counts()["attempts"] == 1


@pytest.mark.parametrize("save_raw", [False, True])
def test_process_death_charges_unknown_or_reuses_raw(package, tmp_path, save_raw):
    data, directory, _ = package
    key, entry = next(iter(requests(data).items()))
    child = """
import os, sys
from memgate.stage_a import DiagnosticLedger, read_sealed, requests
data = read_sealed(sys.argv[1])
ledger = DiagnosticLedger(sys.argv[2], data, initialize=True)
key, entry = next(iter(requests(data).items()))
attempt = ledger.reserve(key, entry["request"])
if sys.argv[3] == "yes":
    import json
    ledger.save_raw(attempt, json.loads(sys.argv[4]))
os._exit(19)
"""
    result = subprocess.run([sys.executable, "-c", child, str(directory / "execution.json.gz"),
                             str(tmp_path), "yes" if save_raw else "no",
                             json.dumps(good_response(entry["request"]))],
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, timeout=20)
    assert result.returncode == 19
    # Inspect live WAL via a read-only connection before any writable recovery.
    state = readonly_snapshot(tmp_path, data)
    assert len(state["attempts"]) == 1
    assert (state["attempts"][0]["raw"] is not None) == save_raw
    transport = StubTransport()
    with DiagnosticLedger(tmp_path, data) as ledger:
        ledger.complete(key, entry["request"], transport, lambda t: {}, "test")
        assert len(transport.calls) == (0 if save_raw else 1)
        assert ledger.counts()["attempts"] == (1 if save_raw else 2)


def test_missing_ledger_and_lost_marker_do_not_reinitialize(package, tmp_path):
    data = package[0]
    with DiagnosticLedger(tmp_path, data, initialize=True):
        pass
    path = tmp_path / "runs/.stage-a-ledgers" / f"{SLOT}.sqlite"
    saved = path.read_bytes()
    path.unlink()
    with pytest.raises(Invalid, match="disappeared"):
        DiagnosticLedger(tmp_path, data, initialize=True)
    path.write_bytes(saved)
    (tmp_path / ".stage-a-authorizations" / f"{SLOT}.json").unlink()
    with pytest.raises(Invalid, match="Orphaned"):
        DiagnosticLedger(tmp_path, data, initialize=True)


def test_writer_lock_and_empty_sqlite_replacement(package, tmp_path):
    data = package[0]
    with DiagnosticLedger(tmp_path, data, initialize=True):
        with pytest.raises(BlockingIOError):
            DiagnosticLedger(tmp_path, data)
    path = tmp_path / "runs/.stage-a-ledgers" / f"{SLOT}.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM meta")
    with pytest.raises(Invalid, match="empty"):
        DiagnosticLedger(tmp_path, data, initialize=True)


def test_input_tampering_and_gold_never_changes_requests(package, tmp_path):
    data, directory, _ = package
    altered = copy.deepcopy(data)
    altered["scoring_gold"] = {q: "Z" for q in data["scoring_gold"]}
    assert requests(altered) == requests(data)
    altered["limits"]["attempts"] += 1
    with pytest.raises(Invalid, match="budget"):
        validate_package(altered)
    tampered = tmp_path / "execution.json.gz"
    tampered.write_bytes(gzip.compress(canonical(seal(altered))))
    with pytest.raises(Invalid, match="tool authorization"):
        load_package(tmp_path, file_hash(directory / "execution.json.gz"))


def test_partial_analysis_has_no_effect_estimate_and_detects_tamper(package, fake_http, tmp_path):
    data, _, extractor = package
    transport, _ = fake_http
    with DiagnosticLedger(tmp_path, data, initialize=True) as ledger:
        preflight(data, ledger, transport)
        before = snapshot(ledger)
        report = analyze(data, before, extractor)
        assert report["overall"] is None and report["accepted_formal"] == 0
        altered = copy.deepcopy(before)
        altered["attempts"][0]["raw_hash"] = "broken"
        with pytest.raises(Invalid, match="Raw response"):
            analyze(data, altered, extractor)
        altered = copy.deepcopy(before)
        value = json.loads(altered["results"][0]["value"])
        value["text"] = "altered"
        altered["results"][0].update(value=canonical(value).decode(), hash=digest(value))
        with pytest.raises(Invalid, match="differs from response"):
            audit_snapshot(data, altered, extractor)
