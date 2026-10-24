"""V5 qualification, migration, incomplete transport and control regression tests."""
import base64
import copy
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memgate.experiments import qualification, study
from memgate.experiments.common import (LEGACY_IDENTITY, REVIEW_PATH, ROOT, SOFTWARE_VERSION,
                                       Invalid, digest, file_hash, read_json, unseal)
from memgate.experiments.ledger import Ledger, Transport, payload
from memgate.experiments.source import (build_snapshots, duplicate_audit, load_public_inputs,
                                       split_public, validate_source)
from test_e2_components import TinyTokenizer, public_rows, source_rows, tiny_snapshots, write_source
from test_e2_runtime import StubTransport, response


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    """Synthetic public/source data, explicitly approved ONLY in this test process."""
    public = public_rows()
    _, new, splits = split_public(public)
    for i, row in enumerate(splits["test"][:3]):
        public[i]["question"] = row["question"] + ("?" if i == 1 else "")
        public[i]["options"] = list(row["options"])
    audit = duplicate_audit(public[:new], splits)
    assert len(audit["candidates"]) == 3
    lookup = {r["qid"]: r for r in public}
    review = copy.deepcopy(read_json(REVIEW_PATH))
    review.update(public_sha256=digest(public), audit_sha256=digest(audit))
    review["decisions"] = [
        dict(left=c["left"], right=c["right"],
             left_public_sha256=digest(lookup[c["left"][1]]),
             right_public_sha256=digest(lookup[c["right"][1]]),
             decision="distinct" if c["left"][1] == "1" else "duplicate",
             reason="Synthetic test-only judgment") for c in audit["candidates"]]
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(qualification, "REVIEW_PATH", review_path)
    monkeypatch.setattr(qualification, "APPROVED_REVIEW", digest(review))
    data, source = tmp_path / "data.jsonl", tmp_path / "source.jsonl"
    data.write_text("".join(json.dumps(dict(question_id=r["qid"], category="engineering",
                                          answer="A", **r)) + "\n" for r in public))
    write_source(source, source_rows(public))
    prefix, raw, _ = validate_source(source, public, new)
    prepared = dict(public=public, splits=splits, prefix_raw=raw, duplicate_audit=audit,
                    snapshots=build_snapshots(prefix, int(.2*len(public)), new))
    runs = tmp_path / "runs/e2_component"
    for name, value in prepared.items():
        study.write_artifact(runs / LEGACY_IDENTITY, name + ".json", value)
    slot = tmp_path / "runs/.e2-ledgers/component-v4.sqlite"
    with Ledger(slot, LEGACY_IDENTITY):
        pass
    cfg = dict(data=data, source=source, seqmem_root=ROOT / "third_party/SeqMem-Eval",
               model_path=tmp_path / "absent-model", embedding_path=tmp_path / "absent-embedding",
               base_url="http://127.0.0.1:1")
    return cfg, slot, runs, prepared, review


def test_a01_a03_a07_audit_is_readonly_and_stable_across_resume(reviewed, monkeypatch):
    cfg, slot, runs, prepared, review = reviewed
    def forbidden(*args, **kwargs):
        raise AssertionError("audit must not load models or write a ledger")
    monkeypatch.setattr(study, "input_fingerprint", forbidden)
    monkeypatch.setattr(study, "load_tokenizer", forbidden)
    monkeypatch.setattr(study, "make_embeddings", forbidden)
    monkeypatch.setattr(Transport, "request", forbidden)
    monkeypatch.setattr(study, "ROOT", runs.parent.parent)
    monkeypatch.setattr(study, "configuration", lambda path: cfg)
    initial = file_hash(slot)
    assert study.main(["audit-data"]) == 0
    stable, _, state = qualification.audit_data(cfg, slot, runs)
    assert file_hash(slot) == initial
    assert state["zero_verified"]
    q = stable["qualification"]
    assert q["data_qualified_counts"] == {"dev": 64, "test": 94}
    assert q["original_counts"] == {"dev": 64, "test": 96}
    assert prepared["splits"]["test"][1] in q["evaluation_splits"]["test"]
    assert len(q["exclusions"]["test"]) == 2 and q["exclusions"]["dev"] == []
    with Ledger(slot, "v5-id", expected_previous=LEGACY_IDENTITY, data_binding=digest(stable)) as ledger:
        ledger.reserve("unknown", payload([1], 10))
        migration = ledger.meta("migration")
        assert migration["from_identity"] == LEGACY_IDENTITY and migration["attempts"] == 0
        # Committed WAL rows must be visible to the separate read-only connection.
        again, _, state = qualification.audit_data(cfg, slot, runs)
        assert again == stable and state["counts"]["attempts"] == 1
        with pytest.raises(Invalid, match="Another"):
            Ledger(slot, "another", expected_previous="v5-id")
    with Ledger(slot, "v5-id", expected_previous="v5-id", data_binding=digest(stable)) as ledger:
        assert ledger.counts()["attempts"] == 1
    with pytest.raises(Invalid, match="identity"):
        Ledger(slot, "changed-code", expected_previous="v5-id", data_binding=digest(stable))


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "duplicate", "unresolved", "decision", "phase", "qid",
    "review_hash", "question", "option_order", "public_order", "audit", "new_conflict", "split",
])
def test_a02_all_review_bindings_fail_closed(reviewed, mutation):
    _, _, _, p, review = reviewed
    public, splits, audit = p["public"], p["splits"], p["duplicate_audit"]
    if mutation == "missing":
        review["decisions"].pop()
    elif mutation == "extra":
        review["decisions"].append(copy.deepcopy(review["decisions"][0]))
    elif mutation == "duplicate":
        review["decisions"][1] = copy.deepcopy(review["decisions"][0])
    elif mutation in ("unresolved", "decision"):
        review["decisions"][0]["decision"] = "unresolved" if mutation == "unresolved" else "distinct"
    elif mutation in ("phase", "qid"):
        review["decisions"][0]["left"][0 if mutation == "phase" else 1] = "changed"
    elif mutation == "review_hash":
        review["decisions"][0]["left_public_sha256"] = "wrong"
    elif mutation == "question":
        public[5]["question"] += "changed"
    elif mutation == "option_order":
        public[0]["options"].reverse()
    elif mutation == "public_order":
        public.reverse()
    elif mutation == "audit":
        audit["comparisons"] += 1
    elif mutation == "new_conflict":
        audit["candidates"].append(dict(left=["dev", "3"], right=["test", "4"]))
    else:
        splits["dev"].reverse()
    with pytest.raises(Invalid):
        qualification.qualify(public, splits, audit, review)


@pytest.mark.parametrize("mutation", ["missing_ledger", "corrupt_ledger", "source", "chain", "review"])
def test_a04_a05_failure_evidence_before_models(reviewed, mutation):
    cfg, slot, runs, _, _ = reviewed
    if mutation == "missing_ledger":
        slot.unlink()
    elif mutation == "corrupt_ledger":
        slot.write_text("invalid sqlite")
    elif mutation in ("source", "chain"):
        rows = [json.loads(line) for line in cfg["source"].read_text().splitlines()]
        if mutation == "source":
            rows.pop()
        else:
            rows[2]["memory_meta"]["insights_before"] = "broken"
        write_source(cfg["source"], rows)
    else:
        qualification.REVIEW_PATH.write_text("{}")
    with pytest.raises(Invalid):
        qualification.audit_data(cfg, slot, runs)
    evidence = unseal(read_json(runs / "audit-data.json"))
    assert not evidence["qualified"] and evidence["error"]
    assert not cfg["model_path"].exists()
    if mutation == "missing_ledger":
        assert not slot.exists()


@pytest.mark.parametrize("kind", ["probe", "unknown", "result"])
def test_a06_migration_rechecks_zero_under_lock(reviewed, kind):
    cfg, slot, runs, _, _ = reviewed
    qualification.audit_data(cfg, slot, runs)  # Previously observed zero is insufficient.
    with Ledger(slot, LEGACY_IDENTITY) as ledger:
        if kind == "result":
            ledger.commit("orphan", {"record": "present"})
        else:
            ledger.reserve("pending", payload([1], 10), probe=kind == "probe")
    with pytest.raises(Invalid, match="zero attempts"):
        qualification.audit_data(cfg, slot, runs)
    with pytest.raises(Invalid, match="submitted attempt"):
        Ledger(slot, "new", expected_previous=LEGACY_IDENTITY, data_binding="new")
    assert qualification.readonly_ledger(slot)["identity"] == LEGACY_IDENTITY


def test_a06_unknown_origin_and_missing_provenance(reviewed):
    cfg, slot, runs, _, _ = reviewed
    (runs / LEGACY_IDENTITY / "prefix_raw.json").unlink()
    with pytest.raises(Invalid):
        qualification.audit_data(cfg, slot, runs)
    with Ledger(slot, "unknown-origin"):
        pass
    with pytest.raises(Invalid, match="Unrecognized"):
        qualification.audit_data(cfg, slot, runs)


@pytest.fixture
def interrupted_http():
    state = dict(replies=[], calls=[])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(request)
            directive = state["replies"].pop(0) if state["replies"] else None
            status, data, framing = directive or (200, response(request)["body"].encode(), "complete")
            self.send_response(status)
            if framing == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(1000 if framing == "short" else len(data)))
            self.end_headers()
            if framing == "chunked":
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            else:
                self.wfile.write(data)
            self.close_connection = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Transport(f"http://127.0.0.1:{server.server_port}", timeout=2), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("framing", ["short", "chunked"])
def test_a08_partial200_is_durable_bounded_same_seed_retry(tmp_path, interrupted_http, framing):
    transport, state = interrupted_http
    partial = b'{"usage":\xff}'  # 11 bytes, not valid UTF-8; retain them losslessly.
    state["replies"] = [(200, partial, framing)]
    request = payload([1, 2], 713)
    with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
        value = ledger.complete("q", request, transport, lambda t: {}, "binding")
        raw = json.loads(ledger.attempts("q")[0]["raw"])
        assert raw["status"] == 200 and raw["transport_error"]
        assert base64.b64decode(raw["partial_base64"]) == partial
        assert raw["partial_bytes"] == 11
        if framing == "short":
            assert raw["expected_remaining"] == 989
        assert ledger.counts() == dict(logical=1, probes=0, retries=1, attempts=2)
        assert state["calls"] == [request, request]
        assert ledger.verify_result("q", request, lambda t: {}, "binding") == value


@pytest.mark.parametrize("status,data,framing", [
    (400, b"context rejected", "short"), (400, b"\xff", "complete"),
    (200, b"\xff", "complete"), (200, b"{broken", "complete"),
])
def test_a08_a09_known_rejections_never_retry_on_restart(tmp_path, interrupted_http, status, data, framing):
    transport, state = interrupted_http
    state["replies"] = [(status, data, framing)]
    for _ in range(2):
        with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
            with pytest.raises(Invalid):
                ledger.complete("q", payload([1], 1), transport, lambda t: {}, "b")
            assert ledger.counts()["attempts"] == 1 and ledger.result("q") is None
            assert json.loads(ledger.attempts("q")[0]["raw"])["status"] == status
    assert len(state["calls"]) == 1


def test_a08_probe_partial_complete_json_still_not_accepted(tmp_path, interrupted_http):
    transport, state = interrupted_http
    request = payload([1], 2)
    # Syntactically complete success JSON is still incomplete HTTP, not a result.
    state["replies"] = [(200, response(request)["body"].encode(), "short")]
    for _ in range(2):
        with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
            with pytest.raises(Invalid, match="probe"):
                ledger.complete("probe:0", request, transport, lambda t: {}, "b", probe=True)
            assert ledger.counts()["attempts"] == 1 and not ledger.results()
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("arm", ["OO", "NN", "ZE"])
def test_a10_each_main_control_full_ids_and_partial_identity(arm):
    def q(ids):
        return dict(change=dict(packed_rules=True, packed_experiences=True),
                    prompts={a: dict(ids=x) for a, x in ids.items()})
    questions = {"1": q(dict(ON=[1], OO=[2], NN=[3], ZE=[4]))}
    assert study.intervention_summary(questions, ["1"])["main_controls_valid"]  # same lengths
    questions["1"]["prompts"][arm]["ids"] = [1]
    assert not study.intervention_summary(questions, ["1"])["main_controls_valid"]
    questions["2"] = q(dict(ON=[1], OO=[2], NN=[3], ZE=[4]))
    summary = study.intervention_summary(questions, ["1", "2"])
    assert summary["main_controls_valid"] and summary["eligible_test"] == 2
    assert not study.intervention_summary({}, [])["main_controls_valid"]


def test_a10_real_builder_old_rule_dropped():
    snap = tiny_snapshots()
    snap["old"]["rules"] = "1. " + "x"*2000
    result = study.prepare_question(study.Official(ROOT / "third_party/SeqMem-Eval"), TinyTokenizer(),
                                    dict(qid="9", question="Evaluate", options=["a", "b"]),
                                    snap, {"old": [[1, 0]], "new": [[1, 0], [0, 1]]}, [0, 1])
    assert result["eligible"] and result["rules"]["old"]["kept"] == 0
    summary = study.intervention_summary({"9": result}, ["9"])
    assert summary["jointly_changed"] == 1 and not summary["main_controls_valid"]
    assert summary["main_controls"]["C3_ON_ZE"]["different"] == 0


@pytest.mark.parametrize("mode", ["empty_old_rules", "all_context_excluded", "one_context_excluded"])
def test_a03_a10_prepare_stops_before_probes_and_reports_common_exclusions(reviewed, monkeypatch, mode):
    cfg, slot, runs, _, _ = reviewed
    stable, prepared, _ = qualification.audit_data(cfg, slot, runs)
    monkeypatch.setattr(study, "load_tokenizer", lambda path: TinyTokenizer())
    def embeddings(path, snapshots, split, official):
        old = len(snapshots["old"]["pool"])
        texts = [r["question"] for r in snapshots["new"]["pool"]]
        return dict(pool=[[1, 0] if i < old else [0, 1] for i in range(len(texts))],
                    queries={r["qid"]: [0, 1] for r in split["dev"]+split["test"]},
                    texts_sha256=digest(texts),
                    queries_sha256=digest([official.query(r) for r in split["dev"]+split["test"]]))
    monkeypatch.setattr(study, "make_embeddings", embeddings)
    original = study.prepare_question
    test_ids = [r["qid"] for r in stable["qualification"]["evaluation_splits"]["test"]]
    def altered_question(official, tokenizer, row, snapshots, vectors, query):
        snapshots = copy.deepcopy(snapshots)
        if mode == "empty_old_rules":
            snapshots["old"]["rules"] = "1. " + "x"*2000
        row = copy.deepcopy(row)
        if row["qid"] in (test_ids if mode == "all_context_excluded" else test_ids[:1]
                          if mode == "one_context_excluded" else []):
            row["question"] = "x"*6000
        return original(official, tokenizer, row, snapshots, vectors, query)
    monkeypatch.setattr(study, "prepare_question", altered_question)
    fingerprint = dict(data_binding=digest(stable), source=stable["source_sha256"], code={})
    directory = runs / "v5"
    with Ledger(slot, "v5", expected_previous=LEGACY_IDENTITY, data_binding=digest(stable)) as ledger:
        if mode == "one_context_excluded":
            manifest = study.prepare(cfg, directory, fingerprint, ledger, stable, prepared)
            assert len(manifest["eligible"]["test"]) == 93
            assert len(manifest["schedule"]["test"]) == 93*14
            assert manifest["data_qualified_counts"]["test"] == 94
            # Same-identity prepare must work after a submitted attempt as well.
            ledger.reserve("unknown", payload([1], 1))
            assert study.prepare(cfg, directory, fingerprint, ledger, stable, prepared) == manifest
        else:
            with pytest.raises(Invalid, match="DEGENERATE_MAIN_CONTROL"):
                study.prepare(cfg, directory, fingerprint, ledger, stable, prepared)
            assert ledger.counts()["attempts"] == 0 and not (directory / "manifest.json").exists()
            assert not study.load_artifact(directory, "intervention.json")["main_controls_valid"]


def test_a12_versions_and_model_free_cli(capsys):
    import memgate
    assert memgate.__version__ == SOFTWARE_VERSION == "0.2.0"
    assert 'version = "0.2.0"' in (ROOT / "pyproject.toml").read_text()
    assert study.main(["version", "--config", "/nonexistent"]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "V5"


def test_real_public_review_binding_optional():
    """Runs with the independently downloaded public file; no external downloads or answers."""
    path = ROOT / ".cache/e2_review_public/test.parquet"
    if not path.is_file():
        pytest.skip("Optional local public parquet is absent")
    public = load_public_inputs(path)
    _, new, splits = split_public(public)
    audit = duplicate_audit(public[:new], splits)
    q = qualification.qualify(public, splits, audit, read_json(REVIEW_PATH))
    assert set(q["exclusions"]["test"]) == {"12215", "11932"}
    assert "11872" in {r["qid"] for r in q["evaluation_splits"]["test"]}
    assert not audit["qualified"] and len(q["evaluation_splits"]["test"]) == 94
