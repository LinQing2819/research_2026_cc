import copy
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memgate.experiments.common import (ARMS, ROOT, Invalid, atomic_json, digest,
                                       file_hash, logical_key, seal, stable_seed)
from memgate.experiments.inference import authorized_projection, select, selector_authorizations
from memgate.experiments.ledger import Ledger, Transport, payload, validate_response
from memgate.experiments.official import Official
from memgate.experiments.prompts import chat, prepare_question, sentinel_prompts
from memgate.experiments.source import build_snapshots, duplicate_audit, split_public, validate_source
from memgate.experiments.study import (check_service, collect, load_artifact, preflight,
                                      prepare, report, run_phase, verify_manifest,
                                      verified_selection, write_artifact)
from test_e2_components import TinyTokenizer, public_rows, source_rows, tiny_snapshots, write_source


def response(request, text="\\boxed{A}", finish="stop"):
    return dict(status=200, body=json.dumps(dict(
        model=request["model"], choices=[dict(text=text, finish_reason=finish)],
        usage=dict(prompt_tokens=len(request["prompt"]), completion_tokens=8,
                   total_tokens=len(request["prompt"])+8))))


class StubTransport:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), []

    def request(self, path, request):
        self.calls.append(copy.deepcopy(request))
        return self.replies.pop(0) if self.replies else response(request)


@pytest.fixture
def fake_http():
    state = dict(calls=[], directives=[], tokenizer=TinyTokenizer(), identity=None, requests=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.handle_request(None)

        def do_POST(self):
            self.handle_request(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def handle_request(self, body):
            state["requests"].append(self.path)
            status = 200
            if self.path == "/v1/completions":
                assert all(type(x) is int for x in body["prompt"])
                state["calls"].append(body)
                directive = state["directives"].pop(0) if state["directives"] else None
                raw = directive if directive is not None else response(body)
                status, output = raw["status"], raw["body"]
            elif self.path == "/e2/identity":
                output = json.dumps(state["identity"])
            elif self.path == "/v1/models":
                output = json.dumps({"data": [{"id": "Qwen/Qwen3-8B"}]})
            elif self.path == "/e2/chat_ids":
                output = json.dumps(chat(state["tokenizer"], body["messages"][0]["content"]))
            elif self.path == "/tokenize":
                output = json.dumps(dict(tokens=state["tokenizer"].encode(body["prompt"]),
                                         max_model_len=8192))
            else:
                status, output = 404, "{}"
            encoded = output.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
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
        thread.join(timeout=2)


def test_t07_service_identity_and_usage_fail_closed():
    manifest = {"fingerprint": {"model": {"x": "hash"}, "code": {"scripts/e2/server.py": "code"}}}
    identity = dict(actual_max_model_len=8192, served_models=["Qwen/Qwen3-8B"],
                    model_files={"x": "hash"}, dtype="torch.bfloat16",
                    tensor_parallel_size=1, max_num_seqs=1, generation_config="vllm", launcher_sha256="code",
                    enforce_eager=True, max_num_batched_tokens=8192, gpu=[{"name": "fake"}])
    check_service(identity, manifest)
    for key, bad in (("actual_max_model_len", 8191), ("served_models", ["other"]),
                     ("model_files", {"x": "wrong"}), ("tensor_parallel_size", 2)):
        changed = {**identity, key: bad}
        with pytest.raises(Invalid):
            check_service(changed, manifest)
    request = payload([1, 2], 5)
    assert validate_response(response(request, finish="length"), request)["finish_reason"] == "length"
    bad = response(payload([1, 2, 3], 5))
    with pytest.raises(Invalid, match="count mismatch"):
        validate_response(bad, request)
    with pytest.raises(Invalid):
        payload([1]*5889, 1)


def test_t08_http_429_500_budget_and_seed_are_durable(tmp_path, fake_http):
    transport, state = fake_http
    state["directives"] = [dict(status=429, body="busy"), dict(status=500, body="failed")]
    request = payload([1, 2, 3], 17)
    path = tmp_path / "ledger.sqlite"
    with Ledger(path, "id") as ledger:
        value = ledger.complete("q", request, transport, lambda t: {"score": 1}, "binding")
        assert value["score"] == 1 and ledger.counts()["retries"] == 2
        assert state["calls"] == [request]*3
    with Ledger(path, "id") as ledger:
        ledger.complete("q", request, transport, lambda t: {"score": 1}, "binding")
        assert len(state["calls"]) == 3
        with pytest.raises(Invalid, match="payload|budget|exhausted"):
            ledger.reserve("q", request)


def test_t08_400_and_malformed_success_never_retry(tmp_path):
    request = payload([1], 8)
    for index, reply in enumerate((dict(status=400, body="context overflow"),
                                   dict(status=200, body='{"choices":[]}'))):
        transport = StubTransport([reply])
        with Ledger(tmp_path / f"{index}.sqlite", "id") as ledger:
            for _ in range(2):
                with pytest.raises(Invalid):
                    ledger.complete("q", request, transport, lambda text: {}, "b")
            assert len(transport.calls) == 1


def test_t08_global_retry_probe_and_first_attempt_caps(tmp_path):
    request = payload([1], 5)
    with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
        for i in range(44):
            key = f"q{i}"
            ledger.reserve(key, request)
            ledger.reserve(key, request)
        ledger.reserve("q44", request)
        with pytest.raises(Invalid, match="Global"):
            ledger.reserve("q44", request)
        ledger.reserve("probe:0", request, probe=True)
        ledger.reserve("probe:1", request, probe=True)
        with pytest.raises(Invalid, match="Global"):
            ledger.reserve("probe:2", request, probe=True)
        with pytest.raises(Invalid, match="only once"):
            ledger.reserve("probe:0", request, probe=True)
    with Ledger(tmp_path / "small.sqlite", "id", limits={"logical": 1}) as ledger:
        ledger.reserve("first", request)
        with pytest.raises(Invalid, match="Global"):
            ledger.reserve("second", request)


def test_t09_unknown_dispatch_and_raw_response_recovery(tmp_path):
    path, request = tmp_path / "ledger.sqlite", payload([1, 2], 71)
    with Ledger(path, "id") as ledger:
        ledger.reserve("unknown_after_dispatch", request)
        attempt = ledger.reserve("raw_before_score", request)
        ledger.save_raw(attempt, response(request))
    with Ledger(path, "id") as ledger:
        transport = StubTransport()
        ledger.complete("raw_before_score", request, transport, lambda t: {"score": 1}, "binding")
        assert transport.calls == []
        ledger.complete("unknown_after_dispatch", request, transport, lambda t: {}, "binding")
        assert len(transport.calls) == 1
        assert ledger.counts()["attempts"] == 3
        assert ledger.counts()["retries"] == 1
        # A second client cannot send anything while the writer holds the identity.
        with pytest.raises(Invalid, match="Another"):
            Ledger(path, "id")


def test_t09_timeout_counts_and_cannot_change_seed(tmp_path):
    request = payload([1], 8)
    transport = StubTransport([dict(status=None, body="", error="TimeoutError")])
    with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
        ledger.complete("q", request, transport, lambda t: {}, "b")
        assert ledger.counts()["attempts"] == 2
        assert transport.calls[0] == transport.calls[1]
        with pytest.raises(Invalid):
            ledger.complete("q", payload([1], 9), transport, lambda t: {}, "b")


@pytest.mark.parametrize("save_response", [False, True])
def test_t09_process_dies_after_reservation_or_raw_commit(tmp_path, save_response):
    path, request = tmp_path / "crash.sqlite", payload([1, 2], 71)
    # os._exit simulates abrupt death: no context manager or graceful DB/lock close.
    script = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
from memgate.experiments.ledger import Ledger
ledger = Ledger(sys.argv[2], "id")
request = json.loads(sys.argv[3])
attempt = ledger.reserve("crash", request)
if sys.argv[4] == "yes":
    ledger.save_raw(attempt, json.loads(sys.argv[5]))
os._exit(19)
"""
    child = subprocess.run([sys.executable, "-c", script, str(ROOT / "src"), str(path),
                            json.dumps(request), "yes" if save_response else "no",
                            json.dumps(response(request))], timeout=10)
    assert child.returncode == 19
    with Ledger(path, "id") as ledger:
        transport = StubTransport()
        ledger.complete("crash", request, transport, lambda t: {}, "binding")
        assert len(transport.calls) == (0 if save_response else 1)
        assert ledger.counts()["attempts"] == (1 if save_response else 2)


def test_t10_result_raw_and_manifest_checksums(tmp_path):
    request = payload([1], 8)
    with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
        ledger.complete("q", request, StubTransport(), lambda t: {"score": 1}, "binding")
        ledger.db.execute("UPDATE attempts SET raw_hash='broken'")
        ledger.db.commit()
        with pytest.raises(Invalid, match="Raw response"):
            ledger.verify_result("q", request, lambda t: {"score": 1}, "binding")
    # Copying/moving an artifact directory does not establish a fresh study slot.
    with pytest.raises(Invalid, match="identity"):
        Ledger(tmp_path / "ledger.sqlite", "different_identity")


def test_t11_authorization_is_a_prefix_and_selector_cannot_read_extra():
    questions = {}
    for qid in ("q1", "q2", "q3"):
        questions[qid] = {"prompts": {a: {"prompt_tokens": 20} for a in ARMS}}
    for a in ("ON", "NO"):
        questions["q2"]["prompts"][a]["prompt_tokens"] = 50000
    auth = selector_authorizations(["q1", "q2", "q3"], questions)
    assert auth["S"]["qids"] == ["q1"]  # Does not skip q2 to read q3 for free.
    rows = [dict(qid=q, arm=a, rep=r, score=1) for q in questions for a in ARMS for r in range(2)]
    projection = authorized_projection(auth["S"], rows)
    assert select(auth["S"], projection)["arm"] == "NN"
    with pytest.raises(Invalid, match="Unauthorized"):
        select(auth["S"], projection+[dict(qid="q3", arm="NN", rep=0, score=1)])
    exact_questions = {q: {"prompts": {a: {"prompt_tokens": 20} for a in ARMS}}
                       for q in ("one", "two")}
    exact = selector_authorizations(["one", "two"], exact_questions)
    assert exact["S"]["qids"] == ["one"] and exact["S"]["remaining"] == 0
    exact_questions["one"]["prompts"]["ON"]["prompt_tokens"] += 1
    with pytest.raises(Invalid, match="cannot afford"):
        selector_authorizations(["one", "two"], exact_questions)


def miniature_study(directory):
    official, tok = Official(ROOT / "third_party/SeqMem-Eval"), TinyTokenizer()
    qids = {"dev": ["901", "902", "903", "904"], "test": ["905", "906"]}
    hashes, questions = {}, {}
    for q in qids["dev"]+qids["test"]:
        public = dict(qid=q, question=f"Evaluation {q}", options=["a", "b"])
        result = prepare_question(official, tok, public, tiny_snapshots(),
                                  {"old": [[1, 0]], "new": [[1, 0], [0, 1]]}, [0, 1])
        questions[q] = result
        name = f"questions/{q}.json"
        hashes[name] = write_artifact(directory, name, result)
    hashes["sentinels.json"] = write_artifact(directory, "sentinels.json", sentinel_prompts(tok))
    hashes["intervention.json"] = write_artifact(directory, "intervention.json", {"jointly_changed": 2})
    manifest = dict(fingerprint={"model": {"weights": "hash"},
                                "code": {"scripts/e2/server.py": "code"}},
                    files=hashes, eligible=qids, original_counts={"dev": 4, "test": 2},
                    data_qualified_counts={"dev": 4, "test": 2}, data_exclusions={"dev": [], "test": []},
                    authorization=selector_authorizations(qids["dev"], questions), schedule={})
    for phase, ids in qids.items():
        manifest["schedule"][phase] = [
            dict(key=logical_key(phase, q, a, r), phase=phase, qid=q, arm=a, rep=r,
                 seed=stable_seed(phase, q, a, r)) for q in ids for a in ARMS for r in range(2)]
    write_artifact(directory, "manifest.json", manifest)
    gold = dict.fromkeys(qids["dev"]+qids["test"], "A")
    return manifest, official, gold


def test_t07_t11_t12_http_staged_smoke_resume_selection_and_report(tmp_path, fake_http):
    transport, state = fake_http
    directory = tmp_path / "artifacts"
    manifest, official, gold = miniature_study(directory)
    state["identity"] = dict(actual_max_model_len=8192, served_models=["Qwen/Qwen3-8B"],
                             model_files=manifest["fingerprint"]["model"], dtype="torch.bfloat16",
                             tensor_parallel_size=1, max_num_seqs=1, generation_config="vllm",
                             launcher_sha256="code", enforce_eager=True,
                             max_num_batched_tokens=8192, gpu=[{"name": "fake"}])
    path = tmp_path / "ledger.sqlite"
    with Ledger(path, "id") as ledger:
        ledger.bind("manifest", digest(manifest))
        preflight(transport, manifest, directory, ledger)
        assert len(state["calls"]) == 2
        preflight(transport, manifest, directory, ledger)
        assert len(state["calls"]) == 2
        with pytest.raises(Invalid, match="Test forbidden"):
            run_phase("test", manifest, directory, ledger, transport, official, gold)
        assert len(state["calls"]) == 2
        with pytest.raises(Invalid, match="frozen dev smoke"):
            run_phase("dev", manifest, directory, ledger, transport, official, gold)
        assert len(state["calls"]) == 2
        run_phase("dev", manifest, directory, ledger, transport, official, gold, smoke=True)
        assert len(state["calls"]) == 58
    with Ledger(path, "id") as ledger:
        run_phase("dev", manifest, directory, ledger, transport, official, gold)
        assert len(state["calls"]) == 58
        selection = verified_selection(manifest, directory, ledger, official, gold)
        assert selection["selectors"]["S"]["arm"] == "NN"
        assert selection["selectors"]["F"]["arm"] == "NN"
        with pytest.raises(Invalid, match="Missing result"):
            report(manifest, directory, ledger, transport, official, gold)
        run_phase("test", manifest, directory, ledger, transport, official, gold)
        assert len(state["calls"]) == 86
        result = report(manifest, directory, ledger, transport, official, gold)
        assert result["contrasts"]["D_F"]["ci"] == [0, 0]
        assert result["attempt_counts"] == dict(logical=84, probes=2, retries=0, attempts=86)
        # Same ID count is insufficient: differing actual token IDs must reject preflight.
        original = state["tokenizer"]
        class ChangedTokenizer(TinyTokenizer):
            def encode(self, text, **kwargs):
                return [i+1 for i in super().encode(text, **kwargs)]
        state["tokenizer"] = ChangedTokenizer()
        with pytest.raises(Invalid, match="IDs differ"):
            preflight(transport, manifest, directory, ledger)
        assert len(state["calls"]) == 86
        state["tokenizer"] = original
        # Even a resealed edited selection is rejected against the separate ledger binding.
        changed = copy.deepcopy(selection)
        changed["selectors"]["S"]["arm"] = "ON"
        atomic_json(directory / "selection.json", seal(changed), immutable=False)
        with pytest.raises(Invalid, match="modified"):
            verified_selection(manifest, directory, ledger, official, gold)


def test_t01_t10_real_prepare_no_models_and_fingerprint_guards(tmp_path, monkeypatch):
    from memgate.experiments import study
    public = public_rows()
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(dict(question_id=p["qid"], category="engineering",
                                          answer="A", **p))+"\n" for p in public))
    source = tmp_path / "source.jsonl"
    write_source(source, source_rows(public))
    cfg = dict(data=data, source=source, seqmem_root=ROOT / "third_party/SeqMem-Eval",
               model_path=tmp_path / "model", embedding_path=tmp_path / "embedding")
    monkeypatch.setattr(study, "load_tokenizer", lambda path: TinyTokenizer())
    def fake_embeddings(path, snapshots, split, official):
        old = len(snapshots["old"]["pool"])
        texts = [r["question"] for r in snapshots["new"]["pool"]]
        queries = [official.query(r) for r in split["dev"]+split["test"]]
        return dict(pool=[[1, 0] if i < old else [0, 1] for i in range(len(texts))],
                    queries={r["qid"]: [0, 1] for r in split["dev"]+split["test"]},
                    texts_sha256=digest(texts), queries_sha256=digest(queries))
    monkeypatch.setattr(study, "make_embeddings", fake_embeddings)
    old, new, splits = split_public(public)
    prefix, raw, source_hash = validate_source(source, public, new)
    prepared = dict(public=public, splits=splits, prefix_raw=raw,
                    snapshots=build_snapshots(prefix, old, new),
                    duplicate_audit=duplicate_audit(public[:new], splits))
    excluded = [r["qid"] for r in splits["test"][:2]]
    stable = dict(source_sha256=source_hash, data_sha256=file_hash(data), qualification=dict(
        data_qualified_counts={"dev": 64, "test": 94}, exclusions={"dev": [], "test": excluded},
        evaluation_splits={"dev": splits["dev"], "test": splits["test"][2:]}))
    fingerprint = dict(source=file_hash(source), model={}, code={}, data_binding=digest(stable))
    directory = tmp_path / "artifacts"
    with Ledger(tmp_path / "ledger.sqlite", "id") as ledger:
        manifest = prepare(cfg, directory, fingerprint, ledger, stable, prepared)
        assert manifest["boundaries"] == dict(total=969, old=193, new=581)
        assert sum(len(v) for v in manifest["schedule"].values()) == 2212
        assert not any((directory / f"questions/{q}.json").exists() for q in excluded)
        assert not set(excluded) & {r["qid"] for r in manifest["schedule"]["test"]}
        assert prepare(cfg, directory, fingerprint, ledger, stable, prepared) == manifest
        assert ledger.counts()["attempts"] == 0
        with pytest.raises(Invalid, match="changed"):
            verify_manifest(directory, {**fingerprint, "new_parameter": 1}, ledger)
        cache = directory / "embeddings.json"
        cache.write_text("{}")
        with pytest.raises(Invalid, match="artifact changed"):
            verify_manifest(directory, fingerprint, ledger)
