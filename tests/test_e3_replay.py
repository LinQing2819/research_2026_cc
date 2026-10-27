"""E3 evidence integrity, budgeted reads and test-label isolation."""
import copy
import gzip
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from memgate.experiments.common import (ARMS, LIMITS, ROOT, Invalid, atomic_json, digest,
                                       file_hash, logical_key, seal, stable_seed)
from memgate.experiments.inference import analyze_scores, authorized_projection, select, selector_authorizations
from memgate.experiments.ledger import Ledger, payload
from memgate.experiments.official import Official
from memgate.experiments.study import write_artifact
from memgate.replay import export as exporter
from memgate.replay.export import read_export
from memgate.replay.selection import (METHODS, ORDER, SEARCH_BUDGET, VALIDATION_BUDGET,
                                      Provider, feature, kernel, run_replay, select_one, validate_table)
from test_e2_components import tiny_snapshots
from test_e2_runtime import StubTransport


@pytest.fixture
def table():
    splits = {"dev": [str(i) for i in range(64)], "test": [str(i) for i in range(64, 158)]}
    snapshots = tiny_snapshots()
    questions, rows = {}, []
    for phase, qs in splits.items():
        for q in qs:
            questions[q] = dict(eligible=True, prompts={
                a: dict(prompt_tokens=100+i*10, ids=[i]*(100+i*10)) for i, a in enumerate(ARMS)},
                rules={"old": {"text": "old compact"}, "new": {"text": "new compact"},
                       "empty": {"text": "(empty)"}},
                cases={v: {"text": v + " case " + str(int(q) % 3)} for v in ("old", "new", "empty")},
                ranked={"old": [{**snapshots["old"]["pool"][0], "retrieval_score": 1.0}],
                        "new": [{**snapshots["new"]["pool"][1], "retrieval_score": .9}]})
            for i, arm in enumerate(ARMS):
                for rep in range(2):
                    rows.append(dict(phase=phase, qid=q, arm=arm, rep=rep,
                                     score=int((int(q)+i+rep) % 4 != 0),
                                     usage=dict(prompt_tokens=100+i*10, completion_tokens=3,
                                                total_tokens=103+i*10)))
    return dict(splits=splits, questions=questions, snapshots=snapshots, rows=rows)


def test_kernel_psd_and_unpacked_builder(table):
    official = Official(ROOT / "third_party/SeqMem-Eval")
    for method in METHODS:
        k = kernel(table, table["splits"]["dev"][:40], method, official)
        assert np.allclose(k, k.T) and np.linalg.eigvalsh(k).min() > 0
        assert np.allclose(np.diag(k), .25)
    assert feature("") and feature("a") and feature("ab")


def test_unauthorized_provider_and_expensive_prefix_not_skipped(table):
    questions = table["questions"]
    questions["0"]["prompts"]["OO"]["prompt_tokens"] = 5888
    p = Provider(table["rows"], questions, "dev", ["0", "1"], 12000)
    assert ("64", "OO", 0) not in p.scores
    assert p.cost("OO", 1) == 15872
    with pytest.raises(Invalid, match="overbudget"):
        p.buy("OO", 1, "cannot skip to cheap q1")
    assert p.spent == 0 and p.count["OO"] == 0
    p.buy("ZZ", 2, "allowed")
    assert p.trace[0]["qids"] == ["0", "1"]
    with pytest.raises(Invalid):
        p.buy("ZZ", 1, "no next qid")


def test_all_methods_budget_and_test_labels_cannot_change_choices(table):
    official = Official(ROOT / "third_party/SeqMem-Eval")
    poisoned = copy.deepcopy(table)
    for row in poisoned["rows"]:
        if row["phase"] == "test":
            row["score"] = 1-row["score"]
    for method in METHODS:
        K = kernel(table, table["splits"]["dev"][:40], method, official)
        left = select_one(table, method, 4, K)
        right = select_one(poisoned, method, 4, K)
        assert left == right
        assert left["search"]["reserved_tokens"] <= SEARCH_BUDGET
        assert left["validation"]["reserved_tokens"] <= VALIDATION_BUDGET
        assert left["validation"]["n"] >= 8
        assert "ZZ" in left["finalists"] and len(set(left["finalists"])) == 3
        assert set(left["validation"]["qids"]) <= set(table["splits"]["dev"][40:])
        for arm in ORDER:
            seen = [q for t in left["search"]["trace"] if t["arm"] == arm for q in t["qids"]]
            assert seen == left["search"]["qids"][:len(seen)]


def test_validation_scores_cannot_change_search_or_finalists(table):
    official = Official(ROOT / "third_party/SeqMem-Eval")
    K = kernel(table, table["splits"]["dev"][:40], "rendered", official)
    left = select_one(table, "rendered", 0, K)
    for row in table["rows"]:
        if row["qid"] in table["splits"]["dev"][40:]:
            row["score"] = int(row["arm"] == "ZZ")
    right = select_one(table, "rendered", 0, K)
    assert left["search"] == right["search"] and left["finalists"] == right["finalists"]
    assert right["selected"] == "ZZ"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "score", "length", "cross_split"])
def test_table_rejects_invalid_records(table, mutation):
    if mutation == "duplicate":
        table["rows"][-1] = table["rows"][0]
    elif mutation == "missing":
        table["rows"].pop()
    elif mutation == "score":
        table["rows"][0]["score"] = 2
    elif mutation == "length":
        table["questions"]["0"]["prompts"]["OO"]["prompt_tokens"] += 1
    else:
        table["splits"]["test"][0] = "0"
    with pytest.raises(Invalid):
        validate_table(table)


def test_complete_replay_reproducible_and_no_fake_history_inference(table):
    official = Official(ROOT / "third_party/SeqMem-Eval")
    a = run_replay(table, official)
    b = run_replay(table, official)
    assert a == b and a["new_generation_calls"] == 0
    assert len(a["choices"]) == 96 and len(a["paired_differences"]["uniform"]) == 16
    assert set(a["fixed_policy_accuracy"]) == set(ARMS)
    assert "p_value" not in a and "confidence_interval" not in a


def seed_export_repo(repo, table, with_retries=False):
    """Create complete fake 64/94 evidence using the real ledger/scorer; no model calls."""
    (repo / "third_party").mkdir()
    (repo / "third_party/SeqMem-Eval").symlink_to(ROOT / "third_party/SeqMem-Eval", target_is_directory=True)
    official = Official(repo / "third_party/SeqMem-Eval")
    names = ["methods/expel_online_st.py", "tasks/task_mmlu_pro.py",
             "tasks/math_answer_utils.py", "prompts/prompt_templates.json", "utils/embedder.py"]
    fingerprint = dict(official={"files": {n: file_hash(official.root/n) for n in names}}, synthetic=True)
    identity = digest(fingerprint)
    directory = repo / "runs/e2_component" / identity
    files = {}
    for q, question in table["questions"].items():
        files[f"questions/{q}.json"] = write_artifact(directory, f"questions/{q}.json", question)
    for name, value in {
        "scoring_gold.json": {q: "A" for q in table["questions"]},
        "snapshots.json": table["snapshots"],
        "sentinels.json": [{"ids": [1, 2]}, {"ids": [2, 3]}],
    }.items():
        files[name] = write_artifact(directory, name, value)
    schedule = {p: [dict(key=logical_key(p, q, a, r), phase=p, qid=q, arm=a, rep=r,
                         seed=stable_seed(p, q, a, r))
                    for q in qs for a in ARMS for r in range(2)] for p, qs in table["splits"].items()}
    manifest = dict(identity=identity, fingerprint=fingerprint, files=files, schedule=schedule,
                    eligible=table["splits"], authorization=selector_authorizations(
                        table["splits"]["dev"], table["questions"]))
    write_artifact(directory, "manifest.json", manifest)
    ledger = Ledger(repo / "runs/.e2-ledgers/component-v4.sqlite", identity)
    mh = digest(manifest)
    ledger.bind("manifest", mh)
    service = {"synthetic": True}
    ledger.bind("service", service)
    for i in range(2):
        ledger.complete(f"probe:{i}", payload([i+1, i+2], 2000000000+i), StubTransport(),
                        lambda text: {}, dict(manifest=mh, service=digest(service)), probe=True)
    preflight = dict(manifest=mh, service=service, probes=[digest(ledger.result(f"probe:{i}")) for i in range(2)])
    ph = digest(preflight)
    write_artifact(directory, "preflight.json", preflight)
    ledger.bind("preflight", ph)
    collected = {}
    def scorer(text):
        pred = official.extract_letter(text)
        return dict(predicted=pred, score=int(pred == "A"))
    selection = None
    for phase in ("dev", "test"):
        collected[phase] = []
        for index, item in enumerate(schedule[phase]):
            q, a, r = item["qid"], item["arm"], item["rep"]
            request = payload(table["questions"][q]["prompts"][a]["ids"], item["seed"])
            binding = dict(manifest=mh, preflight=ph, phase=phase, qid=q, arm=a, rep=r,
                           seed=item["seed"], selection=digest(selection) if phase == "test" else None)
            if with_retries and phase == "dev" and index == 0:
                ledger.reserve(item["key"], request)  # Unknown dispatch is charged.
                attempt = ledger.reserve(item["key"], request)
                ledger.save_raw(attempt, dict(status=200, body="", transport_error=True,
                                              error="IncompleteRead"))
            v = ledger.complete(item["key"], request, StubTransport(), scorer, binding)
            collected[phase].append(dict(qid=q, arm=a, rep=r, score=v["score"], result_sha256=digest(v),
                                         usage=v["usage"], finish_reason=v["finish_reason"]))
        if phase == "dev":
            selection = dict(manifest=mh, preflight=ph, dev_sha256=digest(collected["dev"]), selectors={
                k: select(auth, authorized_projection(auth, collected["dev"]))
                for k, auth in manifest["authorization"].items()})
            ledger.bind("selection", digest(selection))
            write_artifact(directory, "selection.json", selection)
    smoke_items = [i for i in schedule["dev"] if i["qid"] in set(table["splits"]["dev"][:4])]
    write_artifact(directory, "smoke.json", dict(manifest=mh, preflight=ph, rows=56,
                   qids=table["splits"]["dev"][:4],
                   results=[digest(ledger.result(i["key"])) for i in smoke_items]))
    analysis = analyze_scores(table["splits"]["test"], collected["test"],
                              selection["selectors"]["S"]["arm"], selection["selectors"]["F"]["arm"])
    analysis.update(manifest=mh, selection=selection, selection_sha256=digest(selection),
                    result_sha256=digest(collected["dev"]+collected["test"]), attempt_counts=ledger.counts())
    write_artifact(directory, "analysis.json", analysis)
    return identity, directory, ledger


def test_export_readonly_wal_rescore_and_corruption(tmp_path, table, monkeypatch):
    identity, directory, ledger = seed_export_repo(tmp_path, table, with_retries=True)
    monkeypatch.setattr(exporter, "RUN_ID", identity)
    try:
        # Hold the writer open: accepted rows may remain in WAL, which export must see.
        before = dict(ledger.db.execute("SELECT key,value FROM meta").fetchall())
        path = tmp_path / "evidence.json.gz"
        exported = exporter.export_table(tmp_path, path)
        assert len(exported["rows"]) == 2212
        assert exported["verification"]["attempt_counts"]["attempts"] == 2216
        assert exported["verification"]["attempt_counts"]["retries"] == 2
        assert read_export(path) == exported
        assert dict(ledger.db.execute("SELECT key,value FROM meta").fetchall()) == before
        from memgate.replay.cli import main
        assert main(["run", "--input", str(path), "--repo", str(tmp_path),
                     "--output-dir", str(tmp_path / "reports")]) == 0
        reports = list((tmp_path / "reports").glob("*/report.md"))
        assert len(reports) == 1 and "NO_INCREMENTAL_SIGNAL_ON_THIS_TABLE" in reports[0].read_text()
        # Exercise the packaged entry, not just imports from this checkout.
        archive = tmp_path / "tools.pyz"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for source in (ROOT / "src").rglob("*.py"):
                z.write(source, source.relative_to(ROOT / "src"))
            z.writestr("__main__.py",
                       "from memgate.replay import export\n"
                       f"export.RUN_ID={identity!r}\n"
                       "from memgate.replay.cli import main\n"
                       "raise SystemExit(main())\n")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        proc = subprocess.run([sys.executable, str(archive), "run", "--input", str(path),
                               "--repo", str(tmp_path), "--output-dir", str(tmp_path / "reports-zip")],
                              cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        from memgate.experiments.common import read_json, unseal
        zipped = unseal(read_json(next((tmp_path / "reports-zip").glob("*/replay.json"))))
        assert len(zipped["implementation"]["files"]) >= 10
        direct = unseal(read_json(next((tmp_path / "reports").glob("*/replay.json"))))
        assert zipped == direct
        with pytest.raises(FileExistsError):
            exporter.export_table(tmp_path, path)
        row = ledger.db.execute("SELECT key,value FROM results WHERE key NOT LIKE 'probe:%' LIMIT 1").fetchone()
        changed = json.loads(row["value"])
        changed["score"] = 0
        # Even a maliciously resealed score must fail original-response rescoring.
        ledger.db.execute("UPDATE results SET value=?,hash=? WHERE key=?",
                          (json.dumps(changed), digest(changed), row["key"]))
        ledger.db.commit()
        with pytest.raises(Invalid, match="original scorer"):
            exporter.export_table(tmp_path, tmp_path / "bad.json.gz")
        assert not (tmp_path / "bad.json.gz").exists()
    finally:
        ledger.close()
    question = directory / "questions/0.json"
    question.write_text("{}")
    with pytest.raises(Invalid, match="checksum"):
        exporter.export_table(tmp_path, tmp_path / "changed.json.gz")


def test_release_zipapp_propagates_failure_exit_status(tmp_path):
    built = subprocess.run(
        [sys.executable, str(ROOT / "scripts/e3/build_zipapp.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stdout + built.stderr
    archive = ROOT / ".cache/releases/e3-r1-tools.pyz"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    for command in (
        ["run", "--input", str(tmp_path / "missing.json.gz"),
         "--repo", str(tmp_path), "--output-dir", str(tmp_path / "reports")],
        ["export", "--repo", str(tmp_path), "--output", str(tmp_path / "export.json.gz")],
    ):
        failed = subprocess.run(
            [sys.executable, str(archive), *command], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=30)
        assert failed.returncode == 2, failed.stdout + failed.stderr
        assert "ERROR:" in failed.stdout
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "export.json.gz").exists()


def test_missing_ledger_never_created(tmp_path):
    with pytest.raises(Invalid, match="Missing original ledger"):
        exporter.export_table(tmp_path, tmp_path / "no.json.gz")
    assert not (tmp_path / "runs").exists()


def test_export_publication_failure_never_leaves_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "export.json.gz"
    def fail_link(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(exporter.os, "link", fail_link)
    with pytest.raises(OSError, match="publication"):
        exporter.publish_export(output, {"complete": True})
    assert not output.exists() and not list(tmp_path.glob(".e3-export-*"))
