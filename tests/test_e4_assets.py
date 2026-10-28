"""Read-only asset integrity, actual input binding and portable CLI behavior."""
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from memgate.e4_assets import STATUS, VERSION
from memgate.e4_assets import assets
from memgate.e4_assets.cli import main
from memgate.e4_assets.provenance import MODULE_FILES, implementation_identity
from memgate.experiments.common import (
    ROOT, Invalid, canonical, digest, file_hash, seal, unseal,
)
from memgate.experiments.official import Official
from memgate.experiments.prompts import prepare_question
from memgate.experiments.source import build_snapshots
from memgate.replay import export as e3
from test_e2_components import TinyTokenizer, public_rows, source_rows
import test_e3_replay as e3_tests


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    repo = tmp_path_factory.mktemp("e4-source")
    public = public_rows()
    public[0]["question"] += " \u03b1"
    prefix = source_rows(public[:581])
    marker = repo / "RAW_TEXT_WAS_EXECUTED"
    prefix[1]["memory"]["rollout"]["attempts"][0]["trajectory"] = (
        f"__import__('pathlib').Path({str(marker)!r}).touch()"
    )
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\r\n" for r in prefix)
    snapshots = build_snapshots(prefix, 193, 581)
    official, tokenizer = Official(ROOT / "third_party/SeqMem-Eval"), TinyTokenizer()
    split = {"dev": public[581:645], "test": public[645:739]}
    vectors = {v: [[1, 0]] * len(snapshots[v]["pool"]) for v in ("old", "new")}
    table = dict(
        splits={p: [r["qid"] for r in rs] for p, rs in split.items()},
        snapshots=snapshots,
        questions={r["qid"]: prepare_question(official, tokenizer, r, snapshots, vectors, [1, 0])
                   for rs in split.values() for r in rs},
    )
    extras = {
        "prefix_raw.json": raw, "old_prefix_raw.json": "".join(raw.splitlines(keepends=True)[:193]),
        "public.json": public, "splits.json": split, "evaluation_splits.json": split,
    }
    original_write = e3_tests.write_artifact

    def with_sources(directory, name, value):
        if name == "manifest.json":
            value["boundaries"] = dict(old=193, new=581, total=969)
            value["files"].update({n: original_write(directory, n, v) for n, v in extras.items()})
        return original_write(directory, name, value)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(e3_tests, "write_artifact", with_sources)
        identity, directory, ledger = e3_tests.seed_export_repo(repo, table, with_retries=True)
        patch.setattr(e3, "RUN_ID", identity)
        try:
            path = repo / "assets.json.gz"
            summary = assets.export_assets(repo, path)
            bundle = unseal(json.loads(gzip.decompress(path.read_bytes())))
            yield dict(repo=repo, directory=directory, ledger=ledger, path=path, summary=summary,
                       bundle=bundle, identity=identity, marker=marker)
        finally:
            ledger.close()


def write_bundle(path, bundle):
    path.write_bytes(gzip.compress(canonical(seal(bundle)), mtime=0))


def rebind_artifact(bundle, name, value):
    """Adversarially reseal even the manifest links; semantic checks must still run."""
    text = (canonical(seal(value)) + b"\n").decode("utf-8")
    bundle["artifacts_utf8"][name] = text
    table = bundle["e2_table"]
    import hashlib
    table["manifest"]["files"][name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    table["manifest_sha256"] = digest(table["manifest"])
    table["analysis"]["manifest"] = table["manifest_sha256"]
    table["source_analysis_sha256"] = digest(table["analysis"])
    bundle["e2_table_sha256"] = digest(table)


def test_round_trip_exact_raw_bytes_and_readonly_wal(evidence, tmp_path, monkeypatch):
    from memgate.experiments.ledger import Transport
    monkeypatch.setattr(Transport, "request", lambda *a, **kw: pytest.fail("Network access"))
    repo, ledger = evidence["repo"], evidence["ledger"]
    tracked = list(evidence["directory"].rglob("*.json"))
    tracked += [repo / "runs/.e2-ledgers/component-v4.sqlite",
                repo / "runs/.e2-ledgers/component-v4.sqlite-wal"]
    before = {p: file_hash(p) for p in tracked}
    meta = ledger.db.execute("SELECT * FROM meta ORDER BY key").fetchall()
    output, summary_path = tmp_path / "export.json.gz", tmp_path / "summary.json"
    summary = assets.export_assets(repo, output)
    assert assets.inspect_assets(output, summary_path) == summary == evidence["summary"]
    assert unseal(json.loads(summary_path.read_text())) == summary
    assert summary["status"] == STATUS and summary["new_model_calls"] == 0
    assert summary["counts"] == dict(
        development_questions=158, old_answers=632, e2_rows=2212, historical_roots=1,
        public_questions=969, old_prefix_rows=193, new_prefix_rows=581, old_pool=96, new_pool=290,
    )
    assert len(summary["missing_qualification_gates"]) == 6
    assert {p: file_hash(p) for p in tracked} == before
    assert ledger.db.execute("SELECT * FROM meta ORDER BY key").fetchall() == meta
    assert not evidence["marker"].exists()
    for name, text in evidence["bundle"]["artifacts_utf8"].items():
        assert text.encode("utf-8") == (evidence["directory"] / name).read_bytes()
    raw = unseal(json.loads(evidence["bundle"]["artifacts_utf8"]["prefix_raw.json"]))
    assert "\r\n" in raw and "\u03b1" in raw
    assert evidence["bundle"]["e2_table"]["verification"]["attempt_counts"]["retries"] == 2


def test_development_binds_each_original_answer_and_prompt(evidence):
    bundle = evidence["bundle"]
    table, development = bundle["e2_table"], bundle["development"]
    assert development["contrast"] == "NN-OO"
    assert development["role"] == "development_only_including_former_E2_test"
    assert len(set(development["qids"])) == 158
    assert {(r["qid"], r["arm"], r["rep"]) for r in development["answers"]} == {
        (q, a, r) for q in development["qids"] for a in ("OO", "NN") for r in (0, 1)
    }
    for row in development["answers"]:
        question = table["questions"][row["qid"]]
        prompt = question["prompts"][row["arm"]]
        assert row["prompt_ids_sha256"] == digest(prompt["ids"])
        assert row["prompt_record_sha256"] == digest(prompt)
        assert row["public_sha256"] == digest(question["public"])
        assert row["question_artifact_sha256"] == table["manifest"]["files"][
            f"questions/{row['qid']}.json"]


@pytest.mark.parametrize("mutation,match", [
    ("raw_resealed", "raw-byte"),
    ("raw_whitespace", "raw-byte"),
    ("missing", "source artifacts"),
    ("question", "question/input"),
    ("development", "Development"),
    ("chain_rebound", "Broken rule chain"),
    ("old_prefix_rebound", "exact leading prefix"),
    ("public_gold_rebound", "only qid"),
    ("snapshot_rebound", "chain/snapshots"),
    ("span_rebound", "input/span"),
    ("missing_row", "answer rows"),
    ("answer_rebound", "Answer/gold"),
    ("status", "version/status"),
    ("code_identity", "code identity"),
])
def test_resealed_tampering_is_rejected(evidence, tmp_path, mutation, match):
    bundle = copy.deepcopy(evidence["bundle"])
    table = bundle["e2_table"]
    if mutation == "raw_resealed":
        value = unseal(json.loads(bundle["artifacts_utf8"]["prefix_raw.json"]))
        bundle["artifacts_utf8"]["prefix_raw.json"] = canonical(seal(value + "\n")).decode()
    elif mutation == "raw_whitespace":
        bundle["artifacts_utf8"]["prefix_raw.json"] += " "
    elif mutation == "missing":
        del bundle["artifacts_utf8"]["prefix_raw.json"]
    elif mutation == "question":
        q = next(iter(table["questions"].values()))
        q["prompts"]["OO"]["ids"][0] += 1
    elif mutation == "development":
        bundle["development"]["answers"][0]["request_sha256"] = "0" * 64
    elif mutation == "chain_rebound":
        raw = unseal(json.loads(bundle["artifacts_utf8"]["prefix_raw.json"]))
        lines = raw.splitlines(keepends=True)
        row = json.loads(lines[-1])
        row["memory_meta"]["insights_before"] = "(empty)"
        lines[-1] = json.dumps(row) + "\r\n"
        rebind_artifact(bundle, "prefix_raw.json", "".join(lines))
    elif mutation == "old_prefix_rebound":
        raw = unseal(json.loads(bundle["artifacts_utf8"]["old_prefix_raw.json"]))
        rebind_artifact(bundle, "old_prefix_raw.json", raw.replace("\r\n", "\n"))
    elif mutation == "public_gold_rebound":
        public = unseal(json.loads(bundle["artifacts_utf8"]["public.json"]))
        public[-1]["answer"] = "A"
        rebind_artifact(bundle, "public.json", public)
    elif mutation == "snapshot_rebound":
        table["snapshots"]["old"]["rules"] = "1. Invented endpoint."
        rebind_artifact(bundle, "snapshots.json", table["snapshots"])
    elif mutation == "span_rebound":
        qid = next(iter(table["questions"]))
        q = table["questions"][qid]
        q["prompts"]["OO"]["rule_span"]["char_range"][0] += 1
        rebind_artifact(bundle, f"questions/{qid}.json", q)
    elif mutation == "missing_row":
        table["rows"].pop()
    elif mutation == "answer_rebound":
        table["rows"][0]["score"] = 1 - table["rows"][0]["score"]
    elif mutation == "status":
        bundle["status"] = "E4_QUALIFIED"
    else:
        bundle["exporter"]["files"]["experiments/source.py"] = "0" * 64
    bundle["e2_table_sha256"] = digest(table)
    path, summary = tmp_path / "bad.json.gz", tmp_path / "summary.json"
    write_bundle(path, bundle)
    with pytest.raises(Invalid, match=match):
        assets.inspect_assets(path, summary)
    assert not summary.exists()
    assert main(["inspect", "--input", str(path), "--output", str(summary)]) == 2
    assert not summary.exists()


def test_missing_source_rejects_export_without_partial_output(evidence, tmp_path):
    path = evidence["directory"] / "prefix_raw.json"
    hidden = path.with_suffix(".hidden")
    output = tmp_path / "new-directory" / "assets.json.gz"
    path.rename(hidden)
    try:
        assert main(["export", "--repo", str(evidence["repo"]), "--output", str(output)]) == 2
        assert not output.parent.exists()
    finally:
        hidden.rename(path)


@pytest.mark.parametrize("symlink", [False, True])
def test_outputs_never_overwrite_even_dangling_symlinks(tmp_path, monkeypatch, symlink):
    path = tmp_path / "output"
    if symlink:
        path.symlink_to(tmp_path / "missing-target")
    else:
        path.write_bytes(b"original")
    monkeypatch.setattr(e3, "export_table", lambda *a: pytest.fail("Must reject before ledger read"))
    with pytest.raises(Invalid, match="already exists"):
        assets.export_assets(tmp_path, path)
    with pytest.raises(Invalid, match="already exists"):
        assets.inspect_assets(tmp_path / "missing-input", path)
    assert path.is_symlink() if symlink else path.read_bytes() == b"original"
    assert not (tmp_path / "missing-target").exists()


def test_publication_failure_and_collision_cleanup(tmp_path, monkeypatch):
    output = tmp_path / "output"
    original = os.link

    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(assets.os, "link", fail)
    with pytest.raises(OSError, match="publication"):
        assets.publish_bytes(output, b"complete")
    assert not output.exists() and not list(tmp_path.glob(".e4-assets-*"))

    def race(source, target):
        Path(target).write_bytes(b"concurrent publisher")
        original(source, target)

    monkeypatch.setattr(assets.os, "link", race)
    with pytest.raises(FileExistsError):
        assets.publish_bytes(output, b"complete")
    assert output.read_bytes() == b"concurrent publisher"
    assert not list(tmp_path.glob(".e4-assets-*"))


def test_missing_ledger_and_protected_output_are_readonly(tmp_path):
    with pytest.raises(Invalid, match="Missing original ledger"):
        assets.export_assets(tmp_path, tmp_path / "assets.json.gz")
    for output in ("runs/new-assets.json.gz", "third_party/new-assets.json.gz"):
        with pytest.raises(Invalid, match="protected"):
            assets.export_assets(tmp_path, tmp_path / output)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("data", [b"not gzip", gzip.compress(b"{}"), gzip.compress(b'{"value":')])
def test_malformed_input_exits_without_summary(tmp_path, data):
    path, output = tmp_path / "bad.gz", tmp_path / "summary.json"
    path.write_bytes(data)
    assert main(["inspect", "--input", str(path), "--output", str(output)]) == 2
    assert not output.exists()


def test_zipapp_builder_real_entrypoints_and_exit_codes(evidence, tmp_path):
    archive = tmp_path / "e4.pyz"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}

    def run(args, site=True):
        return subprocess.run([sys.executable, *([] if site else ["-S"]), *map(str, args)],
                              cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)

    built = run([ROOT / "scripts/e4/build_zipapp.py", "--output", archive])
    assert built.returncode == 0, built.stdout + built.stderr
    with zipfile.ZipFile(archive) as zipped:
        assert {i.filename for i in zipped.infolist() if not i.is_dir()} == {
            "__main__.py", *("memgate/" + n for n in MODULE_FILES)}
    version = run([archive, "version"], site=False)
    assert version.returncode == 0, version.stderr
    identity = json.loads(version.stdout)
    assert identity["version"] == VERSION
    assert identity["files"] == implementation_identity()["files"]
    assert "experiments/source.py" in identity["files"]
    assert run([ROOT / "scripts/e4/build_zipapp.py", "--output", archive]).returncode == 2
    for entry in (archive, ROOT / "scripts/e4/assets.py"):
        assert run([entry, "version"], site=False).returncode == 0
        for command in (
            ["export", "--repo", tmp_path, "--output", tmp_path / "missing-export.gz"],
            ["inspect", "--input", tmp_path / "missing.gz", "--output", tmp_path / "summary.json"],
        ):
            failed = run([entry, *command])
            assert failed.returncode == 2 and "ERROR:" in failed.stderr
        assert run([entry, "generate"]).returncode == 2

    # Change only the test process's run constant, then execute the real archive
    # entrypoint. The release itself retains the genuine pinned E2 identity.
    bootstrap = (
        "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
        "from memgate.replay import export; "
        f"export.RUN_ID={evidence['identity']!r}; "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    output = tmp_path / "zipped-assets.gz"
    exported = run(["-c", bootstrap, archive, "export", "--repo", evidence["repo"], "--output", output])
    assert exported.returncode == 0, exported.stdout + exported.stderr
    inspected = run(["-c", bootstrap, archive, "inspect", "--input", output], site=False)
    assert inspected.returncode == 0, inspected.stdout + inspected.stderr
    summary = json.loads(inspected.stdout)
    assert summary["status"] == STATUS and summary["counts"]["old_answers"] == 632
    assert summary == json.loads(exported.stdout)
    assert summary["exporter_source_sha256"] == implementation_identity()["source_sha256"]
    assert not (tmp_path / "missing-export.gz").exists()
    assert not (tmp_path / "summary.json").exists()
