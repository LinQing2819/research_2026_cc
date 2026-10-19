"""Verify leakage boundaries, auxiliary events and selection/test separation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memgate.analysis.snapshots import compare_snapshots, snapshot_prefix


def test_snapshot_keeps_boundary_workflow_without_future_sample():
    rows = [
        {"qid": "a", "memory_event": {"type": "sample_observed"}},
        {"qid": "a", "memory_event": {"type": "workflow_induced", "workflow_text": "old"}},
        {"qid": "b", "memory_event": {"type": "sample_observed"}},
        {"qid": "b", "memory_event": {"type": "workflow_induced", "workflow_text": "boundary"}},
        {"qid": "c", "memory_event": {"type": "sample_observed"}},
    ]
    prefix = snapshot_prefix(rows, ["a", "b", "c"], 2)
    assert prefix == rows[:4]
    assert prefix[-1]["memory_event"]["workflow_text"] == "boundary"
    with pytest.raises(ValueError, match="IDs/order"):
        snapshot_prefix(rows, ["b", "a", "c"], 2)
    with pytest.raises(ValueError, match="IDs/order"):
        snapshot_prefix(rows[:-1], ["a", "b", "c"], 2)


def test_test_results_cannot_choose_snapshot():
    selection = pd.DataFrame({"Baseline": [0, 0], "k2": [1, 1], "k6": [0, 1]})
    test = pd.DataFrame({"Baseline": [1, 1], "k2": [0, 0], "k6": [1, 1]})
    policies, ceilings = compare_snapshots(selection, test, "k6", n_boot=100)
    chosen = policies.set_index("policy").loc["validation_selected_with_fallback"]
    assert chosen["arm"] == "k2"
    assert chosen["gain_vs_latest"] == -1
    assert chosen["gain_vs_latest_ci_hi"] == -1
    assert ceilings["test_best_fixed_arm_descriptive"] == 1


def test_ties_prefer_latest_and_missing_pairs_fail():
    frame = pd.DataFrame({"Baseline": [1, 0], "k2": [1, 0], "k6": [1, 0]})
    policies, _ = compare_snapshots(frame, frame, "k6", n_boot=100)
    assert policies.set_index("policy").loc["validation_selected_memory", "arm"] == "k6"
    with pytest.raises(ValueError, match="Missing"):
        compare_snapshots(frame, frame.assign(k2=np.nan), "k6")


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts/e1/snapshot_study.py"
    spec = importlib.util.spec_from_file_location("snapshot_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_run_analyze_public_format_with_stub_backend(tmp_path, monkeypatch):
    """Exercise manifests, split construction, frozen commands and complete jobs."""
    study = _script()
    from types import SimpleNamespace

    data = tmp_path / "math.jsonl"
    raw = [{"subject": "x", "problem": f"p{i}", "solution": f"s{i}"} for i in range(20)]
    study.write_jsonl(data, raw)
    monkeypatch.setitem(study.DATA_OVERRIDES, "MATH500", data)
    runroot, reportroot = tmp_path / "e0", tmp_path / "reports"
    tag = "Qwen__Qwen3-8B"
    source_dir = runroot / tag / "HistoryRAG/rep0"
    source_dir.mkdir(parents=True)
    (source_dir / "MATH500.done").touch()
    study.write_jsonl(source_dir / "MATH500_HistoryRAG_memory.jsonl",
                     [{"task": "MATH500", "qid": f"x-{i}", "score": 1} for i in range(20)])
    (reportroot / tag).mkdir(parents=True)
    pd.DataFrame([{"task": "MATH500", "method": "HistoryRAG", "n": 20,
                   "excess_hurt_rate": .1, "excess_hurt_ci_lo": .02, "uplift": -.05}]
                 ).to_csv(reportroot / tag / "flip_summary.csv", index=False)
    pd.DataFrame([{"task": "MATH500", "log_name": name, "delta": 0}
                  for name in ["Baseline", "Baseline", "HistoryRAG"]]
                 ).to_csv(reportroot / tag / "reproduction.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    e0 = study.load_config(root / "configs/e0_pilot.yaml")
    e0.update(runs_dir=str(runroot), reports_dir=str(reportroot))
    cfg = study.load_config(root / "configs/e1_snapshot.yaml")
    base = tmp_path / "study"
    args = SimpleNamespace(only_task=None, only_method=None, smoke=None)
    study.prepare(cfg, e0, base, args)
    manifest = json.loads((base / "manifest.json").read_text())
    cell = manifest["cells"][0]
    assert cell["qids"] == [f"x-{i}" for i in range(12, 20)]
    assert cell["n_selection"] == 4
    for arm, seed in cell["snapshots"].items():
        records = study.read_jsonl(seed["path"])
        assert len(records) == int(arm[1:])
        assert set(r["qid"] for r in records).isdisjoint(cell["qids"])

    commands = []
    real_run = study.subprocess.run

    def stub(cmd, **kwargs):
        if cmd[0] == "git":
            return real_run(cmd, **kwargs)
        commands.append(cmd)
        assert "--resume" not in cmd
        assert "--no-update-memory" in cmd
        out = Path(cmd[cmd.index("--output-dir") + 1])
        name = "Baseline" if cmd[cmd.index("--method") + 1] == "Baseline" else "HistoryRAG"
        arm = out.parent.name
        values = [1] * 8 if arm == "k4" else [0] * 8
        study.write_json(study.eval_log_path(out, "MATH500", name),
                         [{"qid": i + 1, "score": y} for i, y in enumerate(values)])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, "run", stub)
    study.run(manifest, base, False)
    assert len(commands) == 8
    study.run(manifest, base, False)
    assert len(commands) == 8  # Completed jobs skipped.
    report = tmp_path / "e1reports"
    study.analyze(manifest, base, report)
    recovery = pd.read_csv(report / "recovery.csv").iloc[0]
    assert recovery["selected_memory"] == "k4"
    assert recovery["recovery"] == 1
    assert recovery["promising_exploratory"]
    (Path(cell["snapshots"]["k4"]["path"])).write_text("{}\n")
    with pytest.raises(ValueError, match="input changed"):
        study.run(manifest, base, True)
