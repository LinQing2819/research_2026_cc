import importlib.util
from pathlib import Path


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/e0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e0_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_accuracy_does_not_imply_identical_repetitions():
    audit = load_script("audit")
    a = [{"qid": "a", "score": 1, "model_output": "one"},
         {"qid": "b", "score": 0, "model_output": "two"}]
    b = [{"qid": "a", "score": 0, "model_output": "three"},
         {"qid": "b", "score": 1, "model_output": "four"}]
    report = audit.repeat_comparison(a, b)
    assert report["paired"]
    assert report["same_score_rate"] == 0
    assert report["noise_hurt_rate"] == .5
    assert not report["all_outputs_identical"]
    assert audit.repeat_comparison(a, a)["all_outputs_identical"]
    assert not audit.repeat_comparison(a, b[::-1])["paired"]


def test_partial_or_mismatched_baseline_cannot_enter_analysis(tmp_path, make_log):
    analyze = load_script("analyze")
    from memgate.io.seqmem_logs import eval_log_path, run_dir

    cfg = {"baseline": {"log_name": "Baseline", "repeats": 2}}
    for rep in range(2):
        directory = run_dir(tmp_path, "model", "Baseline", rep)
        directory.mkdir(parents=True)
        eval_log_path(directory, "T", "Baseline").write_text(
            make_log(f"base{rep}.json", [1, 0]).read_text())
    assert analyze._load_baselines(cfg, tmp_path, "model", "T") == []
    for rep in range(2):
        (run_dir(tmp_path, "model", "Baseline", rep) / "T.done").touch()
    assert len(analyze._load_baselines(cfg, tmp_path, "model", "T")) == 2
    eval_log_path(run_dir(tmp_path, "model", "Baseline", 1), "T", "Baseline").write_text(
        make_log("short.json", [1]).read_text())
    assert analyze._load_baselines(cfg, tmp_path, "model", "T") == []
