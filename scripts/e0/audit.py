"""Read-only E0 audit: completeness, baseline repeat identity, reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from memgate.config import load_config, model_tag, resolve
from memgate.io.seqmem_logs import eval_log_path, run_dir


def repeat_comparison(first, second):
    """Compare ordered sample identities, text and outcomes; accuracy alone is insufficient."""
    if not first or not second or [str(r.get("qid")) for r in first] != [
        str(r.get("qid")) for r in second
    ]:
        return {"paired": False, "n_rep0": len(first), "n_rep1": len(second)}
    n = len(first)
    same_text = sum(a.get("model_output") == b.get("model_output") for a, b in zip(first, second))
    same_score = sum(a.get("score") == b.get("score") for a, b in zip(first, second))
    flips = sum(float(a.get("score") or 0) >= 1 and float(b.get("score") or 0) < 1
                for a, b in zip(first, second))
    return {"paired": True, "n": n, "same_output_count": same_text,
            "same_output_rate": same_text / n, "same_score_rate": same_score / n,
            "noise_hurt_rate": flips / n,
            "all_outputs_identical": same_text == n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e0_pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    tag = model_tag(cfg["llm"]["served_model"])
    runs = resolve(cfg["runs_dir"])
    report = resolve(cfg["reports_dir"]) / tag
    result = {"baseline_repeats": [], "job_inventory": [], "reproduction_outliers": []}
    base = cfg["baseline"]["log_name"]
    for task in cfg["tasks"]:
        logs = []
        for name, repeats in [(base, cfg["baseline"]["repeats"])] + [
            (m["log_name"], cfg.get("method_repeats", 1)) for m in cfg["methods"]
        ]:
            for rep in range(repeats):
                directory = run_dir(runs, tag, name, rep)
                path = eval_log_path(directory, task, name)
                entries = json.loads(path.read_text()) if path.exists() else []
                marked = (directory / f"{task}.done").exists()
                inventory = {"task": task, "method": name, "rep": rep, "n": len(entries),
                             "done_marker": marked, "path": str(path)}
                if path.exists():
                    inventory["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                result["job_inventory"].append(inventory)
                if name == base:
                    logs.append(entries)
        if len(logs) >= 2:
            result["baseline_repeats"].append(
                {"task": task, **repeat_comparison(logs[0], logs[1])})
    path = report / "reproduction.csv"
    if path.exists():
        frame = pd.read_csv(path)
        result["reproduction_outliers"] = frame[
            frame["delta"].abs() > cfg["reproduction_tolerance"]
        ].to_dict(orient="records")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
