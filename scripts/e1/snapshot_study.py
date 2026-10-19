"""E1a: prepare / run / analyse frozen public SeqMem-Eval snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "e0"))

from memgate.analysis.snapshots import compare_snapshots, is_sample, snapshot_prefix
from memgate.config import load_config, model_tag, resolve
from memgate.io.seqmem_logs import eval_log_path, load_eval_log, run_dir
from run_matrix import DATA_OVERRIDES, build_command, build_env


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=list) + "\n"
                            for r in rows), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_rows(task):
    """Match official ordering/IDs while keeping labels and evaluator unchanged."""
    path = DATA_OVERRIDES[task]
    if task == "MATH500":
        rows = read_jsonl(path)
        for i, row in enumerate(rows):
            row["unique_id"] = row.get("unique_id") or f"{row.get('subject', '')}-{i}"
        qids = [str(r["unique_id"]) for r in rows]
    elif task.startswith("MMLU-Pro-"):
        category = task.rsplit("-", 1)[1].lower()
        frame = pd.read_parquet(path)
        frame = frame[frame["category"] == category].reset_index(drop=True)
        rows = frame.to_dict(orient="records")
        for i, row in enumerate(rows):
            row["question_id"] = str(row.get("question_id", i))
            row["options"] = list(row["options"])
        qids = [r["question_id"] for r in rows]
    elif task == "HumanEval":
        rows = pd.read_parquet(path).to_dict(orient="records")
        qids = [str(r.get("task_id") or f"row-{i}") for i, r in enumerate(rows)]
    elif task == "APIBench-HF":
        rows = read_jsonl(path / "huggingface_eval.json")
        qids = [str(i + 1) for i in range(len(rows))]
    else:
        raise ValueError(f"Unsupported public task: {task}")
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate public IDs: {task}")
    return rows, qids


def prepare(cfg, e0, base, args):
    tag = model_tag(e0["llm"]["served_model"])
    reports = resolve(e0["reports_dir"]) / tag
    summary = pd.read_csv(reports / "flip_summary.csv")
    repro = pd.read_csv(reports / "reproduction.csv")
    candidates, excluded = [], []
    gc = e0["go_criteria"]
    methods = {m["log_name"]: m for m in e0["methods"]}
    for row in summary.to_dict(orient="records"):
        task, method = row["task"], row["method"]
        if args.only_task and task not in args.only_task:
            continue
        if args.only_method and method not in args.only_method and methods[method]["cli_name"] not in args.only_method:
            continue
        if method not in cfg["eligible_methods"]:
            reason = "method excluded (DC-RS seed restoration drops cheatsheet)"
        elif not (row["excess_hurt_rate"] >= gc["min_excess_hurt_rate"]
                  and row["excess_hurt_ci_lo"] > 0):
            reason = "does not meet E0 H1 exploratory screening"
        else:
            check = repro[(repro["task"] == task) &
                          repro["log_name"].isin([method, e0["baseline"]["log_name"]])]
            expected = e0["baseline"]["repeats"] + 1
            passed = check["delta"].abs().le(e0["reproduction_tolerance"]).all()
            reason = "" if len(check) == expected and passed else "missing or out-of-tolerance reproduction"
        if reason:
            excluded.append({"task": task, "method": method, "reason": reason})
        else:
            candidates.append(row)
    # Predefined screening order; do not rank on any new E1 result.
    candidates.sort(key=lambda r: (-r["excess_hurt_rate"], r["uplift"], r["task"], r["method"]))
    candidates = candidates[:cfg["max_cells"]]
    if not candidates:
        write_json(base / "screening.json", {"excluded": excluded, "selected": []})
        raise ValueError(f"No eligible E0 cells; inspect {base / 'screening.json'}. "
                         "Resolve reproduction discrepancies before launching E1.")
    fractions = cfg["snapshot_fractions"]
    if not fractions or fractions != sorted(set(fractions)) or not (
        0 < fractions[0] < fractions[-1] < cfg["selection_end"] < 1
    ):
        raise ValueError("Need distinct ordered snapshots and nonempty selection/test splits.")
    # Preparation changes protocol inputs: refuse to overwrite completed/running study.
    if (base / "manifest.json").exists():
        raise ValueError("Manifest already exists; reuse it with run/analyze, or use a new artifacts_dir.")
    cells = []
    for row in candidates:
        task, method = row["task"], row["method"]
        source_dir = run_dir(resolve(e0["runs_dir"]), tag, method, 0)
        source = source_dir / f"{task}_{method}_memory.jsonl"
        if not (source_dir / f"{task}.done").exists():
            raise ValueError(f"Source E0 job is incomplete: {source_dir}/{task}")
        records = read_jsonl(source)
        raw, qids = public_rows(task)
        n = len(raw)
        # E0 should cover the complete public set, never an inner-join subset.
        if int(row["n"]) != n:
            raise ValueError(f"Incomplete E0 pairing for {task}/{method}: {row['n']} != {n}")
        train_end, select_end = math.floor(n * fractions[-1]), math.floor(n * cfg["selection_end"])
        directory = base / "inputs" / task / method
        snapshot_paths = {}
        for fraction in fractions:
            count = math.floor(n * fraction)
            seed = directory / f"k{count}.jsonl"
            prefix = snapshot_prefix(records, qids, count)
            write_jsonl(seed, prefix)
            snapshot_paths[f"k{count}"] = {"path": str(seed), "sha256": digest(seed),
                                         "n_samples": count, "n_records": len(prefix)}
        if len(snapshot_paths) != len(fractions):
            raise ValueError("Dataset too small for distinct snapshots.")
        indices = list(range(train_end, select_end)) + list(range(select_end, n))
        n_selection = select_end - train_end
        if args.smoke:
            indices = list(range(train_end, min(train_end + args.smoke, select_end))) + list(
                range(select_end, min(select_end + args.smoke, n)))
            n_selection = min(args.smoke, select_end - train_end)
        if not 0 < n_selection < len(indices):
            raise ValueError("Empty selection or test split.")
        heldout = [raw[i] for i in indices]
        if task == "APIBench-HF":
            data_path = directory / "data"
            write_jsonl(data_path / "huggingface_eval.json", heldout)
            shutil.copyfile(DATA_OVERRIDES[task] / "huggingface_api.jsonl",
                            data_path / "huggingface_api.jsonl")
            data_files = [data_path / "huggingface_eval.json", data_path / "huggingface_api.jsonl"]
        else:
            data_path = directory / "heldout.jsonl"
            write_jsonl(data_path, heldout)
            data_files = [data_path]
        cell = {
            "task": task, "method": method, "cli_name": methods[method]["cli_name"],
            "n_public": n, "train_end": train_end, "selection_end": select_end,
            "n_selection": n_selection, "qids": [qids[i] for i in indices],
            "data_path": str(data_path), "snapshots": snapshot_paths,
            "latest": f"k{train_end}", "source": str(source), "source_sha256": digest(source),
            "data_hashes": {str(p): digest(p) for p in data_files},
            "e0_screening": row,
        }
        cells.append(cell)
    manifest = {"protocol": "E1a-exploratory-60/20/20-v1", "cfg": cfg, "e0": e0,
                "smoke": args.smoke, "cells": cells,
                "submodule_commit": subprocess.check_output(
                    ["git", "-C", str(resolve(e0["harness"]["seqmem_root"])), "rev-parse", "HEAD"],
                    text=True).strip()}
    write_json(base / "screening.json", {"excluded": excluded, "selected": candidates})
    write_json(base / "manifest.json", manifest)
    print(f"Prepared {len(cells)} cells; frozen manifest: {base / 'manifest.json'}")
    for cell in cells:
        print(f"  {cell['task']}/{cell['method']}: snapshots {list(cell['snapshots'])}, "
              f"selection={cell['n_selection']}, test={len(cell['qids'])-cell['n_selection']}")


def jobs(manifest, base):
    for cell in manifest["cells"]:
        for arm in ["Baseline"] + list(cell["snapshots"]):
            for rep in range(manifest["cfg"]["repeats"]):
                out = base / "outputs" / cell["task"] / cell["method"] / arm / f"rep{rep}"
                yield cell, arm, rep, out


def check_inputs(manifest):
    root = resolve(manifest["e0"]["harness"]["seqmem_root"])
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["submodule_commit"]:
        raise ValueError("SeqMem-Eval revision changed since preparation.")
    for cell in manifest["cells"]:
        hashes = dict(cell["data_hashes"])
        hashes.update({v["path"]: v["sha256"] for v in cell["snapshots"].values()})
        for path, expected in hashes.items():
            if digest(path) != expected:
                raise ValueError(f"Protocol input changed: {path}")


def outcomes(out, cell, arm):
    name = "Baseline" if arm == "Baseline" else cell["method"]
    frame = load_eval_log(eval_log_path(out, cell["task"], name))
    # Pinned runner logs entry['qid'] or local 1-based positions, while tasks
    # use _qid internally. Normalize only after exact count/order validation.
    if len(frame) != len(cell["qids"]):
        raise ValueError(f"Incomplete evaluation: {out}: {len(frame)} != {len(cell['qids'])}")
    ids = frame["qid"].tolist()
    local_ids = list(map(str, range(1, len(frame) + 1)))
    if ids != local_ids and ids != cell["qids"]:
        raise ValueError(f"Unexpected evaluation IDs/order: {out}")
    return frame["correct"].to_numpy(float)


def run(manifest, base, dry_run):
    check_inputs(manifest)
    e0 = manifest["e0"]
    failures = []
    fingerprint = digest(base / "manifest.json")
    for cell, arm, rep, out in jobs(manifest, base):
        done = out / "done.json"
        if done.exists():
            marker = json.loads(done.read_text())
            if marker["manifest_sha256"] != fingerprint:
                raise ValueError(f"Stale completion marker: {done}")
            outcomes(out, cell, arm)
            print(f"[done] {cell['task']}/{cell['method']}/{arm}/rep{rep}")
            continue
        baseline = arm == "Baseline"
        cli = e0["baseline"]["cli_name"] if baseline else cell["cli_name"]
        cmd = build_command(e0, cli, cell["task"], out, baseline, None)
        cmd.remove("--resume")  # --resume disables memory seeding in official runner.
        i = cmd.index("--task-data-override")
        cmd[i + 1] = f"{cell['task']}={cell['data_path']}"
        cmd += ["--no-update-memory"]
        if not baseline:
            cmd += ["--memory-seed-file", cell["snapshots"][arm]["path"]]
        print(f"[run] {cell['task']}/{cell['method']}/{arm}/rep{rep}\n  {' '.join(cmd)}", flush=True)
        if dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        # Failed frozen jobs restart from their fixed seed. No partial logs are analysed.
        with (out / "stdout.log").open("w") as log:
            result = subprocess.run(cmd, cwd=resolve(e0["harness"]["seqmem_root"]),
                                    env=build_env(e0), stdout=log, stderr=subprocess.STDOUT)
        try:
            if result.returncode:
                raise ValueError(f"exit {result.returncode}")
            outcomes(out, cell, arm)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            failures.append(str(out))
            print(f"[fail] {out}: {exc}; see stdout.log", file=sys.stderr)
            continue
        write_json(done, {"manifest_sha256": fingerprint})
    if failures:
        raise ValueError(f"{len(failures)} jobs failed; rerun the same command after resolving errors.")


def analyze(manifest, base, reports):
    check_inputs(manifest)
    cfg = manifest["cfg"]
    all_policies, all_arms, all_pairs, verdicts = [], [], [], []
    fingerprint = digest(base / "manifest.json")
    for cell in manifest["cells"]:
        task, method = cell["task"], cell["method"]
        by_arm = {}
        for c, arm, rep, out in jobs(manifest, base):
            if c is not cell:
                continue
            done = out / "done.json"
            if not done.exists() or json.loads(done.read_text())["manifest_sha256"] != fingerprint:
                raise ValueError(f"Missing valid completed job: {out}")
            values = outcomes(out, cell, arm)
            by_arm.setdefault(arm, []).append(values)
            all_pairs.extend({"task": task, "method": method, "arm": arm, "rep": rep,
                              "qid": qid, "split": "selection" if i < cell["n_selection"] else "test",
                              "correct": int(y)}
                             for i, (qid, y) in enumerate(zip(cell["qids"], values)))
        frame = pd.DataFrame({a: np.mean(v, axis=0) for a, v in by_arm.items()})
        cut = cell["n_selection"]
        selection, test = frame.iloc[:cut], frame.iloc[cut:]
        policies, ceilings = compare_snapshots(selection, test, cell["latest"],
                                               cfg["n_bootstrap"], cfg["alpha"], cfg["seed"])
        all_policies.append(policies.assign(task=task, method=method))
        for arm in frame:
            all_arms.append({"task": task, "method": method, "arm": arm,
                             "selection_accuracy": selection[arm].mean(),
                             "test_accuracy": test[arm].mean()})
        chosen = policies[policies["policy"] == "validation_selected_memory"].iloc[0]
        # One prespecified selected-memory vs latest test per cell. Repeat-averaged
        # pair differences retain {-1,-.5,0,.5,1}; use a bounded-mean Hoeffding test,
        # rather than treating repeats of the same item as independent samples.
        d = (test[chosen["arm"]] - test[cell["latest"]]).to_numpy()
        p = min(1.0, float(np.exp(-len(d) * max(float(d.mean()), 0.0) ** 2 / 2)))
        verdicts.append({"task": task, "method": method, **ceilings,
                         "selected_memory": chosen["arm"],
                         "recovery": chosen["gain_vs_latest"],
                         "ci_lo": chosen["gain_vs_latest_ci_lo"],
                         "p_bounded_mean": p, "n_test": len(test)})
    verdict_df = pd.DataFrame(verdicts)
    # Holm correction across all prespecified cells (conservative fixed-time check).
    order = np.argsort(verdict_df["p_bounded_mean"].to_numpy())
    adjusted, running = np.zeros(len(order)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * verdict_df.iloc[idx]["p_bounded_mean"])
        adjusted[idx] = min(1.0, running)
    verdict_df["p_holm"] = adjusted
    verdict_df["promising_exploratory"] = (
        (verdict_df["recovery"] >= cfg["min_recovery"]) &
        (verdict_df["ci_lo"] > 0) &
        (verdict_df["selected_memory"] != [
            c["latest"] for c in manifest["cells"]])
    )
    reports.mkdir(parents=True, exist_ok=True)
    pd.concat(all_policies, ignore_index=True).to_csv(reports / "policies.csv", index=False)
    pd.DataFrame(all_arms).to_csv(reports / "arms.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(reports / "paired_outcomes.csv", index=False)
    verdict_df.to_csv(reports / "recovery.csv", index=False)
    lines = [
        "# E1a frozen snapshot feasibility (exploratory)", "",
        f"Promising cells: {int(verdict_df['promising_exploratory'].sum())}/{len(verdict_df)}.",
        "A promising cell selects an older snapshot on selection items and gains at least "
        f"{100*cfg['min_recovery']:g} points on test, with a positive paired bootstrap lower bound.",
        "This is a screening signal, not a confirmatory claim or an anytime-valid guarantee.", "",
        "## Limits",
        "- E0 inspected all public items; cell screening is exploratory. New data/order/trajectory is needed for confirmation.",
        "- Only prefixes through 60% seed memory; selection (60–80%) and test (80–100%) never update it.",
        "- 'Latest' means the 60% deployment checkpoint, NOT the full E0 final memory.",
        "- Two evaluation repeats share one memory-building trajectory; bootstrap units are items averaged over repeats.",
        "- Fixed-time item intervals assume independent item sampling conditional on the trajectory; "
        "they do not prove robustness to adaptive updates or task drift.",
        "- Test-set best arm/per-item oracle are optimistic descriptive ceilings, not deployable policies.",
        "- Hoeffding/Holm p-values are conservative secondary checks; a promising cell may not pass them.",
        "- Small HumanEval test slices have little power; negative results do not establish equivalence.",
        "- This study evaluates recoverable snapshot differences, not online detection delay or rollback dynamics.", "",
        "## Next decision",
        "If promising cells exist, confirm on new held-out data / independent trajectories before building the monitor.",
        "If only Baseline fallback helps, prioritize memory disabling rather than snapshot selection.",
        "If ceilings are small, investigate harm mechanisms instead of expanding monitoring.", "",
        "See recovery.csv, policies.csv, arms.csv and paired_outcomes.csv.", "",
    ]
    if manifest["smoke"]:
        lines.insert(2, "**SMOKE ONLY — do not use these results as research evidence.**")
    (reports / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"Reports: {reports}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "analyze"])
    parser.add_argument("--config", default="configs/e1_snapshot.yaml")
    parser.add_argument("--smoke", type=int, help="K selection + K test items; isolated artifacts.")
    parser.add_argument("--only-task", action="append")
    parser.add_argument("--only-method", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.smoke is not None and args.smoke <= 0:
        parser.error("--smoke must be positive")
    cfg = load_config(resolve(args.config))
    e0 = load_config(resolve(cfg["e0_config"]))
    suffix = f"_smoke{args.smoke}" if args.smoke else ""
    base = resolve(cfg["artifacts_dir"] + suffix) / model_tag(e0["llm"]["served_model"])
    try:
        if args.action == "prepare":
            prepare(cfg, e0, base, args)
        else:
            if args.only_task or args.only_method:
                raise ValueError("Filters apply only to prepare; run/analyze use the frozen manifest.")
            manifest = json.loads((base / "manifest.json").read_text())
            if args.action == "run":
                run(manifest, base, args.dry_run)
            else:
                analyze(manifest, base, resolve(cfg["reports_dir"] + suffix) /
                        model_tag(e0["llm"]["served_model"]))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
