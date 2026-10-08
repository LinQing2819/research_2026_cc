"""Analyse E0 runs and write the go/no-go report.

Reads SeqMem-Eval eval logs from ``runs/e0/<model_tag>/`` and writes CSV tables,
per-task plots and ``verdict.md`` into ``reports/e0/<model_tag>/``.

    python scripts/e0/analyze.py [--config configs/e0_pilot.yaml] [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from memgate.analysis.headroom import gate_headroom  # noqa: E402
from memgate.analysis.paired import (  # noqa: E402
    build_paired,
    difficulty_profile,
    flip_summary,
    temporal_profile,
    uplift_trend,
)
from memgate.analysis.predictability import leave_one_task_out_auc, time_split_auc  # noqa: E402
from memgate.config import load_config, model_tag, resolve  # noqa: E402
from memgate.features.sample_features import FEATURE_COLS, add_pre_generation_features  # noqa: E402
from memgate.io.seqmem_logs import eval_log_path, load_eval_log, run_dir  # noqa: E402
from memgate.monitor.shadow_sim import simulate_shadow_monitor  # noqa: E402


def _load_baselines(cfg, runs_root, tag, task):
    base = cfg["baseline"]
    frames = []
    for rep in range(base["repeats"]):
        p = eval_log_path(run_dir(runs_root, tag, base["log_name"], rep), task, base["log_name"])
        if p.exists():
            frames.append(load_eval_log(p))
    return frames


def _plot_temporal(temporal: pd.DataFrame, task: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for method, g in temporal.groupby("method"):
        ax.plot(g["window"], g["uplift"], marker="o", label=method)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("stream window")
    ax.set_ylabel("uplift vs. no memory")
    ax.set_title(task)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _verdict(cfg, summary, ts_auc, loto, shadow) -> str:
    gc = cfg["go_criteria"]
    lines = ["# E0 go/no-go verdict", ""]

    h1 = summary[(summary["excess_hurt_rate"] >= gc["min_excess_hurt_rate"]) & (summary["excess_hurt_ci_lo"] > 0)]
    lines += [
        "## H1: memory-induced harm beyond sampling noise",
        f"Cells (task x method) with excess hurt rate >= {gc['min_excess_hurt_rate']:.2f} "
        f"and bootstrap CI above 0: **{len(h1)} / {len(summary)}**.",
        "",
    ]

    valid = ts_auc[(ts_auc["target"] == "hurt_vs_rest") & ts_auc["auc"].notna()]
    loto_valid = loto[loto["auc"].notna()] if not loto.empty else loto
    med_ts = valid["auc"].median() if len(valid) else np.nan
    med_loto = loto_valid["auc"].median() if len(loto_valid) else np.nan
    lines += [
        "## H2: harm is predictable from pre-generation features",
        f"Median time-split AUC (hurt vs. rest): **{med_ts:.3f}** over {len(valid)} cells.",
        f"Median leave-one-task-out AUC: **{med_loto:.3f}** over {len(loto_valid)} cells.",
        f"Threshold: {gc['min_hurt_auc']:.2f}.",
        "",
    ]

    lines += ["## H3: monitor resolution (median final CI half-width on uplift)", ""]
    if not shadow.empty:
        tab = shadow.groupby("eps")["median_final_half_width"].median()
        for eps, hw in tab.items():
            lines.append(f"- eps={eps:g}: {hw:.3f}")
    lines.append("")

    go_h1 = len(h1) > 0
    go_h2 = (not np.isnan(med_ts) and med_ts >= gc["min_hurt_auc"]) or (
        not np.isnan(med_loto) and med_loto >= gc["min_hurt_auc"]
    )
    if go_h1 and go_h2:
        decision = "GO: full gating + monitoring study."
    elif go_h1:
        decision = "PARTIAL: harm exists but is not predictable; pivot to version rollback + monitoring."
    else:
        decision = "NO-GO: no harm beyond noise in these settings."
    lines += ["## Decision", decision, ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/e0_pilot.yaml")
    parser.add_argument("--smoke", action="store_true", help="Analyse runs produced with run_matrix --smoke.")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    an = cfg["analysis"]
    suffix = "_smoke" if args.smoke else ""
    tag = model_tag(cfg["llm"]["served_model"])
    runs_root = resolve(cfg["runs_dir"] + suffix)
    out_dir = resolve(cfg["reports_dir"] + suffix) / tag
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    summaries, temporals, difficulties, ts_rows, headrooms, shadows, repro = [], [], [], [], [], [], []
    paired_by_method = {}

    for task in cfg["tasks"]:
        baselines = _load_baselines(cfg, runs_root, tag, task)
        if not baselines:
            print(f"[skip] {task}: no baseline logs")
            continue
        for rep, b in enumerate(baselines):
            repro.append({"task": task, "log_name": cfg["baseline"]["log_name"], "rep": rep,
                          "n": len(b), "acc": 100 * b["correct"].mean()})
        for m in cfg["methods"]:
            name = m["log_name"]
            p = eval_log_path(run_dir(runs_root, tag, name, 0), task, name)
            if not p.exists():
                print(f"[skip] {task}/{name}: no log")
                continue
            method_df = add_pre_generation_features(load_eval_log(p))
            repro.append({"task": task, "log_name": name, "rep": 0,
                          "n": len(method_df), "acc": 100 * method_df["correct"].mean()})
            paired = build_paired(baselines, method_df)
            paired_by_method.setdefault(name, {})[task] = paired
            key = {"task": task, "method": name}

            summaries.append({**key, **flip_summary(paired, an["n_bootstrap"], an["seed"]), **uplift_trend(paired)})
            temporals.append(temporal_profile(paired, an["n_windows"]).assign(**key))
            diff = difficulty_profile(paired)
            if not diff.empty:
                difficulties.append(diff.assign(**key))
            for r in time_split_auc(paired, FEATURE_COLS, an["time_split"], an["n_bootstrap"] // 2, an["seed"]):
                ts_rows.append({**key, **r})
            headrooms.append({**key, **gate_headroom(paired, FEATURE_COLS, an["time_split"])})
            for eps in an["shadow_eps"]:
                shadows.append({**key, **simulate_shadow_monitor(paired, eps, an["shadow_sims"], an["alpha"], an["seed"])})

    if not summaries:
        print("No paired data found; run scripts/e0/run_matrix.py first.", file=sys.stderr)
        return 1

    loto_rows = []
    for name, by_task in paired_by_method.items():
        for r in leave_one_task_out_auc(by_task, FEATURE_COLS, an["n_bootstrap"] // 2, an["seed"]):
            loto_rows.append({"method": name, **r})

    ref = cfg.get("reference_accuracy", {})
    repro_df = pd.DataFrame(repro)
    repro_df["reference"] = [ref.get(r.log_name, {}).get(r.task, np.nan) for r in repro_df.itertuples()]
    repro_df["delta"] = repro_df["acc"] - repro_df["reference"]
    repro_df["within_tolerance"] = repro_df["delta"].abs() <= cfg.get("reproduction_tolerance", 3.0)

    summary = pd.DataFrame(summaries)
    temporal = pd.concat(temporals, ignore_index=True)
    ts_auc = pd.DataFrame(ts_rows)
    loto = pd.DataFrame(loto_rows)
    shadow = pd.DataFrame(shadows)

    summary.to_csv(out_dir / "flip_summary.csv", index=False)
    temporal.to_csv(out_dir / "temporal.csv", index=False)
    if difficulties:
        pd.concat(difficulties, ignore_index=True).to_csv(out_dir / "difficulty.csv", index=False)
    ts_auc.to_csv(out_dir / "predictability_time_split.csv", index=False)
    loto.to_csv(out_dir / "predictability_loto.csv", index=False)
    pd.DataFrame(headrooms).to_csv(out_dir / "gate_headroom.csv", index=False)
    shadow.to_csv(out_dir / "shadow_monitor.csv", index=False)
    repro_df.to_csv(out_dir / "reproduction.csv", index=False)
    for task, g in temporal.groupby("task"):
        _plot_temporal(g, task, out_dir / "plots" / f"temporal_{task}.png")

    verdict = _verdict(cfg, summary, ts_auc, loto, shadow)
    (out_dir / "verdict.md").write_text(verdict, encoding="utf-8")
    print(verdict)
    print(f"Reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
