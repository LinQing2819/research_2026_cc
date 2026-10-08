"""Paired per-sample comparison of a memory method against no-memory runs.

Notation: ``y1`` is the memory-arm outcome, ``y0`` the outcome of baseline
repetition 0 on the same query. Further baseline repetitions are used only to
estimate how often two no-memory runs disagree (sampling-noise floor) and as
an outcome-independent difficulty proxy.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def build_paired(baselines: List[pd.DataFrame], method: pd.DataFrame) -> pd.DataFrame:
    """Join a method log with >=1 baseline logs on qid, keeping method stream order."""
    if not baselines:
        raise ValueError("At least one baseline run is required.")
    paired = method.rename(columns={"correct": "y1"}).copy()
    for r, base in enumerate(baselines):
        paired = paired.merge(
            base[["qid", "correct"]].rename(columns={"correct": f"y0_rep{r}"}),
            on="qid",
            how="inner",
        )
    paired = paired.sort_values("step").reset_index(drop=True)
    paired["y0"] = paired["y0_rep0"]
    paired["d"] = paired["y1"] - paired["y0"]
    paired["outcome"] = np.select(
        [paired["d"] > 0, paired["d"] < 0], ["help", "hurt"], default="tie"
    )
    paired.attrs["n_baseline_reps"] = len(baselines)
    return paired


def _bootstrap_ci(values: np.ndarray, stat, n_boot: int, rng, level=0.95):
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(values[i]) for i in idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def noise_flip_rate(paired: pd.DataFrame) -> Optional[float]:
    """P(rep0 correct, rep1 wrong) between two no-memory runs, or None if unavailable."""
    if "y0_rep1" not in paired:
        return None
    return float(((paired["y0_rep0"] == 1) & (paired["y0_rep1"] == 0)).mean())


def flip_summary(paired: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y0 = paired["y0"].to_numpy()
    y1 = paired["y1"].to_numpy()
    help_ = (y1 == 1) & (y0 == 0)
    hurt = (y1 == 0) & (y0 == 1)
    n_help, n_hurt = int(help_.sum()), int(hurt.sum())
    n_disc = n_help + n_hurt
    mcnemar_p = (
        stats.binomtest(min(n_help, n_hurt), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    )
    noise = noise_flip_rate(paired)

    out = {
        "n": len(paired),
        "acc_base": float(y0.mean()),
        "acc_mem": float(y1.mean()),
        "uplift": float((y1 - y0).mean()),
        "help_rate": float(help_.mean()),
        "hurt_rate": float(hurt.mean()),
        "discordant_rate": float(n_disc / max(len(paired), 1)),
        "mcnemar_p": float(mcnemar_p),
        "noise_flip_rate": np.nan if noise is None else noise,
    }
    out["uplift_ci_lo"], out["uplift_ci_hi"] = _bootstrap_ci(
        (y1 - y0).astype(float), np.mean, n_boot, rng
    )
    if noise is not None:
        # Per-sample contribution to (hurt - noise flip); its mean is the excess hurt rate.
        excess = hurt.astype(float) - (
            (paired["y0_rep0"] == 1) & (paired["y0_rep1"] == 0)
        ).to_numpy(float)
        out["excess_hurt_rate"] = float(excess.mean())
        out["excess_hurt_ci_lo"], out["excess_hurt_ci_hi"] = _bootstrap_ci(
            excess, np.mean, n_boot, rng
        )
    else:
        out["excess_hurt_rate"] = out["excess_hurt_ci_lo"] = out["excess_hurt_ci_hi"] = np.nan
    return out


def temporal_profile(paired: pd.DataFrame, n_windows: int = 5) -> pd.DataFrame:
    """Accuracy and flip rates in consecutive equal-size windows of the stream."""
    rows = []
    for w, idx in enumerate(np.array_split(np.arange(len(paired)), n_windows)):
        chunk = paired.iloc[idx]
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "window": w,
                "step_start": int(chunk["step"].iloc[0]),
                "step_end": int(chunk["step"].iloc[-1]),
                "n": len(chunk),
                "acc_base": chunk["y0"].mean(),
                "acc_mem": chunk["y1"].mean(),
                "uplift": chunk["d"].mean(),
                "help_rate": (chunk["outcome"] == "help").mean(),
                "hurt_rate": (chunk["outcome"] == "hurt").mean(),
            }
        )
    return pd.DataFrame(rows)


def uplift_trend(paired: pd.DataFrame) -> Dict[str, float]:
    """Spearman correlation between stream position and per-sample uplift."""
    if paired["d"].nunique() < 2:
        return {"spearman_rho": np.nan, "spearman_p": np.nan}
    rho, p = stats.spearmanr(paired["step"], paired["d"])
    return {"spearman_rho": float(rho), "spearman_p": float(p)}


def difficulty_profile(paired: pd.DataFrame) -> pd.DataFrame:
    """Flip rates stratified by difficulty measured on baseline reps other than rep0.

    Using rep0 itself as the difficulty proxy would make "hurt only happens on
    easy items" true by construction, so it is excluded.
    """
    other = [c for c in paired.columns if c.startswith("y0_rep") and c != "y0_rep0"]
    if not other:
        return pd.DataFrame()
    df = paired.assign(base_solve_rate=paired[other].mean(axis=1))
    rows = []
    for level, g in df.groupby("base_solve_rate"):
        rows.append(
            {
                "base_solve_rate_other_reps": level,
                "n": len(g),
                "help_rate": (g["outcome"] == "help").mean(),
                "hurt_rate": (g["outcome"] == "hurt").mean(),
                "uplift": g["d"].mean(),
            }
        )
    return pd.DataFrame(rows)
