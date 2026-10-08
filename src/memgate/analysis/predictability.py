"""Is memory-induced harm predictable from pre-generation features? (hypothesis H2)

Two protocols:
* time split within one stream: fit on the first ``split`` fraction, test on the rest;
* leave-one-task-out: fit on all other tasks' streams, test on the held-out task.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_POSITIVES = 5


def _model():
    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
    )


def _auc_with_ci(y: np.ndarray, s: np.ndarray, n_boot: int, rng) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan}
    auc = roc_auc_score(y, s)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
    return {"auc": float(auc), "auc_ci_lo": float(lo), "auc_ci_hi": float(hi)}


def _fit_score(train: pd.DataFrame, test: pd.DataFrame, cols, label, n_boot, rng) -> Dict:
    y_tr, y_te = label(train), label(test)
    res = {
        "n_train": len(train),
        "n_test": len(test),
        "pos_train": int(y_tr.sum()),
        "pos_test": int(y_te.sum()),
    }
    if y_tr.sum() < MIN_POSITIVES or len(np.unique(y_tr)) < 2:
        res.update(auc=np.nan, auc_ci_lo=np.nan, auc_ci_hi=np.nan, note="too few train positives")
        return res
    model = _model().fit(train[cols].to_numpy(), y_tr)
    scores = model.predict_proba(test[cols].to_numpy())[:, 1]
    res.update(_auc_with_ci(y_te, scores, n_boot, rng))
    res["note"] = "" if not np.isnan(res["auc"]) else "single-class test set"
    return res


def _hurt_label(df: pd.DataFrame) -> np.ndarray:
    return (df["outcome"] == "hurt").to_numpy(int)


def time_split_auc(
    paired: pd.DataFrame, cols: Sequence[str], split: float = 0.6, n_boot: int = 1000, seed: int = 0
) -> List[Dict]:
    """AUC for (a) hurt vs. rest on all samples, (b) hurt vs. help on discordant samples."""
    rng = np.random.default_rng(seed)
    cut = int(len(paired) * split)
    train, test = paired.iloc[:cut], paired.iloc[cut:]
    out = [{"target": "hurt_vs_rest", **_fit_score(train, test, cols, _hurt_label, n_boot, rng)}]
    disc_tr = train[train["outcome"] != "tie"]
    disc_te = test[test["outcome"] != "tie"]
    out.append(
        {"target": "hurt_vs_help", **_fit_score(disc_tr, disc_te, cols, _hurt_label, n_boot, rng)}
    )
    return out


def leave_one_task_out_auc(
    paired_by_task: Dict[str, pd.DataFrame], cols: Sequence[str], n_boot: int = 1000, seed: int = 0
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for held_out, test in paired_by_task.items():
        others = [df for t, df in paired_by_task.items() if t != held_out]
        if not others:
            continue
        train = pd.concat(others, ignore_index=True)
        rows.append(
            {"held_out_task": held_out, "target": "hurt_vs_rest",
             **_fit_score(train, test, cols, _hurt_label, n_boot, rng)}
        )
    return rows
