"""Offline headroom of a per-query gate on a paired table.

A multinomial model predicts {help, hurt, tie} from pre-generation features on
the first part of the stream; on the rest, the gate injects memory iff
P(help) >= P(hurt). Compared against always-off, always-on and the per-sample
oracle. This bounds what gating can buy before any guarantee is added.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gate_headroom(paired: pd.DataFrame, cols: Sequence[str], split: float = 0.6) -> Dict[str, float]:
    cut = int(len(paired) * split)
    train, test = paired.iloc[:cut], paired.iloc[cut:]
    res = {
        "n_test": len(test),
        "acc_always_off": float(test["y0"].mean()) if len(test) else np.nan,
        "acc_always_on": float(test["y1"].mean()) if len(test) else np.nan,
        "acc_oracle": float(np.maximum(test["y0"], test["y1"]).mean()) if len(test) else np.nan,
    }
    classes = set(train["outcome"])
    if len(test) == 0 or not {"help", "hurt"} <= classes:
        res.update(acc_gate=np.nan, gate_on_rate=np.nan, note="train lacks help or hurt samples")
        return res
    model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
    ).fit(train[list(cols)].to_numpy(), train["outcome"].to_numpy())
    proba = model.predict_proba(test[list(cols)].to_numpy())
    idx = {c: i for i, c in enumerate(model.classes_)}
    use_mem = proba[:, idx["help"]] >= proba[:, idx["hurt"]]
    res["acc_gate"] = float(np.where(use_mem, test["y1"], test["y0"]).mean())
    res["gate_on_rate"] = float(use_mem.mean())
    res["note"] = ""
    return res
