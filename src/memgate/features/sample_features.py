"""Pre-generation features for predicting whether memory will help or hurt.

Only information available *before* the memory-arm answer is generated is used
(query, what was retrieved, stream position), so a gate built on these
features could run online. Text similarity uses a stateless hashing vectorizer
so no statistics leak from later stream positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

FEATURE_COLS = [
    "f_step_frac",
    "f_log_step",
    "f_n_retrieved",
    "f_frac_failed_retrieved",
    "f_query_len",
    "f_memory_len",
    "f_sim_max",
    "f_sim_mean",
    "f_sim_memory",
]

_VEC = HashingVectorizer(
    ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm="l2"
)


def _cos_rows(a, b) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def add_pre_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a ``load_eval_log`` frame with ``FEATURE_COLS`` added."""
    out = df.copy()
    n = max(len(out), 1)
    out["f_step_frac"] = out["step"] / n
    out["f_log_step"] = np.log1p(out["step"])
    out["f_n_retrieved"] = out["n_retrieved"].astype(float)
    out["f_frac_failed_retrieved"] = [
        (sum(fb.strip().lower() == "failure" for fb in fbs) / len(fbs)) if fbs else 0.0
        for fbs in out["retrieved_feedback"]
    ]
    out["f_query_len"] = np.log1p(out["question"].str.len())
    out["f_memory_len"] = np.log1p(out["memory_text"].str.len())

    q_vec = _VEC.transform(out["question"].tolist())
    out["f_sim_memory"] = _cos_rows(q_vec, _VEC.transform(out["memory_text"].tolist()))

    sim_max, sim_mean = [], []
    for i, rq in enumerate(out["retrieved_questions"]):
        if not rq:
            sim_max.append(0.0)
            sim_mean.append(0.0)
            continue
        sims = (_VEC.transform(rq) @ q_vec[i].T).toarray().ravel()
        sim_max.append(float(sims.max()))
        sim_mean.append(float(sims.mean()))
    out["f_sim_max"] = sim_max
    out["f_sim_mean"] = sim_mean
    return out
