"""Replay shadow paired probing on a fully observed paired table (hypothesis H3).

At each stream position a probe fires independently with probability ``eps``;
on probed positions both arms are observed, giving a paired difference
``d = y1 - y0``. A confidence sequence over probed differences tells how fast a
deployment-time monitor could certify gain (lower bound > 0) or raise a
regression alarm (upper bound < 0), and how wide it still is at the end.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from memgate.monitor.confseq import paired_diff_confseq


def _first_step(mask: np.ndarray, steps: np.ndarray) -> float:
    hit = np.flatnonzero(mask)
    return float(steps[hit[0]]) if hit.size else np.nan


def simulate_shadow_monitor(
    paired: pd.DataFrame, eps: float, n_sims: int = 200, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    d_all = paired["d"].to_numpy(float)
    steps_all = paired["step"].to_numpy()
    half_widths, gain_steps, alarm_steps, n_probes = [], [], [], []
    for _ in range(n_sims):
        mask = rng.random(len(d_all)) < eps
        d, steps = d_all[mask], steps_all[mask]
        n_probes.append(mask.sum())
        if d.size == 0:
            half_widths.append(np.nan)
            gain_steps.append(np.nan)
            alarm_steps.append(np.nan)
            continue
        lo, hi, _ = paired_diff_confseq(d, alpha=alpha)
        half_widths.append((hi[-1] - lo[-1]) / 2)
        gain_steps.append(_first_step(lo > 0, steps))
        alarm_steps.append(_first_step(hi < 0, steps))
    gain_steps, alarm_steps = np.array(gain_steps), np.array(alarm_steps)
    return {
        "eps": eps,
        "true_uplift": float(d_all.mean()),
        "mean_probes": float(np.mean(n_probes)),
        "median_final_half_width": float(np.nanmedian(half_widths)),
        "p_certified_gain": float(np.mean(~np.isnan(gain_steps))),
        "median_gain_step": float(np.nanmedian(gain_steps)) if np.any(~np.isnan(gain_steps)) else np.nan,
        "p_alarm": float(np.mean(~np.isnan(alarm_steps))),
        "median_alarm_step": float(np.nanmedian(alarm_steps)) if np.any(~np.isnan(alarm_steps)) else np.nan,
    }
