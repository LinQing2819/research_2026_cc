"""Anytime-valid confidence sequences for bounded streams.

Predictable-plug-in empirical-Bernstein (PrPl-EB) confidence sequence of
Waudby-Smith & Ramdas, "Estimating means of bounded random variables by
betting", JRSSB 2024, Theorem 2. For X_t in [0, 1] with conditional means
mu_t, the sequence covers the lambda-weighted running average
sum(lambda_i mu_i) / sum(lambda_i) uniformly over time with probability
>= 1 - alpha; for a constant mean this is the mean itself.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _psi_e(lam: np.ndarray) -> np.ndarray:
    return (-np.log1p(-lam) - lam) / 4.0


def eb_confseq(
    x: np.ndarray, alpha: float = 0.05, c: float = 0.5, two_sided: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lower, upper, center) arrays of length len(x) for x in [0, 1]."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        empty = np.array([])
        return empty, empty, empty
    if np.any((x < 0) | (x > 1)):
        raise ValueError("eb_confseq expects observations in [0, 1].")
    t = np.arange(1, x.size + 1, dtype=float)
    log_term = np.log((2.0 if two_sided else 1.0) / alpha)

    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    sigma2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    mu_prev = np.concatenate([[0.5], mu_hat[:-1]])
    sigma2_prev = np.concatenate([[0.25], sigma2[:-1]])

    lam = np.minimum(np.sqrt(2 * log_term / (sigma2_prev * t * np.log1p(t))), c)
    v = 4.0 * (x - mu_prev) ** 2

    lam_sum = np.cumsum(lam)
    center = np.cumsum(lam * x) / lam_sum
    margin = (log_term + np.cumsum(v * _psi_e(lam))) / lam_sum
    return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1), center


def paired_diff_confseq(
    d: np.ndarray, alpha: float = 0.05, c: float = 0.5, two_sided: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence sequence for the mean of paired differences d in [-1, 1]."""
    d = np.asarray(d, dtype=float)
    lo, hi, center = eb_confseq((d + 1) / 2, alpha=alpha, c=c, two_sided=two_sided)
    return 2 * lo - 1, 2 * hi - 1, 2 * center - 1
