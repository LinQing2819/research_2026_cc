"""Snapshot reconstruction and held-out selection for E1a."""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_sample(row: dict) -> bool:
    event = row.get("memory_event")
    return not isinstance(event, dict) or event.get("type") in (None, "sample_observed")


def snapshot_prefix(records: list, qids: list, count: int) -> list:
    """Include all events after sample K, stopping before sample K+1.

    Validate the entire source trajectory, not just a potentially plausible
    prefix. AWM auxiliary workflow events must stay attached to their sample.
    """
    samples = [str(r.get("qid")) for r in records if is_sample(r)]
    if samples != list(map(str, qids)):
        raise ValueError("Memory sample IDs/order do not match the complete public dataset.")
    if not 0 < count <= len(samples):
        raise ValueError("Invalid snapshot size.")
    result, seen = [], 0
    for row in records:
        if is_sample(row):
            if seen == count:
                break
            seen += 1
        result.append(row)
    return result


def compare_snapshots(
    selection: pd.DataFrame, test: pd.DataFrame, latest: str,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 19,
) -> tuple:
    """Choose using selection only; bootstrap paired, repeat-averaged test rows.

    Columns are arms, including ``Baseline``. These are fixed-time exploratory
    intervals across items, conditional on the observed memory trajectory.
    Test-set oracles are deliberately marked as descriptive, biased ceilings.
    """
    arms = list(selection.columns)
    if set(arms) != set(test.columns) or latest not in arms or "Baseline" not in arms:
        raise ValueError("Selection and test must have the same complete arm pool.")
    if selection.empty or test.empty or selection.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing or empty paired outcomes.")
    memories = sorted(a for a in arms if a != "Baseline")
    # Conservative deterministic tie break: prefer latest, then Baseline.
    order = [latest, "Baseline"] + [a for a in memories if a != latest]
    selected = max(order, key=lambda a: selection[a].mean())
    selected_memory = max([latest] + [a for a in memories if a != latest],
                          key=lambda a: selection[a].mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(test), (n_boot, len(test)))
    rows = []
    policies = {
        "latest": latest,
        "no_memory": "Baseline",
        "validation_selected_memory": selected_memory,
        "validation_selected_with_fallback": selected,
    }
    for policy, arm in policies.items():
        values = test[arm].to_numpy()
        row = {"policy": policy, "arm": arm, "n_test": len(test),
               "accuracy": float(values.mean())}
        for ref in [latest, "Baseline"]:
            diff = values - test[ref].to_numpy()
            lo, hi = np.quantile(diff[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2])
            key = "latest" if ref == latest else "baseline"
            row.update({f"gain_vs_{key}": float(diff.mean()),
                        f"gain_vs_{key}_ci_lo": float(lo),
                        f"gain_vs_{key}_ci_hi": float(hi)})
        rows.append(row)
    ceilings = {
        "test_best_fixed_arm_descriptive": float(test.mean().max()),
        "test_per_item_oracle_descriptive": float(test.max(axis=1).mean()),
    }
    return pd.DataFrame(rows), ceilings
