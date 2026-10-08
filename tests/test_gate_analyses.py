import numpy as np
import pandas as pd

from memgate.analysis.headroom import gate_headroom
from memgate.analysis.predictability import leave_one_task_out_auc, time_split_auc
from memgate.features.sample_features import FEATURE_COLS, add_pre_generation_features
from memgate.io.seqmem_logs import load_eval_log
from memgate.monitor.shadow_sim import simulate_shadow_monitor


def _synthetic_paired(n=600, seed=0):
    """Paired table where memory hurts exactly when the feature x is large."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y0 = rng.binomial(1, 0.7, n)
    hurt = (x > 0.8) & (y0 == 1)
    help_ = (x < -0.8) & (y0 == 0)
    y1 = np.where(hurt, 0, np.where(help_, 1, y0))
    d = y1 - y0
    return pd.DataFrame(
        {
            "step": np.arange(n),
            "x": x,
            "noise": rng.normal(size=n),
            "y0": y0,
            "y1": y1,
            "d": d,
            "outcome": np.select([d > 0, d < 0], ["help", "hurt"], "tie"),
        }
    )


def test_time_split_auc_recovers_signal():
    rows = {r["target"]: r for r in time_split_auc(_synthetic_paired(), ["x", "noise"], n_boot=100)}
    assert rows["hurt_vs_rest"]["auc"] > 0.8
    assert rows["hurt_vs_help"]["auc"] > 0.9


def test_leave_one_task_out_runs():
    by_task = {f"t{i}": _synthetic_paired(seed=i) for i in range(3)}
    rows = leave_one_task_out_auc(by_task, ["x", "noise"], n_boot=50)
    assert len(rows) == 3 and all(r["auc"] > 0.8 for r in rows)


def test_gate_headroom_beats_both_fixed_policies():
    res = gate_headroom(_synthetic_paired(n=1500), ["x", "noise"])
    assert res["acc_gate"] > max(res["acc_always_on"], res["acc_always_off"])
    assert res["acc_gate"] <= res["acc_oracle"] + 1e-12


def test_shadow_monitor_full_probing_flags_harm():
    df = _synthetic_paired(n=1500)
    df["d"] = np.where(df["step"] % 4 == 0, -1, 0)
    res = simulate_shadow_monitor(df, eps=1.0, n_sims=3)
    assert res["p_alarm"] == 1.0 and res["mean_probes"] == 1500


def test_features_are_finite(make_log):
    df = add_pre_generation_features(load_eval_log(make_log("m.json", [1, 0, 1, 1], with_memory=True)))
    assert np.isfinite(df[FEATURE_COLS].to_numpy()).all()
    assert df.loc[0, "f_sim_max"] == 0.0 and df.loc[1, "f_sim_max"] > 0.0
