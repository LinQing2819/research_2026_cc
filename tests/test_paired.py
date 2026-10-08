import numpy as np
import pytest

from memgate.analysis.paired import (
    build_paired,
    difficulty_profile,
    flip_summary,
    temporal_profile,
)
from memgate.io.seqmem_logs import load_eval_log


def test_load_eval_log_parses_memory_fields(make_log):
    df = load_eval_log(make_log("m.json", [1, 0, 1], with_memory=True))
    assert list(df["step"]) == [0, 1, 2]
    assert list(df["correct"]) == [1, 0, 1]
    assert list(df["n_retrieved"]) == [0, 1, 1]
    assert df.loc[1, "retrieved_feedback"] == ["failure"]


def test_flip_counts_and_noise_floor(make_log):
    base0 = load_eval_log(make_log("b0.json", [1, 1, 0, 0, 1, 1]))
    base1 = load_eval_log(make_log("b1.json", [1, 0, 0, 0, 1, 1]))
    mem = load_eval_log(make_log("m.json", [0, 1, 1, 0, 1, 1], with_memory=True))
    paired = build_paired([base0, base1], mem)

    assert list(paired["outcome"]) == ["hurt", "tie", "help", "tie", "tie", "tie"]
    s = flip_summary(paired, n_boot=200)
    assert s["hurt_rate"] == pytest.approx(1 / 6)
    assert s["help_rate"] == pytest.approx(1 / 6)
    assert s["noise_flip_rate"] == pytest.approx(1 / 6)
    assert s["excess_hurt_rate"] == pytest.approx(0.0)
    assert s["mcnemar_p"] == pytest.approx(1.0)


def test_pairing_follows_method_order_and_drops_unmatched(make_log):
    base = load_eval_log(make_log("b0.json", [1, 0, 1]))
    mem = load_eval_log(make_log("m.json", [1, 1], with_memory=True))
    paired = build_paired([base], mem)
    assert list(paired["qid"]) == ["q0", "q1"]
    assert "noise_flip_rate" in flip_summary(paired, n_boot=50)
    assert np.isnan(flip_summary(paired, n_boot=50)["excess_hurt_rate"])


def test_temporal_and_difficulty_profiles(make_log):
    n = 50
    base0 = load_eval_log(make_log("b0.json", [1] * n))
    base1 = load_eval_log(make_log("b1.json", [1] * 25 + [0] * 25))
    mem = load_eval_log(make_log("m.json", [1] * 40 + [0] * 10, with_memory=True))
    paired = build_paired([base0, base1], mem)

    temporal = temporal_profile(paired, n_windows=5)
    assert len(temporal) == 5
    assert temporal["hurt_rate"].iloc[-1] == pytest.approx(1.0)
    assert temporal["hurt_rate"].iloc[0] == pytest.approx(0.0)

    diff = difficulty_profile(paired).set_index("base_solve_rate_other_reps")
    assert diff.loc[0.0, "hurt_rate"] == pytest.approx(10 / 25)
    assert diff.loc[1.0, "hurt_rate"] == pytest.approx(0.0)
