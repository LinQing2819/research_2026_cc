import numpy as np

from memgate.monitor.confseq import eb_confseq, paired_diff_confseq


def test_time_uniform_coverage_bernoulli():
    rng = np.random.default_rng(0)
    alpha, mu, n_sims, T = 0.1, 0.3, 300, 400
    misses = 0
    for _ in range(n_sims):
        lo, hi, _ = eb_confseq(rng.binomial(1, mu, T), alpha=alpha)
        misses += np.any((lo > mu) | (hi < mu))
    # Time-uniform miscoverage must stay below alpha (allow Monte Carlo slack).
    assert misses / n_sims <= alpha + 0.03


def test_width_shrinks_and_bounds_ordered():
    rng = np.random.default_rng(1)
    lo, hi, center = eb_confseq(rng.uniform(size=2000))
    assert np.all(lo <= center) and np.all(center <= hi)
    assert (hi - lo)[-1] < (hi - lo)[99] < (hi - lo)[9]


def test_paired_diff_detects_strong_harm():
    d = np.r_[np.zeros(200), -np.ones(100)]
    rng = np.random.default_rng(2)
    rng.shuffle(d)
    lo, hi, _ = paired_diff_confseq(d)
    assert hi[-1] < 0
    assert -1 <= lo[-1] <= hi[-1] <= 1
