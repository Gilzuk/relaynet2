"""Unit tests for statistical testing utilities."""

import math
import numpy as np
import pytest

from relaynet.simulation.statistics import (
    compute_confidence_interval,
    wilcoxon_test,
    _norm_ppf,
    _t_ppf,
)


class TestNormPPF:
    def test_median(self):
        assert _norm_ppf(0.5) == pytest.approx(0.0, abs=0.01)

    def test_95th_percentile(self):
        assert _norm_ppf(0.975) == pytest.approx(1.96, abs=0.05)


class TestTPPF:
    def test_large_df_approaches_normal(self):
        # With large df, t-distribution ≈ normal
        assert _t_ppf(0.975, df=1000) == pytest.approx(1.96, abs=0.05)

    def test_small_df(self):
        # df=9: t(0.975) ≈ 2.262
        val = _t_ppf(0.975, df=9)
        assert 2.1 < val < 2.5


class TestConfidenceInterval:
    def test_output_shape(self):
        ber_trials = np.random.rand(5, 10)
        lower, upper = compute_confidence_interval(ber_trials)
        assert lower.shape == (5,)
        assert upper.shape == (5,)

    def test_lower_le_mean_le_upper(self):
        np.random.seed(0)
        ber_trials = np.abs(np.random.randn(4, 20)) * 0.01 + 0.05
        lower, upper = compute_confidence_interval(ber_trials)
        mean = ber_trials.mean(axis=1)
        assert np.all(lower <= mean + 1e-9)
        assert np.all(upper >= mean - 1e-9)

    def test_nonnegative_lower(self):
        ber_trials = np.zeros((3, 10))  # perfect zero BER
        lower, upper = compute_confidence_interval(ber_trials)
        assert np.all(lower >= 0)


class TestWilcoxonTest:
    def test_clear_winner(self):
        np.random.seed(0)
        n_snr, n_trials = 3, 30
        a = np.random.rand(n_snr, n_trials) * 0.01  # very low BER
        b = np.random.rand(n_snr, n_trials) * 0.1   # higher BER
        p_vals, sig, a_wins = wilcoxon_test(a, b)
        assert a_wins.all(), "Method A should win (lower BER)"

    def test_no_difference(self):
        np.random.seed(1)
        data = np.random.rand(3, 20) * 0.05
        p_vals, sig, a_wins = wilcoxon_test(data, data.copy())
        # With identical data no significant difference should be found
        assert not sig.any()

    def test_output_shapes(self):
        a = np.random.rand(5, 15)
        b = np.random.rand(5, 15)
        p, s, w = wilcoxon_test(a, b)
        assert p.shape == (5,)
        assert s.shape == (5,)
        assert w.shape == (5,)
