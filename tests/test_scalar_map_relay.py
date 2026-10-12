"""ScalarMAPISIRelay: symbol-by-symbol MAP decision on y[n] over known real ISI."""

import itertools

import numpy as np

from relaynet.relays import ScalarMAPISIRelay

TAPS = np.array([1.0, 0.7, 0.5])


def _noiseless(taps):
    h = taps / np.linalg.norm(taps)
    rows = []
    for x0, x1, x2 in itertools.product((1.0, -1.0), repeat=3):
        rows.append((x0, h[0] * x0 + h[1] * x1 + h[2] * x2))
    return rows


def test_decodes_every_noiseless_pattern_correctly():
    # for these taps the zero-threshold slicer gets one pattern in four wrong;
    # the scalar MAP rule gets none wrong, because the conditional sets are disjoint
    relay = ScalarMAPISIRelay(TAPS, snr_db=30)
    rows = _noiseless(TAPS)
    y = np.array([r[1] for r in rows]); x = np.array([r[0] for r in rows])
    assert np.array_equal(relay.process(y), x)
    assert np.mean(np.sign(y) != x) == 0.25          # the slicer's 0.25 floor


def test_decision_regions_are_not_a_single_threshold():
    relay = ScalarMAPISIRelay(TAPS, snr_db=20)
    assert relay.process([0.15])[0] == -1.0 and relay.process([-0.15])[0] == 1.0
    assert relay.process([0.6])[0] == 1.0 and relay.process([-0.6])[0] == -1.0


def test_llr_is_odd_symmetric():
    relay = ScalarMAPISIRelay(TAPS, snr_db=8)
    y = np.linspace(-2, 2, 101)
    assert np.allclose(relay.log_likelihood_ratio(y), -relay.log_likelihood_ratio(-y))


def test_without_isi_reduces_to_the_zero_threshold_slicer():
    relay = ScalarMAPISIRelay([1.0], snr_db=5)
    y = np.random.default_rng(0).standard_normal(1000)
    assert np.array_equal(relay.process(y), np.where(y >= 0, 1.0, -1.0))
