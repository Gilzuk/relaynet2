"""Trellis correctness for the Viterbi MLSE relays.

These check the two properties that make a trellis a trellis, rather than
re-checking a BER against a stored number:

1. With no ISI the detector must reduce to the plain symbol-by-symbol slicer.
   Any trellis bookkeeping error shows up immediately here.
2. On a channel with memory it must never return a path whose metric exceeds
   that of the true transmitted sequence. That is the defining property of a
   maximum-likelihood sequence estimator, and it holds for every noise
   realization, not just on average.

The L=1 cases are regressions: both classes used to raise on a memoryless
channel, because the successor of the single (empty) state was computed as a
one-element tuple that is not a state.
"""

import itertools

import numpy as np
import pytest

from relaynet.relays.viterbi import (FadingAwareViterbiQPSKRelay,
                                     ViterbiMLSEQPSKRelay, ViterbiMLSERelay)

H3 = np.array([1.0, 0.7, 0.5]) / np.linalg.norm([1.0, 0.7, 0.5])
A = ViterbiMLSEQPSKRelay.ALPHABET


def nearest(sym):
    return np.argmin(np.abs(np.asarray(sym)[:, None] - A[None, :]), axis=1)


def test_bpsk_memoryless_reduces_to_slicer():
    rng = np.random.default_rng(0)
    v = ViterbiMLSERelay(channel_taps=[1.0])
    x = 1.0 - 2.0 * rng.integers(0, 2, 5000).astype(float)
    y = x + rng.normal(0, 0.4, 5000)
    assert np.array_equal(np.sign(v.process(y)), np.sign(y))


def test_qpsk_memoryless_reduces_to_slicer():
    rng = np.random.default_rng(0)
    v = ViterbiMLSEQPSKRelay(channel_taps=[1.0])
    x = A[rng.integers(0, 4, 5000)]
    y = x + (rng.normal(0, 0.3, 5000) + 1j * rng.normal(0, 0.3, 5000))
    assert np.array_equal(nearest(v.process(y)), nearest(y))


@pytest.mark.parametrize("snr_db", [0, 6, 12])
def test_qpsk_path_is_at_least_as_likely_as_the_truth(snr_db):
    """The defining ML property, checked per realization rather than on average.

    Scored from index L-1 onward: the first L-1 observations depend on a
    pre-history the decoder never sees, and the transmitter's zero pre-history
    is not a state the trellis can represent, so those terms would compare two
    different models rather than two paths.
    """
    rng = np.random.default_rng(1)
    L = len(H3)
    v = ViterbiMLSEQPSKRelay(channel_taps=H3)

    def metric(seq, obs):
        exp = np.array([np.dot(H3, seq[i - L + 1:i + 1][::-1])
                        for i in range(L - 1, len(seq))])
        return float(np.sum(np.abs(obs[L - 1:] - exp) ** 2))

    for _ in range(5):
        x = A[rng.integers(0, 4, 300)]
        xp = np.concatenate([np.zeros(L - 1, complex), x])
        clean = np.array([np.dot(H3, xp[i:i + L][::-1]) for i in range(len(x))])
        sigma = np.sqrt(1 / (2 * 10 ** (snr_db / 10)))
        obs = clean + (rng.normal(0, sigma, len(x))
                       + 1j * rng.normal(0, sigma, len(x)))
        assert metric(v.process(obs), obs) <= metric(x, obs) + 1e-9


def test_qpsk_trellis_shape_and_transitions():
    v = ViterbiMLSEQPSKRelay(channel_taps=H3)
    assert v.num_states == 4 ** 2
    # every state has exactly M successors and the whole trellis is M-regular
    assert v.nxt.shape == (16, 4)
    assert sorted(np.bincount(v.nxt.ravel(), minlength=16)) == [4] * 16
    # the expected observation must be h0 x[i] + h1 x[i-1] + h2 x[i-2]
    for s, state in enumerate(v.states):
        for u in range(4):
            want = (H3[0] * A[u] + H3[1] * A[state[1]] + H3[2] * A[state[0]])
            assert v.exp_y[s, u] == pytest.approx(want)


def test_alphabet_matches_the_project_qpsk_mapping():
    from relaynet.modulation.qpsk import qpsk_demodulate, qpsk_modulate
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    assert np.allclose(qpsk_modulate(bits), A)
    assert np.array_equal(qpsk_demodulate(A), bits)


def test_fading_aware_matches_parent_without_gains():
    """Unset gains must leave the parent's behaviour untouched."""
    rng = np.random.default_rng(2)
    x = A[rng.integers(0, 4, 500)]
    y = x + (rng.normal(0, 0.3, 500) + 1j * rng.normal(0, 0.3, 500))
    base = ViterbiMLSEQPSKRelay(channel_taps=H3).process(y)
    same = FadingAwareViterbiQPSKRelay(channel_taps=H3).process(y)
    assert np.array_equal(nearest(base), nearest(same))


def test_fading_aware_beats_taps_only_on_a_faded_channel():
    """The point of the class: on g[n]*(h*x)[n] + v[n], taps alone is not CSI."""
    from relaynet.channels.e6_channels import ComplexISIRayleighChannel
    rng = np.random.default_rng(3)
    x = A[rng.integers(0, 4, 20000)]
    ch = ComplexISIRayleighChannel(H3, seed=5)
    y = ch(x, 16)
    taps_only = ViterbiMLSEQPSKRelay(channel_taps=H3).process(y)
    genie = FadingAwareViterbiQPSKRelay(channel_taps=H3)
    genie.set_gains(ch.last_gains)
    ser_taps = float(np.mean(nearest(taps_only) != nearest(x)))
    ser_genie = float(np.mean(nearest(genie.process(y)) != nearest(x)))
    assert ser_genie < 0.5 * ser_taps, (ser_genie, ser_taps)


def test_gain_count_must_match_the_block():
    genie = FadingAwareViterbiQPSKRelay(channel_taps=H3).set_gains(np.ones(10))
    with pytest.raises(ValueError, match="gains"):
        genie.process(np.zeros(11, dtype=complex))
