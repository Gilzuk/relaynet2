"""The BCJR detector must reduce to known answers before it is believed.

The withdrawn QPSK result in this thesis came from a benchmark that looked
plausible and was model-mismatched. The defence against repeating that is to
pin the new detector to cases whose answers are known independently, and to
run those before any comparison is measured.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from relaynet.relays.bcjr import BCJRQPSKRelay
from relaynet.relays.viterbi import FadingAwareViterbiQPSKRelay

ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def _syms(rng, n):
    return ALPHABET[rng.integers(0, 4, n)]


def test_posteriors_are_normalised():
    rng = np.random.default_rng(0)
    x = _syms(rng, 200)
    r = BCJRQPSKRelay(channel_taps=[1.0]).set_sigma(sigma=0.3)
    app = r.posteriors(x + 0.3 * (rng.standard_normal(200)
                                  + 1j * rng.standard_normal(200)) / np.sqrt(2))
    assert np.allclose(np.exp(app).sum(axis=1), 1.0)


def test_no_isi_reduces_to_the_per_axis_slicer():
    """With a single tap the channel is memoryless, so the BER-optimal rule is
    the per-axis hard decision. Anything else means the recursion is wrong."""
    rng = np.random.default_rng(1)
    x = _syms(rng, 4000)
    y = x + 0.4 * (rng.standard_normal(4000)
                   + 1j * rng.standard_normal(4000)) / np.sqrt(2)
    out = BCJRQPSKRelay(channel_taps=[1.0], channel_len=1).set_sigma(sigma=0.4).process(y)
    slicer = (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)
    assert np.array_equal(out, slicer)


def test_symbol_map_matches_viterbi_at_high_snr():
    """As noise vanishes both rules must find the transmitted sequence."""
    rng = np.random.default_rng(2)
    h = np.array([0.758, 0.531, 0.379]); h = h / np.linalg.norm(h)
    x = _syms(rng, 800)
    y = np.convolve(x, h)[:x.size] + 1e-3 * rng.standard_normal(x.size)
    v = FadingAwareViterbiQPSKRelay(channel_taps=h).process(y)
    b = BCJRQPSKRelay(channel_taps=h, decision="symbol").set_sigma(sigma=1e-3).process(y)
    assert np.allclose(v, x) and np.allclose(b, x)


def test_posterior_factorises_so_the_two_map_rules_coincide():
    """On this channel bit-MAP and symbol-MAP are the same detector, exactly.

    The taps are real and the per-symbol gain is a real magnitude, so the real
    and imaginary axes never mix: y = g(h*x) + v splits into two independent
    real ISI channels with independent noise. The symbol posterior therefore
    factorises as P(b0)P(b1), and the argmax of the product equals the pair of
    per-bit argmaxes.

    This is stronger than what Chapter 7 currently claims. The chapter measures
    the Gray-map effect as small (1.073 against 1.090 bits per symbol error);
    on this channel the two MAP criteria do not merely nearly agree, they are
    identical. The residual MLSE-vs-MAP question is real, but it is a
    sequence-versus-symbol question, not a Gray-mapping one.
    """
    rng = np.random.default_rng(3)
    h = np.array([0.758, 0.531, 0.379]); h = h / np.linalg.norm(h)
    x = _syms(rng, 3000)
    g = np.abs((rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
               / np.sqrt(2))
    sigma = 0.7
    y = g * np.convolve(x, h)[:x.size] + sigma * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)

    app = np.exp(BCJRQPSKRelay(channel_taps=h).set_sigma(sigma=sigma)
                 .set_gains(g).posteriors(y))
    p0 = app[:, 0] + app[:, 1]           # P(b0 = 0)
    p1 = app[:, 0] + app[:, 2]           # P(b1 = 0)
    product = np.stack([p0 * p1, p0 * (1 - p1),
                        (1 - p0) * p1, (1 - p0) * (1 - p1)], axis=1)
    assert np.abs(app - product).max() < 1e-12, "posterior does not factorise"

    kw = dict(channel_taps=h)
    bit = BCJRQPSKRelay(**kw, decision="bit").set_sigma(sigma=sigma).set_gains(g).process(y)
    sym = BCJRQPSKRelay(**kw, decision="symbol").set_sigma(sigma=sigma).set_gains(g).process(y)
    assert np.array_equal(bit, sym)


def test_gains_are_required_to_match_the_faded_channel():
    """The trap that produced the withdrawn result: taps alone are not genie CSI
    on a channel that also fades per symbol."""
    rng = np.random.default_rng(4)
    h = np.array([0.758, 0.531, 0.379]); h = h / np.linalg.norm(h)
    x = _syms(rng, 2000)
    g = np.abs((rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
               / np.sqrt(2))
    y = g * np.convolve(x, h)[:x.size] + 1e-2 * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
    blind = BCJRQPSKRelay(channel_taps=h).set_sigma(sigma=1e-2).process(y)
    aware = BCJRQPSKRelay(channel_taps=h).set_sigma(sigma=1e-2).set_gains(g).process(y)
    err_blind = np.mean(blind != x)
    err_aware = np.mean(aware != x)
    assert err_aware < err_blind, (err_aware, err_blind)
    assert err_aware < 1e-2, err_aware


def test_sigma_must_be_set():
    with pytest.raises(ValueError):
        BCJRQPSKRelay(channel_taps=[1.0]).process(np.zeros(4, dtype=complex))


def test_bad_decision_rule_is_rejected():
    with pytest.raises(ValueError):
        BCJRQPSKRelay(channel_taps=[1.0], decision="maximum-likelihood")
