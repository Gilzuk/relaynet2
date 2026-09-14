"""Independent enumeration oracle for likelihood and posterior correctness."""
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.coding.codex_matched import (
    CodexMatchedViterbiDecoder, CodexMatchedBCJRDecoder, qpsk_soft_observations,
)
from relaynet.relays.codex_coded_df import CodexMatchedCodedRelay
from relaynet.modulation.qpsk import qpsk_modulate


@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('amplitude', [1.0, 1 / np.sqrt(2)])
def test_matches_exhaustive_codewords(k, amplitude):
    encoder = ConvolutionalEncoder(k)
    info = np.array(list(product([0, 1], repeat=4)))
    codes = np.array([encoder.encode(b) for b in info])
    rng = np.random.default_rng(671)
    var = np.exp(rng.uniform(-4, 3, codes.shape[1]))
    y = amplitude * (1 - 2 * codes[9]) + rng.normal(size=len(var)) * np.sqrt(var)
    score = -np.sum((y - amplitude * (1 - 2 * codes)) ** 2 / (2 * var), axis=1)
    weights = np.exp(score - logsumexp(score))
    vit = CodexMatchedViterbiDecoder(k).decode(y, var, symbol_amplitude=amplitude)
    np.testing.assert_array_equal(vit, info[score.argmax()])
    bcjr = CodexMatchedBCJRDecoder(k)
    np.testing.assert_allclose(bcjr.information_bit_posteriors(y, var, symbol_amplitude=amplitude), weights @ info, atol=1e-12)
    np.testing.assert_allclose(bcjr.coded_bit_posteriors(y, var, symbol_amplitude=amplitude).ravel(), weights @ codes, atol=1e-12)


def test_constant_variance_viterbi_reduces_to_legacy():
    y = np.random.default_rng(81).normal(size=44)
    np.testing.assert_array_equal(CodexMatchedViterbiDecoder().decode(y, 0.7), ViterbiCodeDecoder().decode(y))


def test_pre_and_post_equalization_likelihoods_agree():
    rng = np.random.default_rng(4)
    encoder = ConvolutionalEncoder()
    info = np.array(list(product([0, 1], repeat=3)))
    tx = np.array([qpsk_modulate(encoder.encode(b)) for b in info])
    h = rng.normal(size=5) + 1j * rng.normal(size=5)
    n0 = 0.8
    y = h * tx[3] + np.sqrt(n0 / 2) * (rng.normal(size=5) + 1j * rng.normal(size=5))
    oracle = np.argmin(np.sum(np.abs(y - h * tx) ** 2 / n0, axis=1))
    obs, var = qpsk_soft_observations(y / h, n0 / (2 * np.abs(h) ** 2))
    np.testing.assert_array_equal(CodexMatchedViterbiDecoder().decode(obs, var, symbol_amplitude=1 / np.sqrt(2)), info[oracle])


@pytest.mark.parametrize('soft', [False, True])
def test_matched_relay_clean_frame_and_rejects_truncation(soft):
    relay = CodexMatchedCodedRelay(frame_info_bits=4, soft=soft)
    tx = qpsk_modulate(relay.encoder.encode([1, 0, 0, 1]))
    np.testing.assert_allclose(relay.process(tx, axis_noise_var=1e-6), tx, atol=1e-10)
    with pytest.raises(ValueError):
        relay.process(tx[:-1], axis_noise_var=1)


@pytest.mark.parametrize('var', [0, -1, float('nan'), float('inf'), [1, 2]])
def test_rejects_invalid_variance(var):
    with pytest.raises(ValueError):
        CodexMatchedViterbiDecoder().decode(np.ones(10), var)


def test_log_domain_handles_confident_inconsistent_observations():
    # The zero-tail constraint can force a locally very unlikely transition.
    p = CodexMatchedBCJRDecoder().information_bit_posteriors(np.full(20, -1.0), 1e-8)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))
