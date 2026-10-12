"""Per-symbol reliability weights in the convolutional Viterbi and BCJR decoders."""

import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.coding.bcjr import BCJRCodeDecoder
from relaynet.modulation.qpsk import qpsk_modulate


def _frame(seed, snr_db=6.0, n_info=200):
    rng = np.random.default_rng(seed)
    enc = ConvolutionalEncoder()
    info = rng.integers(0, 2, n_info)
    x = qpsk_modulate(enc.encode(info))
    h = (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
    sigma = np.sqrt(1 / (2 * 10 ** (snr_db / 10)))
    n = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    y = (h * x + n) / h                      # zero-forced, as in the pipeline
    soft = np.empty(2 * x.size)
    soft[0::2], soft[1::2] = y.real, y.imag
    return info, soft, h


def test_viterbi_default_and_unit_weights_agree():
    dec = ViterbiCodeDecoder()
    _, soft, h = _frame(1)
    assert np.array_equal(dec.decode(soft), dec.decode(soft, weights=np.ones(len(h))))


def test_viterbi_weights_are_scale_invariant():
    dec = ViterbiCodeDecoder()
    _, soft, h = _frame(2)
    w = np.abs(h) ** 2
    assert np.array_equal(dec.decode(soft, weights=w), dec.decode(soft, weights=7.3 * w))


def test_matched_weights_reduce_errors_under_rayleigh():
    dec = ViterbiCodeDecoder()
    plain = matched = 0
    for s in range(40):
        info, soft, h = _frame(100 + s, snr_db=4.0)
        plain += np.sum(dec.decode(soft) != info)
        matched += np.sum(dec.decode(soft, weights=np.abs(h) ** 2) != info)
    assert matched < plain


def test_bcjr_scalar_and_constant_array_variance_agree():
    b = BCJRCodeDecoder(noise_var=0.4)
    _, soft, h = _frame(3)
    p_default = b.coded_bit_posteriors(soft)
    p_array = b.coded_bit_posteriors(soft, noise_var=np.full(len(h), 0.4))
    assert np.allclose(p_default, p_array)
    assert np.array_equal(b.decode(soft), b.decode(soft, noise_var=np.full(len(h), 0.4)))
