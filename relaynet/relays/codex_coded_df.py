"""Explicit CSI-aware QPSK relay paths for new validation experiments."""
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder
from relaynet.coding.codex_matched import (
    CodexMatchedBCJRDecoder, CodexMatchedViterbiDecoder, qpsk_soft_observations,
)
from relaynet.modulation.qpsk import qpsk_modulate


class CodexMatchedCodedRelay:
    """Hard decode/re-encode or BCJR coded-bit posterior-mean forwarding.

    ``process`` requires actual post-equalization per-axis noise variance.
    Soft output uses whole-input normalization, matching the buffered baseline.
    This is not an end-to-end MAP receiver for a soft or learned relay chain.
    """

    def __init__(self, frame_info_bits=200, constraint_length=3, soft=False):
        if not isinstance(frame_info_bits, int) or frame_info_bits <= 0:
            raise ValueError("frame_info_bits must be a positive integer")
        self.encoder = ConvolutionalEncoder(constraint_length)
        self.decoder = (CodexMatchedBCJRDecoder if soft else CodexMatchedViterbiDecoder)(constraint_length)
        self.frame_symbols = frame_info_bits + self.encoder.num_tail
        self.soft = soft

    def process(self, received_signal, *, axis_noise_var):
        y = np.asarray(received_signal)
        observations, variances = qpsk_soft_observations(y, axis_noise_var)
        if not len(y) or len(y) % self.frame_symbols:
            raise ValueError("received_signal must contain complete frames")
        out = []
        stride = 2 * self.frame_symbols
        for start in range(0, len(observations), stride):
            obs, var = observations[start:start + stride], variances[start:start + stride]
            if self.soft:
                p = self.decoder.coded_bit_posteriors(obs, var, symbol_amplitude=1 / np.sqrt(2))
                out.append(((1 - 2 * p[:, 0]) + 1j * (1 - 2 * p[:, 1])) / np.sqrt(2))
            else:
                info = self.decoder.decode(obs, var, symbol_amplitude=1 / np.sqrt(2))
                out.append(qpsk_modulate(self.encoder.encode(info)))
        out = np.concatenate(out)
        if self.soft:
            power = np.mean(np.abs(out) ** 2)
            if power > 0:
                out /= np.sqrt(power)
        return out
