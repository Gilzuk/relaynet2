"""Coded Decode-and-Forward relay (information-theoretic / block DF).

Unlike :class:`DecodeAndForwardRelay` (symbol-wise hard slicing, the DF
baseline used everywhere else in this thesis), this relay implements the
*block* DF described in Remark~rem:df-terminology: it decodes an entire
outer-coded frame via :class:`relaynet.coding.convolutional.ViterbiCodeDecoder`,
then re-encodes and re-modulates the recovered information bits before
forwarding. It exists to make the "reported DF results should not be
read as bounds on coded block-DF performance" caveat measurable rather
than only asserted.
"""

import numpy as np

from .base import Relay
from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.modulation.qpsk import qpsk_modulate


class CodedDecodeAndForwardRelay(Relay):
    """Block DF over a rate-1/2 convolutional code on QPSK.

    Each frame is `frame_info_bits` information bits, coded and mapped
    one-to-one onto QPSK symbols (each trellis step is exactly one
    Gray-coded QPSK symbol, so no bit/symbol reshaping is needed). The
    relay decodes the frame using the noisy hop-1 observation, then
    re-encodes and re-modulates from the decoded bits -- a fresh, clean
    codeword -- before forwarding, exactly as information-theoretic DF
    requires.

    Parameters
    ----------
    frame_info_bits : int, optional
        Information bits per frame (default 100).
    constraint_length : int, optional
        Passed through to the encoder/decoder (default 3). One of
        {3, 5, 7} -- see :data:`relaynet.coding.convolutional.STANDARD_GENERATORS`.
    """

    def __init__(self, frame_info_bits=100, constraint_length=3):
        self.frame_info_bits = frame_info_bits
        self.encoder = ConvolutionalEncoder(constraint_length=constraint_length)
        self.decoder = ViterbiCodeDecoder(constraint_length=constraint_length)
        self.frame_symbols = frame_info_bits + self.decoder.num_tail

    def process(self, received_signal):
        received_signal = np.asarray(received_signal)
        n_frames = len(received_signal) // self.frame_symbols
        usable = n_frames * self.frame_symbols

        output = np.empty(usable, dtype=complex)
        for f in range(n_frames):
            seg = received_signal[f * self.frame_symbols:(f + 1) * self.frame_symbols]
            soft = np.empty(2 * self.frame_symbols, dtype=float)
            soft[0::2] = seg.real
            soft[1::2] = seg.imag

            info_hat = self.decoder.decode(soft)
            coded_hat = self.encoder.encode(info_hat)
            # qpsk_modulate is unit-power by construction (Section~sec:modulation),
            # so no post-hoc power normalization is needed here.
            output[f * self.frame_symbols:(f + 1) * self.frame_symbols] = qpsk_modulate(coded_hat)

        return output
