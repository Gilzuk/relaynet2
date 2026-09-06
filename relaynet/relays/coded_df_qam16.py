"""Coded Decode-and-Forward relay, 16-QAM variant.

Same block-DF idea as :class:`relaynet.relays.coded_df.CodedDecodeAndForwardRelay`
(decode the full frame, re-encode, re-modulate, forward), but for 16-QAM:
two trellis steps pack into each 16-QAM symbol (4 coded bits/symbol,
Gray-coded jointly per axis -- see
:mod:`relaynet.coding.convolutional_qam16`), so decoding needs the
QAM16-specific branch metric rather than QPSK's independent per-axis one.
"""

import numpy as np

from .base import Relay
from relaynet.coding.convolutional import ConvolutionalEncoder
from relaynet.coding.convolutional_qam16 import QAM16CodeDecoder
from relaynet.modulation.qam import qam16_modulate


class CodedDecodeAndForwardRelayQAM16(Relay):
    """Block DF over a rate-1/2 convolutional code on 16-QAM.

    Parameters
    ----------
    frame_info_bits : int, optional
        Information bits per frame. Combined with the code's tail
        (K-1 bits), (frame_info_bits + K - 1) must be even so the coded
        frame packs into a whole number of 16-QAM symbols (default 200,
        which is even-tailed for K in {3, 5, 7}: tails 2, 4, 6).
    constraint_length : int, optional
        Passed through to the encoder/decoder (default 3). One of
        {3, 5, 7}.
    """

    def __init__(self, frame_info_bits=200, constraint_length=3):
        self.frame_info_bits = frame_info_bits
        self.encoder = ConvolutionalEncoder(constraint_length=constraint_length)
        self.decoder = QAM16CodeDecoder(constraint_length=constraint_length)
        n_steps = frame_info_bits + self.decoder.num_tail
        if n_steps % 2 != 0:
            raise ValueError(
                f"frame_info_bits + tail ({n_steps}) must be even to pack "
                "into whole 16-QAM symbols."
            )
        self.frame_symbols = n_steps // 2  # 2 trellis steps per 16-QAM symbol

    def process(self, received_signal):
        received_signal = np.asarray(received_signal)
        n_frames = len(received_signal) // self.frame_symbols
        usable = n_frames * self.frame_symbols

        output = np.empty(usable, dtype=complex)
        for f in range(n_frames):
            seg = received_signal[f * self.frame_symbols:(f + 1) * self.frame_symbols]
            axis_vals = np.empty(2 * self.frame_symbols, dtype=float)
            axis_vals[0::2] = seg.real
            axis_vals[1::2] = seg.imag

            info_hat = self.decoder.decode(axis_vals)
            coded_hat = self.encoder.encode(info_hat)
            # qam16_modulate is unit-power by construction, so no post-hoc
            # power normalization is needed here (matches CodedDecodeAndForwardRelay).
            output[f * self.frame_symbols:(f + 1) * self.frame_symbols] = qam16_modulate(coded_hat)

        return output
