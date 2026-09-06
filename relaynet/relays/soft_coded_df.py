"""Soft-decision relays for the coded two-hop link.

Both relays here forward *soft* symbols -- posterior means rather than
hard constellation points -- so the destination's decoder still receives
the relay's confidence, not just its best guess.

This addresses the specific liability the hard relays have. Hard block-DF
(:class:`relaynet.relays.coded_df.CodedDecodeAndForwardRelay`) decodes and
re-encodes, so a wrong decode puts a different but perfectly valid codeword
on the air and the destination cannot repair it. The hard learned relay
(:class:`relaynet.relays.mlp.MLPQPSKClassifierRelay`) at least leaves
isolated symbol errors the destination can fix, but it still throws away
how sure it was. Forwarding a posterior mean keeps that information: a
symbol the relay is unsure about is shrunk toward the origin, which the
destination's squared-distance metric reads as weak evidence rather than
as a confident wrong answer.

Both classes normalize output to unit average power, matching the power
convention every other relay in this package uses, so the comparison is
at equal transmit power.
"""

import numpy as np

from .base import Relay
from relaynet.coding.bcjr import BCJRCodeDecoder

# Gray-coded QPSK alphabet, index-for-index identical to
# relaynet.modulation.qpsk and MLPQPSKClassifierRelay.ALPHABET.
ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def _unit_power(x, target_power=1.0):
    p = np.mean(np.abs(x) ** 2)
    return x * np.sqrt(target_power / p) if p > 0 else x


class SoftCodedDecodeAndForwardRelay(Relay):
    """Soft block-DF: BCJR at the relay, posterior-mean symbols forwarded.

    Uses the code's redundancy to clean up hop 1 exactly as hard block-DF
    does, but never commits to a codeword. For each trellis step the BCJR
    posteriors on the two coded bits give a posterior-mean symbol

        E[x] = (1 - 2 P(b_I=1)) + j (1 - 2 P(b_Q=1))

    up to normalization -- confident symbols land near a constellation
    point, unsure ones shrink toward the origin.

    Parameters
    ----------
    frame_info_bits : int, optional
        Information bits per frame (default 200).
    constraint_length : int, optional
        One of {3, 5, 7} (default 3).
    noise_var : float, optional
        Per-real-dimension noise variance the BCJR metric assumes. The relay
        is assumed to know the nominal operating SNR only, not per-symbol
        CSI; :meth:`set_snr_db` sets this from an SNR in dB.
    target_power : float, optional
        Output power normalization (default 1.0).
    """

    def __init__(self, frame_info_bits=200, constraint_length=3,
                 noise_var=1.0, target_power=1.0):
        self.frame_info_bits = frame_info_bits
        self.decoder = BCJRCodeDecoder(constraint_length=constraint_length,
                                       noise_var=noise_var)
        self.frame_symbols = frame_info_bits + self.decoder.num_tail
        self.target_power = target_power

    def set_snr_db(self, snr_db):
        """Set the assumed noise variance from the nominal per-hop SNR.

        Matches relaynet.channels.fading.rayleigh_fading_channel: unit signal
        power, noise_power = 1/snr_linear split across two real dimensions,
        so the per-real-dimension variance is 1/(2*snr_linear). This is the
        pre-equalization nominal value -- the relay does not see |h|, so it
        cannot use the true per-symbol effective SNR, and this is what a
        receiver knowing only the operating point would assume.
        """
        self.decoder.set_noise_var(1.0 / (2.0 * 10 ** (snr_db / 10.0)))

    def process(self, received_signal):
        y = np.asarray(received_signal)
        n_frames = len(y) // self.frame_symbols
        usable = n_frames * self.frame_symbols
        out = np.empty(usable, dtype=complex)

        for f in range(n_frames):
            seg = y[f * self.frame_symbols:(f + 1) * self.frame_symbols]
            soft = np.empty(2 * self.frame_symbols, dtype=float)
            soft[0::2] = seg.real
            soft[1::2] = seg.imag

            p1 = self.decoder.coded_bit_posteriors(soft)  # (frame_symbols, 2)
            mean_i = 1.0 - 2.0 * p1[:, 0]
            mean_q = 1.0 - 2.0 * p1[:, 1]
            out[f * self.frame_symbols:(f + 1) * self.frame_symbols] = (
                (mean_i + 1j * mean_q) / np.sqrt(2)
            )

        return _unit_power(out, self.target_power)


class SoftLearnedRelay(Relay):
    """Soft output mode for an already-trained QPSK classifier relay.

    Wraps :class:`relaynet.relays.mlp.MLPQPSKClassifierRelay` (or any object
    exposing the same ``_extract_windows``/``fwd`` pair) and forwards the
    softmax posterior mean over the constellation instead of the ``argmax``
    constellation point. No retraining and no architectural change -- the
    same weights, read out differently -- so any difference measured against
    the hard variant isolates the decision rule alone.
    """

    def __init__(self, classifier, target_power=1.0):
        self.classifier = classifier
        self.target_power = target_power

    def process(self, received_signal):
        windows = self.classifier._extract_windows(received_signal)
        probs = self.classifier.fwd(windows)          # (n, 4)
        soft = probs @ ALPHABET                        # posterior mean
        return _unit_power(soft, self.target_power)
