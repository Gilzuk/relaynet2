"""Punctured convolutional codes: higher rates from the rate-1/2 mother code.

Rate adaptation needs more than one rate, and the standard way to get one
is puncturing -- run the rate-1/2 encoder, then delete a fixed periodic
subset of its output bits. The decoder re-inserts the deleted positions as
soft zeros, which is exactly the "no evidence either way" value under the
squared-distance metric of
:mod:`relaynet.coding.convolutional`, so the same Viterbi and BCJR
decoders work unmodified on every rate.

Patterns are the standard maximum-free-distance ones used in IEEE 802.11
and DVB, written as (2 x period) keep/delete masks over the mother code's
(out1, out2) pairs:

    rate 2/3   [[1, 1],      period 2: 4 mother bits -> 3 transmitted
                [1, 0]]
    rate 3/4   [[1, 1, 0],   period 3: 6 mother bits -> 4 transmitted
                [1, 0, 1]]

Rate 1 is not a puncturing of this code -- it means "no coding at all" --
and is handled by the callers as a separate branch rather than faked here.
"""

import numpy as np

from .convolutional import ConvolutionalEncoder

# pattern[j][t] == 1 keeps output j of trellis step (t mod period).
PUNCTURE_PATTERNS = {
    "1/2": np.array([[1], [1]], dtype=bool),
    "2/3": np.array([[1, 1], [1, 0]], dtype=bool),
    "3/4": np.array([[1, 1, 0], [1, 0, 1]], dtype=bool),
}

RATE_VALUE = {"1/2": 0.5, "2/3": 2.0 / 3.0, "3/4": 0.75}


class PuncturedCode:
    """Rate-adaptive wrapper around :class:`ConvolutionalEncoder`.

    Parameters
    ----------
    rate : str
        One of ``"1/2"``, ``"2/3"``, ``"3/4"``.
    constraint_length : int, optional
        Mother-code constraint length (default 3).
    """

    def __init__(self, rate="1/2", constraint_length=3):
        if rate not in PUNCTURE_PATTERNS:
            raise ValueError(f"unsupported rate {rate!r}; "
                             f"choose one of {sorted(PUNCTURE_PATTERNS)}")
        self.rate = rate
        self.pattern = PUNCTURE_PATTERNS[rate]
        self.period = self.pattern.shape[1]
        self.encoder = ConvolutionalEncoder(constraint_length=constraint_length)
        self.num_tail = self.encoder.num_tail

    def _mask(self, n_steps):
        """Boolean keep-mask over the flattened (out1,out2) mother stream."""
        reps = int(np.ceil(n_steps / self.period))
        m = np.tile(self.pattern, (1, reps))[:, :n_steps]   # (2, n_steps)
        return m.T.reshape(-1)                               # interleaved

    def encode(self, info_bits):
        """Encode and puncture one frame."""
        mother = self.encoder.encode(info_bits)              # 2*(n_info+tail)
        n_steps = len(mother) // 2
        return mother[self._mask(n_steps)]

    def depuncture(self, soft_punctured, n_steps):
        """Re-insert deleted positions as soft zeros (no evidence)."""
        mask = self._mask(n_steps)
        full = np.zeros(2 * n_steps, dtype=float)
        full[mask] = soft_punctured[: int(mask.sum())]
        return full

    def n_steps(self, n_info_bits):
        return n_info_bits + self.num_tail

    def n_coded_bits(self, n_info_bits):
        return int(self._mask(self.n_steps(n_info_bits)).sum())
