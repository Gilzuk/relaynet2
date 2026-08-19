"""Rate-1/2 convolutional codes: encoder and a soft-decision Viterbi
decoder for their trellis, generalized across constraint length K.

This is a different trellis from :mod:`relaynet.relays.viterbi`, which
decodes *channel*-induced ISI memory. Here the memory is introduced by
the code itself, applied on the canonical memoryless channel: no channel
taps are involved, and the trellis has exactly 2**(K-1) states regardless
of channel condition.

Supported constraint lengths and their standard maximal-free-distance
generator polynomials (rate 1/2), as tabulated in, e.g., Proakis & Salehi
\\cite{ProakisSalehi2008DigitalComm}:

    K=3  (4 states,  d_free=5):  g1 = 7 (octal),   g2 = 5 (octal)
    K=5  (16 states, d_free=7):  g1 = 23 (octal),  g2 = 35 (octal)
    K=7  (64 states, d_free=10): g1 = 171 (octal),  g2 = 133 (octal) -- the
        NASA/Voyager-era de facto standard.

K=3 is the code used by the original coded-DF baseline; K=5 and K=7 are
stronger, larger-trellis codes swept to show how code strength shifts
the low-SNR threshold effect. Frames are zero-tail terminated: K-1 zero
bits appended to the information block flush the encoder register back
to state 0, so every frame is independently decodable.

Bit-to-symbol convention matches :mod:`relaynet.modulation.qpsk` and
:mod:`relaynet.modulation.bpsk` exactly: 0 -> +1, 1 -> -1. Each trellis
step emits exactly 2 coded bits (out1, out2), which is exactly one
Gray-coded QPSK symbol (out1 -> I, out2 -> Q) with no reshaping needed
(the QAM16 case packs 2 trellis steps, i.e. 4 coded bits, per symbol --
see relaynet.relays.coded_df for the modulation-specific packing).
"""

import numpy as np

# (g1, g2) in octal, MSB first, length-K binary once expanded: bit 0 of
# the window is the current input bit u, bit j (j>=1) is b_{i-j}.
STANDARD_GENERATORS = {
    3: (0o7, 0o5),
    5: (0o23, 0o35),
    7: (0o171, 0o133),
}


def _generator_taps(g_octal, K):
    """Expand an octal generator into a length-K boolean tap mask, MSB first."""
    return [(g_octal >> (K - 1 - i)) & 1 for i in range(K)]


class ConvolutionalEncoder:
    """Rate-1/2 convolutional encoder with zero-tail termination.

    Parameters
    ----------
    constraint_length : int, optional
        K, the encoder memory depth including the current bit (default 3,
        giving 2**(K-1) states). One of {3, 5, 7} -- see
        :data:`STANDARD_GENERATORS`.
    """

    RATE = 0.5

    def __init__(self, constraint_length=3):
        if constraint_length not in STANDARD_GENERATORS:
            raise NotImplementedError(
                f"constraint_length={constraint_length} not supported; "
                f"choose one of {sorted(STANDARD_GENERATORS)}."
            )
        self.K = constraint_length
        self.num_tail = self.K - 1
        g1, g2 = STANDARD_GENERATORS[self.K]
        self.taps1 = _generator_taps(g1, self.K)
        self.taps2 = _generator_taps(g2, self.K)

    def encode(self, info_bits):
        """Encode one frame of information bits.

        Parameters
        ----------
        info_bits : array-like of {0,1}
            Information bits for one frame.

        Returns
        -------
        coded_bits : ndarray of {0,1}, length 2*(len(info_bits) + K - 1)
            Coded bits (out1, out2, out1, out2, ...) including the
            zero-tail termination.
        """
        info_bits = np.asarray(info_bits, dtype=int)
        padded = np.concatenate([info_bits, np.zeros(self.num_tail, dtype=int)])

        reg = [0] * (self.K - 1)  # reg[0] = b_{i-1} (newest) ... reg[-1] = b_{i-K+1}
        out1 = np.empty(len(padded), dtype=int)
        out2 = np.empty(len(padded), dtype=int)
        for i, u in enumerate(padded):
            window = [int(u)] + reg  # length K: [u, b_{i-1}, ..., b_{i-K+1}]
            out1[i] = np.bitwise_xor.reduce([w for w, t in zip(window, self.taps1) if t])
            out2[i] = np.bitwise_xor.reduce([w for w, t in zip(window, self.taps2) if t])
            reg = [int(u)] + reg[:-1]

        coded = np.empty(2 * len(padded), dtype=int)
        coded[0::2] = out1
        coded[1::2] = out2
        return coded

    def n_info_bits(self, n_coded_bits):
        """Number of information bits carried by a frame of coded bits."""
        n_padded = n_coded_bits // 2
        return n_padded - self.num_tail

    def n_coded_bits(self, n_info_bits):
        """Number of coded bits (including tail) for a frame of info bits."""
        return 2 * (n_info_bits + self.num_tail)


class ViterbiCodeDecoder:
    """Soft-decision Viterbi decoder for the :class:`ConvolutionalEncoder` trellis.

    Operates directly on real-valued soft samples (e.g. the I/Q components
    of a received QPSK symbol after coherent compensation, each an
    independent noisy +-1 observation of one coded bit), using squared
    Euclidean distance as the branch metric -- the same style of metric
    already used by :class:`relaynet.relays.viterbi.ViterbiMLSERelay`.

    Parameters
    ----------
    constraint_length : int, optional
        Must match the encoder's (default 3). One of {3, 5, 7}.
    """

    def __init__(self, constraint_length=3):
        if constraint_length not in STANDARD_GENERATORS:
            raise NotImplementedError(
                f"constraint_length={constraint_length} not supported; "
                f"choose one of {sorted(STANDARD_GENERATORS)}."
            )
        self.K = constraint_length
        self.num_tail = self.K - 1
        self.num_states = 2 ** (self.K - 1)
        g1, g2 = STANDARD_GENERATORS[self.K]
        self.taps1 = _generator_taps(g1, self.K)
        self.taps2 = _generator_taps(g2, self.K)
        self._build_trellis()

    def _state_bits(self, s):
        """Unpack state integer s into [b_{i-1}, ..., b_{i-K+1}], MSB first."""
        return [(s >> (self.K - 2 - j)) & 1 for j in range(self.K - 1)]

    def _build_trellis(self):
        self.nxt = np.zeros((self.num_states, 2), dtype=np.int32)
        out_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)  # [state, u, (out1,out2)]
        for s in range(self.num_states):
            reg = self._state_bits(s)  # [b_{i-1}, ..., b_{i-K+1}]
            for u in (0, 1):
                window = [u] + reg
                out1 = 0
                for w, t in zip(window, self.taps1):
                    if t:
                        out1 ^= w
                out2 = 0
                for w, t in zip(window, self.taps2):
                    if t:
                        out2 ^= w
                # Generic shift-register state update: new state packs u in
                # as the new most-significant bit, dropping the oldest bit.
                ns = (u << (self.K - 2)) | (s >> 1) if self.K > 1 else 0
                self.nxt[s, u] = ns
                out_bits[s, u] = (out1, out2)
        # Expected soft symbol per coded bit: 0 -> +1, 1 -> -1 (matches
        # relaynet.modulation.qpsk / bpsk convention).
        self.exp_symbol = 1.0 - 2.0 * out_bits.astype(float)

        # Predecessor lookup for a fully vectorized decode step. Every next
        # state ns has exactly 2 predecessors, which always share the same
        # input bit u = ns >> (K-2) and differ only in the dropped (LSB)
        # bit of the predecessor state -- a direct consequence of the
        # generic shift-register update ns = (u << (K-2)) | (s >> 1).
        mask = (1 << (self.K - 2)) - 1 if self.K > 2 else 0
        self.pred_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.pred_u = np.zeros(self.num_states, dtype=np.int32)
        for ns in range(self.num_states):
            base = ns & mask
            self.pred_state[ns, 0] = 2 * base
            self.pred_state[ns, 1] = 2 * base + 1
            self.pred_u[ns] = ns >> (self.K - 2) if self.K > 1 else 0

    def decode(self, soft_bits):
        """Decode one frame of soft coded-bit observations.

        Parameters
        ----------
        soft_bits : array-like of float, length 2*(n_info + K - 1)
            Real-valued soft observations of the coded bits (sign carries
            the hard decision; magnitude, the confidence), interleaved
            (out1, out2, out1, out2, ...) as produced by
            :meth:`ConvolutionalEncoder.encode`.

        Returns
        -------
        info_bits : ndarray of {0,1}
            Decoded information bits (tail bits stripped).
        """
        soft_bits = np.asarray(soft_bits, dtype=float)
        n_steps = len(soft_bits) // 2
        y1 = soft_bits[0::2]
        y2 = soft_bits[1::2]

        metric = np.full(self.num_states, np.inf)
        metric[0] = 0.0  # trellis starts in the all-zero state
        bp_state = np.zeros((n_steps, self.num_states), dtype=np.int32)
        bp_input = np.zeros((n_steps, self.num_states), dtype=np.int32)

        pred0, pred1 = self.pred_state[:, 0], self.pred_state[:, 1]
        pred_u = self.pred_u
        for i in range(n_steps):
            branch = (y1[i] - self.exp_symbol[:, :, 0]) ** 2 + (y2[i] - self.exp_symbol[:, :, 1]) ** 2

            # Fully vectorized ACS (add-compare-select): every next state has
            # exactly 2 predecessors sharing one input bit (see _build_trellis).
            cost0 = metric[pred0] + branch[pred0, pred_u]
            cost1 = metric[pred1] + branch[pred1, pred_u]
            take1 = cost1 < cost0

            metric = np.where(take1, cost1, cost0)
            bp_state[i] = np.where(take1, pred1, pred0)
            bp_input[i] = pred_u

        # Zero-tail termination: the trellis is known to end in state 0.
        s = 0
        decoded = np.empty(n_steps, dtype=int)
        for i in range(n_steps - 1, -1, -1):
            decoded[i] = bp_input[i, s]
            s = bp_state[i, s]

        return decoded[: n_steps - self.num_tail]
