"""Rate-1/2, constraint-length-3 convolutional code: encoder and a
soft-decision Viterbi decoder for its trellis.

This is a different trellis from :mod:`relaynet.relays.viterbi`, which
decodes *channel*-induced ISI memory. Here the memory is introduced by
the code itself, applied on the canonical memoryless channel: no channel
taps are involved, and the trellis has exactly 2**(K-1) = 4 states
regardless of channel condition.

Generators (K=3, the standard textbook rate-1/2 code, e.g. Proakis &
Salehi \\cite{ProakisSalehi2008DigitalComm}): g1 = 111 (octal 7),
g2 = 101 (octal 5). Frames are zero-tail terminated: K-1 = 2 zero bits
appended to the information block flush the encoder register back to
state 0, so every frame is independently decodable.

Bit-to-symbol convention matches :mod:`relaynet.modulation.qpsk` and
:mod:`relaynet.modulation.bpsk` exactly: 0 -> +1, 1 -> -1. Each trellis
step emits exactly 2 coded bits (out1, out2), which is exactly one
Gray-coded QPSK symbol (out1 -> I, out2 -> Q) with no reshaping needed.
"""

import numpy as np


class ConvolutionalEncoder:
    """Rate-1/2, K=3 convolutional encoder with zero-tail termination.

    Parameters
    ----------
    constraint_length : int, optional
        K, the encoder memory depth including the current bit (default 3,
        giving 2**(K-1) = 4 trellis states). Only K=3 is implemented.
    """

    RATE = 0.5

    def __init__(self, constraint_length=3):
        if constraint_length != 3:
            raise NotImplementedError("Only K=3 (4-state) is implemented.")
        self.K = constraint_length
        self.num_tail = self.K - 1

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

        b1, b2 = 0, 0  # b1 = b_{i-1}, b2 = b_{i-2}
        out1 = np.empty(len(padded), dtype=int)
        out2 = np.empty(len(padded), dtype=int)
        for i, u in enumerate(padded):
            out1[i] = u ^ b1 ^ b2
            out2[i] = u ^ b2
            b1, b2 = int(u), b1

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
    """

    def __init__(self, constraint_length=3):
        if constraint_length != 3:
            raise NotImplementedError("Only K=3 (4-state) is implemented.")
        self.K = constraint_length
        self.num_tail = self.K - 1
        self.num_states = 2 ** (self.K - 1)
        self._build_trellis()

    def _build_trellis(self):
        # State s encodes (b_{i-1}, b_{i-2}) as s = 2*b_{i-1} + b_{i-2}.
        self.nxt = np.zeros((self.num_states, 2), dtype=np.int32)
        out_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)  # [state, u, (out1,out2)]
        for s in range(self.num_states):
            b1 = (s >> 1) & 1
            b2 = s & 1
            for u in (0, 1):
                out1 = u ^ b1 ^ b2
                out2 = u ^ b2
                ns = 2 * u + b1
                self.nxt[s, u] = ns
                out_bits[s, u] = (out1, out2)
        # Expected soft symbol per coded bit: 0 -> +1, 1 -> -1 (matches
        # relaynet.modulation.qpsk / bpsk convention).
        self.exp_symbol = 1.0 - 2.0 * out_bits.astype(float)

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

        for i in range(n_steps):
            branch = (y1[i] - self.exp_symbol[:, :, 0]) ** 2 + (y2[i] - self.exp_symbol[:, :, 1]) ** 2
            cand = metric[:, None] + branch

            new_metric = np.full(self.num_states, np.inf)
            bs = np.zeros(self.num_states, dtype=np.int32)
            bi = np.zeros(self.num_states, dtype=np.int32)
            for s in range(self.num_states):
                for u in range(2):
                    ns = self.nxt[s, u]
                    if cand[s, u] < new_metric[ns]:
                        new_metric[ns] = cand[s, u]
                        bs[ns] = s
                        bi[ns] = u
            metric = new_metric
            bp_state[i] = bs
            bp_input[i] = bi

        # Zero-tail termination: the trellis is known to end in state 0.
        s = 0
        decoded = np.empty(n_steps, dtype=int)
        for i in range(n_steps - 1, -1, -1):
            decoded[i] = bp_input[i, s]
            s = bp_state[i, s]

        return decoded[: n_steps - self.num_tail]
