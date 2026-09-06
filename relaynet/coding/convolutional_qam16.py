"""16-QAM branch-metric variant of the rate-1/2 convolutional trellis.

Unlike QPSK, where each trellis step's 2 coded bits map independently
onto the I and Q axes (bit 0 -> +1/-1 on I, bit 1 -> +1/-1 on Q), 16-QAM
Gray-codes 2 bits *jointly* onto one PAM-4 axis level
(``relaynet.modulation.qam``: 00->+3, 01->+1, 11->-1, 10->-3, normalized
by sqrt(10)) -- not decomposable into two independent per-bit soft
observations. So a trellis step's (out1, out2) here determines a single
real PAM-4 level, and two consecutive trellis steps pack into one 16-QAM
symbol's (I, Q), matching ``qam16_modulate``'s 4-bits-per-symbol
convention exactly (out1_i, out2_i -> I; out1_{i+1}, out2_{i+1} -> Q).

This duplicates the trellis-construction logic of
:mod:`relaynet.coding.convolutional` rather than sharing it, to avoid
touching the already-validated QPSK decoder while it's in active use
elsewhere; the two are algorithmically related but not interchangeable.
"""

import numpy as np

from .convolutional import STANDARD_GENERATORS, _generator_taps

# Matches relaynet.modulation.qam._IDX_TO_LEVEL / _QAM16_NORM exactly:
# index = b0*2 + b1 -> PAM-4 level (unit-average-power normalized).
_PAM4_IDX_TO_LEVEL = np.array([3.0, 1.0, -3.0, -1.0]) / np.sqrt(10.0)


class QAM16CodeDecoder:
    """Soft-decision Viterbi decoder for the convolutional trellis, with a
    16-QAM (PAM-4, Gray-coded, joint 2-bit) branch metric.

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
        return [(s >> (self.K - 2 - j)) & 1 for j in range(self.K - 1)]

    def _build_trellis(self):
        self.nxt = np.zeros((self.num_states, 2), dtype=np.int32)
        out_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)
        for s in range(self.num_states):
            reg = self._state_bits(s)
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
                ns = (u << (self.K - 2)) | (s >> 1) if self.K > 1 else 0
                self.nxt[s, u] = ns
                out_bits[s, u] = (out1, out2)

        # Joint PAM-4 expected level per (state, input) -- the key
        # difference from the QPSK decoder's independent per-bit exp_symbol.
        idx = out_bits[:, :, 0] * 2 + out_bits[:, :, 1]
        self.exp_level = _PAM4_IDX_TO_LEVEL[idx]  # shape (num_states, 2)

        mask = (1 << (self.K - 2)) - 1 if self.K > 2 else 0
        self.pred_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.pred_u = np.zeros(self.num_states, dtype=np.int32)
        for ns in range(self.num_states):
            base = ns & mask
            self.pred_state[ns, 0] = 2 * base
            self.pred_state[ns, 1] = 2 * base + 1
            self.pred_u[ns] = ns >> (self.K - 2) if self.K > 1 else 0

    def decode(self, soft_axis_values):
        """Decode one frame from a stream of real per-trellis-step observations.

        Parameters
        ----------
        soft_axis_values : array-like of float, length (n_info + K - 1)
            One real soft observation per trellis step: the I component
            for even steps, the Q component for odd steps, of the
            corresponding 16-QAM symbols (i.e. flatten each received
            symbol's (I, Q) into two consecutive real entries).

        Returns
        -------
        info_bits : ndarray of {0,1}
            Decoded information bits (tail bits stripped).
        """
        y = np.asarray(soft_axis_values, dtype=float)
        n_steps = len(y)

        metric = np.full(self.num_states, np.inf)
        metric[0] = 0.0
        bp_state = np.zeros((n_steps, self.num_states), dtype=np.int32)
        bp_input = np.zeros((n_steps, self.num_states), dtype=np.int32)

        pred0, pred1 = self.pred_state[:, 0], self.pred_state[:, 1]
        pred_u = self.pred_u
        for i in range(n_steps):
            branch = (y[i] - self.exp_level) ** 2  # shape (num_states, 2)

            cost0 = metric[pred0] + branch[pred0, pred_u]
            cost1 = metric[pred1] + branch[pred1, pred_u]
            take1 = cost1 < cost0

            metric = np.where(take1, cost1, cost0)
            bp_state[i] = np.where(take1, pred1, pred0)
            bp_input[i] = pred_u

        s = 0
        decoded = np.empty(n_steps, dtype=int)
        for i in range(n_steps - 1, -1, -1):
            decoded[i] = bp_input[i, s]
            s = bp_state[i, s]

        return decoded[: n_steps - self.num_tail]
