"""BCJR (forward-backward MAP) decoder for the rate-1/2 convolutional trellis.

Where :class:`relaynet.coding.convolutional.ViterbiCodeDecoder` returns the
single most likely *sequence* (hard output), this returns per-coded-bit
posterior probabilities -- the soft information a relay can forward instead
of committing to a hard codeword.

The distinction matters at a relay specifically. Hard block-DF decodes and
then *re-encodes*, so a wrong decode leaves the relay emitting a different
but perfectly valid codeword: the redundancy has been spent and regenerated
around the error, and the destination cannot detect or repair it. A relay
that forwards posterior means never commits, so the destination's decoder
still has the code's redundancy available to fix what the relay got wrong.

Same trellis, generators and bit-to-symbol convention (0 -> +1, 1 -> -1) as
:mod:`relaynet.coding.convolutional`; only the metric and the recursion
differ. Alpha/beta are rescaled at every step, which is the standard
guard against underflow over long frames and does not change the
posteriors (any per-step constant cancels in the final normalization).
"""

import numpy as np

from .convolutional import STANDARD_GENERATORS, _generator_taps


class BCJRCodeDecoder:
    """Forward-backward MAP decoder returning coded-bit posteriors.

    Parameters
    ----------
    constraint_length : int, optional
        Must match the encoder's (default 3). One of {3, 5, 7}.
    noise_var : float, optional
        Per-real-dimension noise variance assumed by the branch metric
        (default 1.0). Unlike Viterbi -- whose pure squared-distance metric
        is invariant to a uniform scaling -- BCJR needs this, because it
        sets how confident the posteriors are. The relay is assumed to know
        only the nominal operating SNR, not per-symbol CSI, so callers pass
        the nominal value; see :meth:`set_noise_var`.
    """

    def __init__(self, constraint_length=3, noise_var=1.0):
        if constraint_length not in STANDARD_GENERATORS:
            raise NotImplementedError(
                f"constraint_length={constraint_length} not supported; "
                f"choose one of {sorted(STANDARD_GENERATORS)}."
            )
        self.K = constraint_length
        self.num_tail = self.K - 1
        self.num_states = 2 ** (self.K - 1)
        self.noise_var = float(noise_var)
        g1, g2 = STANDARD_GENERATORS[self.K]
        self.taps1 = _generator_taps(g1, self.K)
        self.taps2 = _generator_taps(g2, self.K)
        self._build_trellis()

    def set_noise_var(self, noise_var):
        self.noise_var = float(noise_var)

    def _state_bits(self, s):
        return [(s >> (self.K - 2 - j)) & 1 for j in range(self.K - 1)]

    def _build_trellis(self):
        self.nxt = np.zeros((self.num_states, 2), dtype=np.int32)
        out_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)
        for s in range(self.num_states):
            reg = self._state_bits(s)
            for u in (0, 1):
                window = [u] + reg
                o1 = 0
                for w, t in zip(window, self.taps1):
                    if t:
                        o1 ^= w
                o2 = 0
                for w, t in zip(window, self.taps2):
                    if t:
                        o2 ^= w
                self.nxt[s, u] = (u << (self.K - 2)) | (s >> 1) if self.K > 1 else 0
                out_bits[s, u] = (o1, o2)
        self.out_bits = out_bits
        self.exp_symbol = 1.0 - 2.0 * out_bits.astype(float)

    def coded_bit_posteriors(self, soft_bits):
        """Posterior P(coded bit = 1) for every coded bit in one frame.

        Parameters
        ----------
        soft_bits : array-like of float, length 2*(n_info + K - 1)
            Interleaved (out1, out2, ...) real soft observations, the same
            layout :meth:`ConvolutionalEncoder.encode` produces.

        Returns
        -------
        p1 : ndarray, shape (n_steps, 2)
            Posterior probability that each of the two coded bits emitted at
            each trellis step is a 1.
        """
        y = np.asarray(soft_bits, dtype=float)
        n_steps = len(y) // 2
        y1, y2 = y[0::2], y[1::2]
        S, tiny = self.num_states, 1e-300

        # Branch likelihoods gamma[i][s,u].
        gamma = np.empty((n_steps, S, 2))
        for i in range(n_steps):
            d = ((y1[i] - self.exp_symbol[:, :, 0]) ** 2
                 + (y2[i] - self.exp_symbol[:, :, 1]) ** 2)
            g = np.exp(-d / (2.0 * self.noise_var) + d.min() / (2.0 * self.noise_var))
            gamma[i] = g

        # Forward recursion (trellis starts in the all-zero state).
        alpha = np.zeros((n_steps + 1, S))
        alpha[0, 0] = 1.0
        for i in range(n_steps):
            a = np.zeros(S)
            contrib = alpha[i][:, None] * gamma[i]
            for u in range(2):
                np.add.at(a, self.nxt[:, u], contrib[:, u])
            tot = a.sum()
            alpha[i + 1] = a / tot if tot > tiny else np.full(S, 1.0 / S)

        # Backward recursion (zero-tail terminated: ends in the all-zero state).
        beta = np.zeros((n_steps + 1, S))
        beta[n_steps, 0] = 1.0
        for i in range(n_steps - 1, -1, -1):
            b = np.zeros(S)
            for u in range(2):
                b += gamma[:, :, u][i] * beta[i + 1][self.nxt[:, u]]
            tot = b.sum()
            beta[i] = b / tot if tot > tiny else np.full(S, 1.0 / S)

        # Per-branch posteriors, then marginalize onto each coded bit.
        p1 = np.zeros((n_steps, 2))
        for i in range(n_steps):
            w = alpha[i][:, None] * gamma[i] * beta[i + 1][self.nxt]  # (S,2)
            tot = w.sum()
            if tot <= tiny:
                p1[i] = 0.5
                continue
            w = w / tot
            for bit in range(2):
                p1[i, bit] = w[self.out_bits[:, :, bit] == 1].sum()
        return np.clip(p1, 0.0, 1.0)

    def decode(self, soft_bits):
        """Hard MAP information-bit decisions (for reference/testing).

        Note this is bit-wise MAP, not the sequence-MAP that Viterbi gives;
        the two can disagree, which is exactly the distinction exploited by
        forwarding soft information rather than a re-encoded codeword.
        """
        y = np.asarray(soft_bits, dtype=float)
        n_steps = len(y) // 2
        S, tiny = self.num_states, 1e-300
        y1, y2 = y[0::2], y[1::2]

        gamma = np.empty((n_steps, S, 2))
        for i in range(n_steps):
            d = ((y1[i] - self.exp_symbol[:, :, 0]) ** 2
                 + (y2[i] - self.exp_symbol[:, :, 1]) ** 2)
            gamma[i] = np.exp(-d / (2.0 * self.noise_var) + d.min() / (2.0 * self.noise_var))

        alpha = np.zeros((n_steps + 1, S))
        alpha[0, 0] = 1.0
        for i in range(n_steps):
            a = np.zeros(S)
            contrib = alpha[i][:, None] * gamma[i]
            for u in range(2):
                np.add.at(a, self.nxt[:, u], contrib[:, u])
            tot = a.sum()
            alpha[i + 1] = a / tot if tot > tiny else np.full(S, 1.0 / S)

        beta = np.zeros((n_steps + 1, S))
        beta[n_steps, 0] = 1.0
        for i in range(n_steps - 1, -1, -1):
            b = np.zeros(S)
            for u in range(2):
                b += gamma[:, :, u][i] * beta[i + 1][self.nxt[:, u]]
            tot = b.sum()
            beta[i] = b / tot if tot > tiny else np.full(S, 1.0 / S)

        info = np.zeros(n_steps, dtype=int)
        for i in range(n_steps):
            w = alpha[i][:, None] * gamma[i] * beta[i + 1][self.nxt]
            info[i] = 1 if w[:, 1].sum() > w[:, 0].sum() else 0
        return info[: n_steps - self.num_tail]
