"""CSI-aware convolutional decoding; historical decoders remain reproducible.

Inputs are real coded-bit observations, interleaved I/Q for QPSK. Variance
means per-real-axis variance *after* equalization. For unit-power QPSK use
symbol_amplitude=1/sqrt(2); do not pass the nominal pre-equalization variance.
Uniform information-bit priors and zero-tail termination are assumed.
"""

import numpy as np
from scipy.special import logsumexp

from .convolutional import ViterbiCodeDecoder
from .bcjr import BCJRCodeDecoder


def _branch_cost(soft_bits, noise_var, amplitude, symbols, num_tail):
    y = np.asarray(soft_bits, dtype=float)
    if y.ndim != 1 or y.size % 2 or y.size < 2 * (num_tail + 1):
        raise ValueError("expected a complete, nonempty zero-tail frame")
    if not np.all(np.isfinite(y)):
        raise ValueError("observations must be finite")
    variance = np.asarray(noise_var, dtype=float)
    if variance.ndim != 0 and variance.shape != y.shape:
        raise ValueError("noise_var must be scalar or one variance per coded bit")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("noise_var must be finite and positive")
    if not np.isscalar(amplitude) or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("symbol_amplitude must be finite and positive")
    variance = np.broadcast_to(variance, y.shape).reshape(-1, 2)
    residual = y.reshape(-1, 1, 1, 2) - amplitude * symbols[None]
    cost = np.sum(residual ** 2 / (2 * variance[:, None, None, :]), axis=-1)
    if not np.all(np.isfinite(cost)):
        raise ValueError("branch costs overflow; rescale observations and variance")
    return cost


def qpsk_soft_observations(equalized, axis_noise_var):
    """Interleave complex QPSK observations and duplicate each axis variance."""
    y = np.asarray(equalized)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("expected a finite 1D QPSK observation")
    v = np.asarray(axis_noise_var, dtype=float)
    if v.ndim != 0 and v.shape != y.shape:
        raise ValueError("expected one axis variance per QPSK symbol or a scalar")
    if not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("axis_noise_var must be finite and positive")
    soft = np.column_stack((y.real, y.imag)).reshape(-1)
    return soft, np.repeat(np.broadcast_to(v, y.shape), 2)


class CodexMatchedViterbiDecoder(ViterbiCodeDecoder):
    """Sequence ML with heteroscedastic Gaussian branch metrics."""

    def decode(self, soft_bits, noise_var, *, symbol_amplitude=1.0):
        cost = _branch_cost(soft_bits, noise_var, symbol_amplitude,
                            self.exp_symbol, self.num_tail)
        metric = np.full(self.num_states, np.inf)
        metric[0] = 0.0
        back = np.empty((len(cost), self.num_states), dtype=int)
        p0, p1 = self.pred_state.T
        for i, branch in enumerate(cost):
            c0 = metric[p0] + branch[p0, self.pred_u]
            c1 = metric[p1] + branch[p1, self.pred_u]
            take1 = c1 < c0
            metric = np.where(take1, c1, c0)
            metric -= np.min(metric)
            back[i] = np.where(take1, p1, p0)
        state = 0
        bits = np.empty(len(cost), dtype=int)
        for i in range(len(cost) - 1, -1, -1):
            bits[i] = self.pred_u[state]
            state = back[i, state]
        return bits[:-self.num_tail]


class CodexMatchedBCJRDecoder(BCJRCodeDecoder):
    """Bit MAP in the log domain, with no uniform fallback on underflow."""

    def _marginals(self, soft_bits, noise_var, amplitude):
        gamma = -_branch_cost(soft_bits, noise_var, amplitude,
                              self.exp_symbol, self.num_tail)
        n = len(gamma)
        alpha = np.full((n + 1, self.num_states), -np.inf)
        beta = np.full_like(alpha, -np.inf)
        alpha[0, 0] = beta[n, 0] = 0.0
        for i in range(n):
            contrib = alpha[i, :, None] + gamma[i]
            for state in range(self.num_states):
                alpha[i + 1, state] = logsumexp(contrib[self.nxt == state])
            alpha[i + 1] -= logsumexp(alpha[i + 1])
        for i in range(n - 1, -1, -1):
            beta[i] = logsumexp(gamma[i] + beta[i + 1][self.nxt], axis=1)
            beta[i] -= logsumexp(beta[i])
        info = np.empty(n)
        coded = np.empty((n, 2))
        for i in range(n):
            logw = alpha[i, :, None] + gamma[i] + beta[i + 1][self.nxt]
            logw -= logsumexp(logw)
            info[i] = np.exp(logsumexp(logw[:, 1]))
            for bit in range(2):
                coded[i, bit] = np.exp(logsumexp(logw[self.out_bits[:, :, bit] == 1]))
        return np.clip(info[:-self.num_tail], 0, 1), np.clip(coded, 0, 1)

    def information_bit_posteriors(self, soft_bits, noise_var, *, symbol_amplitude=1.0):
        return self._marginals(soft_bits, noise_var, symbol_amplitude)[0]

    def coded_bit_posteriors(self, soft_bits, noise_var, *, symbol_amplitude=1.0):
        return self._marginals(soft_bits, noise_var, symbol_amplitude)[1]

    def decode(self, soft_bits, noise_var, *, symbol_amplitude=1.0):
        return (self.information_bit_posteriors(
            soft_bits, noise_var, symbol_amplitude=symbol_amplitude) > 0.5).astype(int)
