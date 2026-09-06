r"""BCJR/APP detector for QPSK over the unknown-ISI relay channel.

Why this exists. Chapter 7 benchmarks the learned relay against genie-CSI
Viterbi MLSE, which minimizes the probability that the whole *sequence* is
wrong. The thesis reports BER, and sequence-ML is not BER-optimal on a channel
with memory. The detector that is BER-optimal is the bit-wise MAP rule, taken
from the per-bit posteriors a BCJR forward-backward recursion delivers
[Bahl et al. 1974]. Until it is measured, the margin between the classical
benchmark and the learned relay has never been compared against a BER-optimal
comparator at either modulation order.

Channel model, matching :class:`FadingAwareViterbiQPSKRelay` exactly:

    y[n] = g[n] (h * x)[n] + v[n],    v[n] ~ CN(0, sigma^2)

with ``g[n]`` an independent per-symbol Rayleigh magnitude. Giving the detector
``h`` but not ``g`` is *not* genie CSI on this channel; that mismatch is what
produced the withdrawn QPSK result, so this class takes the gains too.

Two decision rules are available on the same posteriors, and the difference
between them is the whole point:

``decision="bit"``     bit-wise MAP -- marginalize the symbol posteriors down
                       to per-bit LLRs and slice each bit. BER-optimal. Default.
``decision="symbol"``  symbol-MAP -- argmax of the symbol posterior. Minimizes
                       symbol error rate, not bit error rate.

Reassembling independently-decided bits can yield a symbol that is not the
symbol-MAP choice. That is not a bug; it is the distinction being measured.
"""

import numpy as np

from .viterbi import FadingAwareViterbiQPSKRelay

NEG_INF = -np.inf


def _logsumexp(a, axis=None):
    """Stable log-sum-exp that tolerates all -inf slices."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out


class BCJRQPSKRelay(FadingAwareViterbiQPSKRelay):
    """Forward-backward APP detector on the QPSK ISI trellis.

    Shares the trellis of :class:`ViterbiMLSEQPSKRelay` -- same states, same
    ``nxt`` table, same noiseless branch outputs ``exp_y`` -- so that a
    comparison against Viterbi differs in the decision rule alone and in
    nothing else.

    Parameters
    ----------
    sigma : float
        Noise standard deviation, with ``E|v|^2 = sigma^2``. Unlike Viterbi,
        whose argmin is invariant to metric scale, the APP recursion needs the
        true scale: it weighs paths against each other rather than picking one.
        Set it per block with :meth:`set_sigma`, or pass ``snr_db`` there.
    decision : {"bit", "symbol"}
        Which rule to apply to the posteriors. See the module docstring.
    """

    #: bit b0 is the real-axis bit, b1 the imaginary; symbol index = 2*b0 + b1,
    #: matching ``qpsk_mod``: ((1-2*b0) + 1j*(1-2*b1)) / sqrt(2).
    BIT0 = np.array([0, 0, 1, 1])
    BIT1 = np.array([0, 1, 0, 1])

    def __init__(self, channel_taps=None, pilot_symbols=None, channel_len=3,
                 gains=None, sigma=None, decision="bit"):
        super().__init__(channel_taps=channel_taps, pilot_symbols=pilot_symbols,
                         channel_len=channel_len, gains=gains)
        if decision not in ("bit", "symbol"):
            raise ValueError('decision must be "bit" or "symbol", '
                             f'got {decision!r}')
        self.decision = decision
        self.sigma = None if sigma is None else float(sigma)
        self.last_bit_llr = None      # (2, n): LLRs from the most recent block

    def set_sigma(self, sigma=None, snr_db=None):
        """Set the noise scale, directly or from an SNR in dB.

        The SNR convention is the project's: ``sigma = 10 ** (-snr_db / 20)``,
        matching ``ComplexISIRayleighChannel``.
        """
        if (sigma is None) == (snr_db is None):
            raise ValueError("pass exactly one of sigma, snr_db")
        self.sigma = float(sigma) if sigma is not None else 10 ** (-snr_db / 20.0)
        return self

    def _branch_loglik(self, y, g):
        """(n, num_states, M) log p(y[i] | branch), up to a constant."""
        exp_y = self.exp_y[None, :, :]                       # (1, S, M)
        scale = 1.0 if g is None else g[:, None, None]       # (n, 1, 1)
        resid = y[:, None, None] - scale * exp_y
        return -np.abs(resid) ** 2 / (self.sigma ** 2)

    def posteriors(self, received_signal):
        """Per-symbol log posteriors over the M inputs, shape (n, M).

        Both recursions start and end uniform over states, which mirrors the
        Viterbi implementation this is compared against: it initializes every
        state metric to zero and takes an argmin over all terminal states,
        i.e. it assumes neither a known start nor a known end.
        """
        if self.sigma is None:
            raise ValueError("noise scale unknown: call set_sigma() first")
        y = np.asarray(received_signal)
        n, S, M = y.size, self.num_states, self.M
        g = self.gains
        if g is not None and g.size != n:
            raise ValueError(f"got {g.size} gains for {n} symbols")

        gamma = self._branch_loglik(y, g)                    # (n, S, M)
        nxt = self.nxt                                       # (S, M)

        alpha = np.zeros((n + 1, S))                         # uniform start
        for i in range(n):
            contrib = alpha[i][:, None] + gamma[i]           # (S, M)
            # scatter log-sum-exp into successor states
            flat_targets = nxt.reshape(-1)
            flat_vals = contrib.reshape(-1)
            mx = np.full(S, NEG_INF)
            np.maximum.at(mx, flat_targets, flat_vals)
            shifted = np.exp(flat_vals - mx[flat_targets])
            tot = np.zeros(S)
            np.add.at(tot, flat_targets, shifted)
            with np.errstate(divide="ignore"):
                alpha[i + 1] = mx + np.log(tot)
            alpha[i + 1] -= np.max(alpha[i + 1])             # renormalise

        beta = np.zeros((n + 1, S))                          # uniform end
        for i in range(n - 1, -1, -1):
            contrib = gamma[i] + beta[i + 1][nxt]            # (S, M)
            beta[i] = _logsumexp(contrib, axis=1)
            beta[i] -= np.max(beta[i])

        # log APP of input u at time i, marginalising the state
        app = np.empty((n, M))
        for i in range(n):
            app[i] = _logsumexp(alpha[i][:, None] + gamma[i]
                                + beta[i + 1][nxt], axis=0)
        return app - _logsumexp(app, axis=1)[:, None]

    def process(self, received_signal):
        app = self.posteriors(received_signal)
        if self.decision == "symbol":
            idx = np.argmax(app, axis=1)
        else:
            # bit-wise MAP: LLR = log P(b=0) - log P(b=1), sliced independently
            llr0 = (_logsumexp(app[:, self.BIT0 == 0], axis=1)
                    - _logsumexp(app[:, self.BIT0 == 1], axis=1))
            llr1 = (_logsumexp(app[:, self.BIT1 == 0], axis=1)
                    - _logsumexp(app[:, self.BIT1 == 1], axis=1))
            self.last_bit_llr = np.vstack([llr0, llr1])
            b0 = (llr0 < 0).astype(int)
            b1 = (llr1 < 0).astype(int)
            idx = 2 * b0 + b1
        return self.ALPHABET[idx]
