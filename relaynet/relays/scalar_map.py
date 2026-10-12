"""Symbol-by-symbol MAP relay for a known real ISI channel.

The memoryless relays of the unknown-ISI study (AF, zero-threshold DF) decide
each symbol from y[n] with a threshold at zero. For a channel with memory that
is not the best memoryless rule: conditioned on x[n] = b, the observation
y[n] = h0 b + sum_k h_k x[n-k] + v[n] is a Gaussian mixture over the
equiprobable interfering patterns, and the scalar MAP rule compares the two
mixtures. Its decision regions are in general not a single threshold.

The relay is told the taps and the noise variance -- the same information the
genie-CSI Viterbi detector receives -- but uses no temporal context. Comparing
it with the windowed learned relay separates the benefit of a better scalar
decision rule from the benefit of observing neighbouring samples.
"""

import itertools

import numpy as np

from .base import Relay


class ScalarMAPISIRelay(Relay):
    """Hard symbol-by-symbol MAP decision on y[n] for BPSK over known real ISI.

    Parameters
    ----------
    channel_taps : array-like
        Real taps (h0, h1, ..., h_{L-1}); normalized to unit energy, as
        ISIChannel does internally.
    snr_db : float
        SNR in the convention of ISIChannel: noise variance 1/(2 gamma) per real
        dimension, gamma = 10^(snr_db/10).
    """

    def __init__(self, channel_taps, snr_db):
        h = np.asarray(channel_taps, dtype=float)
        self.h = h / np.linalg.norm(h)
        self.set_snr(snr_db)
        patterns = np.array(list(itertools.product((1.0, -1.0), repeat=len(self.h) - 1)))
        isi = patterns @ self.h[1:] if len(self.h) > 1 else np.zeros(1)
        self.means_plus = self.h[0] + isi          # noiseless y given x[n] = +1
        self.means_minus = -self.h[0] + isi        # ... given x[n] = -1

    def set_snr(self, snr_db):
        self.sigma2 = 1.0 / (2.0 * 10 ** (snr_db / 10.0))
        return self

    def log_likelihood_ratio(self, y):
        """log p(y | x=+1) - log p(y | x=-1), each a mixture over ISI patterns."""
        y = np.asarray(y, dtype=float)[:, None]
        a = -(y - self.means_plus[None, :]) ** 2 / (2 * self.sigma2)
        b = -(y - self.means_minus[None, :]) ** 2 / (2 * self.sigma2)
        lse = lambda z: z.max(axis=1) + np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1))
        return lse(a) - lse(b)

    def process(self, received_signal):
        llr = self.log_likelihood_ratio(received_signal)
        return np.where(llr >= 0, 1.0, -1.0)
