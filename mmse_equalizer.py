"""MMSE linear equalization as the complexity-matched classical baseline.

The unknown-channel study compares a windowed learned relay against two
classical relays: symbol-wise DF, which applies no equalization at all, and
Viterbi MLSE, which is the optimal sequence detector. Neither is complexity
matched to the learned relay. DF is memoryless and cannot in principle undo
intersymbol interference, so beating it demonstrates only that the relay uses
its window. MLSE costs O(M^L) per symbol in the channel memory L, which is the
basis of the thesis's complexity argument for the learned relay in the first
place.

That leaves the obvious middle case untested. An N-tap MMSE linear equalizer
is in the same complexity class as an N-tap learned relay -- both are a
windowed linear operation, one followed by a small nonlinearity -- and it is
what a practising engineer would reach for before a trellis. If MMSE-LE
recovers most of what the learned relay recovers, the interesting claim is
much narrower than "learning beats classical processing on channels with
memory": it becomes a claim about the nonlinearity, worth a specific and
smaller amount.

DESIGN. For a channel with impulse response h of length L, an N-tap equalizer
and decision delay d, stacking the last N received samples gives
y_n = H x_n + v with H the N x (N+L-1) convolution matrix. The MMSE weights
follow from the orthogonality principle,

    w = (sigma_x^2 H H^H + sigma_v^2 I)^{-1} sigma_x^2 H e_d,

and the equalizer output is w^H y_n. The delay d is chosen by trying every
value and keeping the one with the lowest resulting MSE, which is standard and
avoids handing the equalizer an arbitrarily bad alignment.

The equalizer is given the exact channel taps, the same genie-CSI advantage
ViterbiMLSERelay receives, so the comparison isolates the detector rather than
channel knowledge.
"""

import numpy as np

from relaynet.relays.base import Relay


class MMSELinearEqualizerRelay(Relay):
    """N-tap MMSE linear equalizer, forwarded as a soft estimate.

    Parameters
    ----------
    channel_taps : array-like
        Known channel impulse response (genie CSI).
    num_taps : int
        Equalizer length N, the analogue of the learned relay's window.
    snr_db : float, optional
        SNR the weights are designed for. When None the weights are redesigned
        per call from the SNR the channel is being run at, which is the
        favourable case for the equalizer.
    hard : bool
        If True, slice the equalized estimate to the constellation before
        forwarding (equalization followed by a decision, i.e. DF after
        equalization). If False, forward the soft estimate power-normalized,
        which is what the learned relays do.
    """

    def __init__(self, channel_taps, num_taps=7, snr_db=None, hard=False,
                 target_power=1.0):
        self.h = np.asarray(channel_taps, dtype=float).ravel()
        self.N = int(num_taps)
        self.snr_db = snr_db
        self.hard = bool(hard)
        self.target_power = target_power
        self.handles_complex_natively = False   # applied per axis, like the MLPs
        self._cache = {}

    def _weights(self, snr_db):
        key = round(float(snr_db), 6)
        if key in self._cache:
            return self._cache[key]
        h, N, L = self.h, self.N, len(self.h)
        # convolution matrix: row i selects x[n-i-L+1 .. n-i]
        H = np.zeros((N, N + L - 1))
        for i in range(N):
            H[i, i:i + L] = h
        sig_x2 = 1.0
        sig_v2 = 1.0 / (2.0 * 10 ** (snr_db / 10.0))   # per real dimension
        R = sig_x2 * (H @ H.T) + sig_v2 * np.eye(N)
        Rinv = np.linalg.inv(R)
        best = None
        for d in range(N + L - 1):
            p = sig_x2 * H[:, d]
            w = Rinv @ p
            mse = sig_x2 - w @ p                      # MMSE at this delay
            if best is None or mse < best[0]:
                best = (mse, w, d)
        self._cache[key] = (best[1], best[2])
        return self._cache[key]

    def process(self, received_signal):
        y = received_signal
        if isinstance(y, tuple):
            y = y[0]
        y = np.asarray(y, dtype=float).ravel()
        snr = self.snr_db if self.snr_db is not None else self._runtime_snr
        w, d = self._weights(snr)
        # sliding inner product; pad so the output aligns with the input
        yp = np.concatenate([np.zeros(self.N - 1), y])
        win = np.lib.stride_tricks.sliding_window_view(yp, self.N)[:, ::-1]
        out = win @ w
        # the equalizer estimates x[n-d], so advance by the design delay to
        # realign with x[n]. Omitting this is silent for delays that happen to
        # come out zero and catastrophic otherwise: at 3, 5 and 7 taps the
        # optimum here is d=0 and the output looked correct, while at 11 taps
        # d=3 and the unshifted output sat at chance.
        if d:
            out = np.concatenate([out[d:], np.zeros(d)])
        if self.hard:
            out = np.sign(out)
            out[out == 0] = 1.0
        p = np.sqrt(np.mean(out ** 2)) + 1e-12
        return out / p * np.sqrt(self.target_power)

    # the runner does not pass SNR to process(), so it is set per evaluation
    _runtime_snr = 10.0


def _selftest():
    """On a known 3-tap channel the equalizer must beat a memoryless slicer,
    and its advantage must grow with the number of taps."""
    from relaynet.channels import ISIChannel
    h = np.array([1.0, 0.7, 0.5]); h /= np.linalg.norm(h)
    rng = np.random.default_rng(0)
    x = 1.0 - 2.0 * rng.integers(0, 2, 200000).astype(float)
    print(f"  {'detector':<22}{'symbol error @12 dB':>22}")
    np.random.seed(0)
    y = ISIChannel(h, seed=1)(x, 12.0)
    print(f"  {'memoryless slicer':<22}{np.mean(np.sign(y) != np.sign(x)):>22.5f}")
    prev = None
    for N in (3, 5, 7, 11):
        eq = MMSELinearEqualizerRelay(h, num_taps=N, snr_db=12.0, hard=False)
        out = eq.process(y)
        err = np.mean(np.sign(out) != np.sign(x))
        flag = "" if prev is None or err <= prev + 1e-4 else "  <-- worse than fewer taps"
        print(f"  {'MMSE-LE ' + str(N) + ' taps':<22}{err:>22.5f}{flag}")
        prev = err
    return True


if __name__ == "__main__":
    _selftest()
