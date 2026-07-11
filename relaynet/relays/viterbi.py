"""Viterbi MLSE relay for ISI channels."""

import numpy as np
from .base import Relay


class ViterbiMLSERelay(Relay):
    """Viterbi Maximum Likelihood Sequence Estimator for ISI channels.

    Implements MLSE decoding for BPSK signals over an L-tap FIR channel.
    Uses a 2^(L-1)-state Viterbi trellis decoder.

    Parameters
    ----------
    channel_taps : array-like
        Channel impulse response [h0, h1, ..., h_{L-1}].
    pilot_symbols : tuple of (y, x), optional
        Pilot symbols (received_y, transmitted_x) for LS channel estimation.
        If provided, channel_taps are ignored (estimated from pilots).
    """

    def __init__(self, channel_taps=None, pilot_symbols=None, channel_len=3):
        """Initialize Viterbi decoder.

        Parameters
        ----------
        channel_taps : array-like, optional
            Known channel taps (genie CSI case).
        pilot_symbols : tuple, optional
            (y_pilot, x_pilot) for LS estimation of unknown channel.
        channel_len : int, optional
            Channel length for LS estimation (default 3 for 3-tap ISI).
        """
        self.L = channel_len
        self.num_states = 2 ** (self.L - 1)

        if pilot_symbols is not None:
            # LS estimate from pilots
            y_p, x_p = pilot_symbols
            self.h = self._ls_estimate(y_p, x_p)
        elif channel_taps is not None:
            self.h = np.asarray(channel_taps, dtype=float)
            self.L = len(self.h)
            self.num_states = 2 ** (self.L - 1)
        else:
            raise ValueError("Either channel_taps or pilot_symbols must be provided")

        # Build state transition table
        self._build_trellis()

    def _ls_estimate(self, y_pilot, x_pilot):
        """Estimate channel from pilot symbols using LS.

        Parameters
        ----------
        y_pilot : array-like
            Received pilot symbols.
        x_pilot : array-like
            Transmitted pilot symbols.

        Returns
        -------
        h : ndarray
            Estimated channel taps.
        """
        n = len(x_pilot)
        X = np.zeros((n, self.L), dtype=float)
        X[:, 0] = x_pilot

        for i in range(1, self.L):
            X[i:, i] = x_pilot[:-i]

        # Solve X @ h = y using lstsq
        h, *_ = np.linalg.lstsq(X, y_pilot[:n], rcond=None)
        return h

    def _build_trellis(self):
        """Build state transition and output tables."""
        # States represent (x[i-L+1], x[i-L+2], ..., x[i-1])
        # For L=3: states = (x[i-2], x[i-1])
        self.states = []
        for i in range(self.num_states):
            state = []
            s = i
            for _ in range(self.L - 1):
                state.append(2 * (s & 1) - 1.0)
                s >>= 1
            self.states.append(tuple(reversed(state)))

        # Next state table: nxt[state][input] = next_state_index
        self.nxt = np.zeros((self.num_states, 2), dtype=np.int32)
        # Expected output: exp_y[state][input] = h·(state + input concatenation)
        self.exp_y = np.zeros((self.num_states, 2), dtype=float)

        for s, state in enumerate(self.states):
            for u_idx, u in enumerate((-1.0, 1.0)):
                # Next state: (state[1:], u)
                next_state = state[1:] + (u,)
                next_s = self.states.index(next_state)
                self.nxt[s, u_idx] = next_s

                # Expected output: h[0]*u + h[1]*state[-1] + ... + h[L-1]*state[0]
                expected = self.h[0] * u
                for j in range(self.L - 1):
                    expected += self.h[j + 1] * state[self.L - 2 - j]
                self.exp_y[s, u_idx] = expected

    def process(self, received_signal):
        """Decode received signal using Viterbi algorithm.

        Parameters
        ----------
        received_signal : ndarray
            Received samples.

        Returns
        -------
        decoded : ndarray
            Decoded BPSK symbols {-1, 1}.
        """
        y = received_signal
        n = len(y)

        # Initialize Viterbi
        metric = np.zeros(self.num_states)
        bp_state = np.zeros((n, self.num_states), dtype=np.int32)
        bp_input = np.zeros((n, self.num_states), dtype=np.int32)

        # Forward pass
        for i in range(n):
            # Candidate metrics: metric[s] + (y[i] - expected_output[s,u])^2
            cand = metric[:, None] + (y[i] - self.exp_y) ** 2

            new_metric = np.full(self.num_states, np.inf, dtype=float)
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

        # Traceback
        s = int(np.argmin(metric))
        decoded = np.empty(n, dtype=float)

        for i in range(n - 1, -1, -1):
            u_idx = bp_input[i, s]
            decoded[i] = 2.0 * u_idx - 1.0
            s = bp_state[i, s]

        return decoded

    def set_channel(self, channel_taps=None, pilot_symbols=None):
        """Update channel estimate.

        Parameters
        ----------
        channel_taps : array-like, optional
        pilot_symbols : tuple, optional
        """
        if pilot_symbols is not None:
            y_p, x_p = pilot_symbols
            self.h = self._ls_estimate(y_p, x_p)
        elif channel_taps is not None:
            self.h = np.asarray(channel_taps, dtype=float)
            self.L = len(self.h)
            self.num_states = 2 ** (self.L - 1)
        else:
            raise ValueError("Either channel_taps or pilot_symbols must be provided")

        self._build_trellis()
