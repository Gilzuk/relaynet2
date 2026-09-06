"""Viterbi MLSE relay for ISI channels."""

import itertools
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
                # Next state: drop the oldest symbol, append the new one.
                # Written as (state + (u,))[1:] rather than state[1:] + (u,)
                # so that L=1 works: there the state is empty and the successor
                # of the single state is itself, where the latter form produces
                # a one-element tuple that is not a state at all.
                next_state = (state + (u,))[1:]
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


class ViterbiMLSEQPSKRelay(Relay):
    """Viterbi Maximum Likelihood Sequence Estimator for QPSK over an ISI channel.

    Same trellis-Viterbi structure as :class:`ViterbiMLSERelay`, generalized
    from the 2-symbol BPSK alphabet {-1, +1} to the 4-symbol Gray-coded QPSK
    alphabet, with complex branch metrics. Channel taps remain real-valued
    (the physical ISI impulse response), applied to complex QPSK symbols.

    Parameters
    ----------
    channel_taps : array-like, optional
        Known real-valued channel taps [h0, h1, ..., h_{L-1}] (genie CSI).
    pilot_symbols : tuple of (y, x), optional
        Pilot symbols (received_y, transmitted_x) for LS channel estimation.
        If provided, channel_taps are ignored (estimated from pilots).
    channel_len : int, optional
        Channel length for LS estimation (default 3 for 3-tap ISI).
    """

    # Gray-coded QPSK alphabet, matching relaynet.modulation.qpsk exactly:
    # index -> (I, Q) -> bits (first bit selects I: 0->+1,1->-1; second selects Q likewise)
    ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    def __init__(self, channel_taps=None, pilot_symbols=None, channel_len=3):
        self.M = len(self.ALPHABET)
        self.L = channel_len
        self.num_states = self.M ** (self.L - 1)

        if pilot_symbols is not None:
            y_p, x_p = pilot_symbols
            self.h = self._ls_estimate(y_p, x_p)
        elif channel_taps is not None:
            self.h = np.asarray(channel_taps, dtype=float)
            self.L = len(self.h)
            self.num_states = self.M ** (self.L - 1)
        else:
            raise ValueError("Either channel_taps or pilot_symbols must be provided")

        self._build_trellis()

    def _ls_estimate(self, y_pilot, x_pilot):
        """Estimate real-valued channel taps from complex pilot symbols using LS."""
        n = len(x_pilot)
        X = np.zeros((n, self.L), dtype=complex)
        X[:, 0] = x_pilot

        for i in range(1, self.L):
            X[i:, i] = x_pilot[:-i]

        h, *_ = np.linalg.lstsq(X, y_pilot[:n], rcond=None)
        return h.real

    def _build_trellis(self):
        """Build state transition and output tables (base-M digit states)."""
        # States are (L-1)-tuples of alphabet indices representing
        # (x[i-L+1], ..., x[i-1]).
        self.states = list(itertools.product(range(self.M), repeat=self.L - 1))
        state_index = {state: idx for idx, state in enumerate(self.states)}

        self.nxt = np.zeros((self.num_states, self.M), dtype=np.int32)
        self.exp_y = np.zeros((self.num_states, self.M), dtype=complex)

        for s, state in enumerate(self.states):
            for u in range(self.M):
                next_state = (state + (u,))[1:]      # L=1 safe; see ViterbiMLSERelay
                self.nxt[s, u] = state_index[next_state]

                expected = self.h[0] * self.ALPHABET[u]
                for j in range(self.L - 1):
                    expected += self.h[j + 1] * self.ALPHABET[state[self.L - 2 - j]]
                self.exp_y[s, u] = expected

    def process(self, received_signal):
        """Decode received QPSK signal using the Viterbi algorithm.

        Parameters
        ----------
        received_signal : ndarray
            Received complex samples.

        Returns
        -------
        decoded : ndarray
            Decoded QPSK symbols (complex, unit average power).
        """
        y = received_signal
        n = len(y)

        metric = np.zeros(self.num_states)
        bp_state = np.zeros((n, self.num_states), dtype=np.int32)
        bp_input = np.zeros((n, self.num_states), dtype=np.int32)

        for i in range(n):
            cand = metric[:, None] + np.abs(y[i] - self.exp_y) ** 2

            new_metric = np.full(self.num_states, np.inf, dtype=float)
            bs = np.zeros(self.num_states, dtype=np.int32)
            bi = np.zeros(self.num_states, dtype=np.int32)

            for s in range(self.num_states):
                for u in range(self.M):
                    ns = self.nxt[s, u]
                    if cand[s, u] < new_metric[ns]:
                        new_metric[ns] = cand[s, u]
                        bs[ns] = s
                        bi[ns] = u

            metric = new_metric
            bp_state[i] = bs
            bp_input[i] = bi

        s = int(np.argmin(metric))
        decoded = np.empty(n, dtype=complex)

        for i in range(n - 1, -1, -1):
            u_idx = bp_input[i, s]
            decoded[i] = self.ALPHABET[u_idx]
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
            self.num_states = self.M ** (self.L - 1)
        else:
            raise ValueError("Either channel_taps or pilot_symbols must be provided")

        self._build_trellis()


class TruncatedViterbiQPSKRelay(ViterbiMLSEQPSKRelay):
    """QPSK MLSE with a bounded decision delay (sliding-window traceback).

    :class:`ViterbiMLSEQPSKRelay` traces back from the end of the block, so
    its structural latency is the whole block -- it cannot be placed on a
    latency axis alongside a windowed relay. This subclass emits the
    decision for symbol ``n`` after observing ``y[n + traceback]``, which is
    how MLSE is actually deployed, and makes the decision delay an explicit
    parameter of the equalizer.

    The add-compare-select step is vectorized over the trellis (a gather on
    the predecessor table rather than a Python loop over states), so the
    ``M**(L-1)`` state count can be swept without the runtime becoming the
    limiting factor.

    Parameters
    ----------
    channel_taps : array-like, optional
        Known real-valued taps (genie CSI).
    pilot_symbols : tuple, optional
        ``(y_pilot, x_pilot)`` for LS estimation.
    channel_len : int, optional
        Channel length used for LS estimation.
    traceback : int, optional
        Decision delay in symbols (default 5 * channel_len, the usual rule
        of thumb). ``traceback=0`` commits each symbol as soon as it is
        observed; the decision still comes from the accumulated path
        metric over the whole history, so it is zero *look-ahead* rather
        than a memoryless slicer.
    """

    def __init__(self, channel_taps=None, pilot_symbols=None, channel_len=3,
                 traceback=None):
        super().__init__(channel_taps=channel_taps, pilot_symbols=pilot_symbols,
                         channel_len=channel_len)
        self.traceback = int(5 * self.L) if traceback is None else int(traceback)
        if self.traceback < 0:
            raise ValueError("traceback must be non-negative, got "
                             f"{self.traceback}")
        self._build_predecessors()

    def _build_predecessors(self):
        """Invert the trellis: for each state, the M branches arriving at it."""
        S, M = self.num_states, self.M
        self.pred_state = np.zeros((S, M), dtype=np.int64)
        self.pred_input = np.zeros((S, M), dtype=np.int64)
        fill = np.zeros(S, dtype=np.int64)
        for s in range(S):
            for u in range(M):
                ns = self.nxt[s, u]
                self.pred_state[ns, fill[ns]] = s
                self.pred_input[ns, fill[ns]] = u
                fill[ns] += 1
        if not np.all(fill == M):
            raise RuntimeError("trellis is not M-regular; predecessor table invalid")

    def n_states(self):
        return self.num_states

    def process(self, received_signal):
        y = np.asarray(received_signal)
        n = len(y)
        D = self.traceback
        S, M = self.num_states, self.M
        depth = D + 1

        bp_state = np.zeros((depth, S), dtype=np.int64)
        bp_input = np.zeros((depth, S), dtype=np.int64)
        metric = np.zeros(S)
        rows = np.arange(S)
        out_idx = np.zeros(n, dtype=np.int64)

        for i in range(n):
            cand = metric[:, None] + np.abs(y[i] - self.exp_y) ** 2   # (S, M)
            arriving = cand[self.pred_state, self.pred_input]          # (S, M)
            k = np.argmin(arriving, axis=1)
            metric = arriving[rows, k]
            metric -= metric.min()

            slot = i % depth
            bp_state[slot] = self.pred_state[rows, k]
            bp_input[slot] = self.pred_input[rows, k]

            if i >= D:
                s = int(np.argmin(metric))
                for j in range(i, i - D, -1):
                    s = bp_state[j % depth, s]
                out_idx[i - D] = bp_input[(i - D) % depth, s]

        # Flush the final D symbols from the surviving path.
        if n:
            s = int(np.argmin(metric))
            start = max(n - D, 0)
            for j in range(n - 1, start - 1, -1):
                out_idx[j] = bp_input[j % depth, s]
                s = bp_state[j % depth, s]

        return self.ALPHABET[out_idx]

    def set_channel(self, channel_taps=None, pilot_symbols=None):
        """Update the channel estimate and re-invert the new trellis.

        The inherited implementation rebuilds ``nxt`` and ``exp_y`` (and,
        for a different tap count, ``num_states``) but knows nothing about
        the predecessor table this class decodes from. Without the rebuild
        below, a relay re-estimated from pilots would decode against a
        trellis inversion belonging to the previous channel.
        """
        default_before = self.traceback == 5 * self.L
        super().set_channel(channel_taps=channel_taps, pilot_symbols=pilot_symbols)
        # A default traceback is a function of the channel length, so it has to
        # follow the channel. An explicitly chosen one is the caller's and is
        # left alone -- the point of the parameter is to fix the delay budget.
        if default_before:
            self.traceback = 5 * self.L
        self._build_predecessors()


class FadingAwareViterbiQPSKRelay(ViterbiMLSEQPSKRelay):
    """QPSK MLSE told the per-symbol fading gain as well as the channel taps.

    :class:`ViterbiMLSEQPSKRelay` assumes ``y[n] = (h * x)[n] + v[n]``. The
    channel used in the QPSK unknown-ISI study is
    :class:`~relaynet.channels.e6_channels.ComplexISIRayleighChannel`, which is

        y[n] = g[n] (h * x)[n] + v[n],   g[n] = |CN(0,1)|,

    an independent fading magnitude on every symbol. A detector given only ``h``
    is therefore *not* genie CSI on that channel: its branch metrics compare
    ``y[n]`` against unfaded expected values, so the residual
    ``|g[n] A - A|^2 = A^2 (g[n] - 1)^2`` does not vanish as the noise does, and
    its error rate flattens instead of falling. That mismatch, not any
    sequence-versus-bit subtlety, is what a learned relay trained on the faded
    channel exploits.

    This subclass takes the gains and scales each branch's expected observation
    by ``g[n]``, which is the genie-CSI detector for that channel. Pass them
    with :meth:`set_gains` before :meth:`process`, or leave them unset to get
    the unfaded behaviour of the parent unchanged.
    """

    def __init__(self, channel_taps=None, pilot_symbols=None, channel_len=3,
                 gains=None):
        super().__init__(channel_taps=channel_taps, pilot_symbols=pilot_symbols,
                         channel_len=channel_len)
        self.gains = None if gains is None else np.asarray(gains, dtype=float)

    def set_gains(self, gains):
        """Supply the per-symbol fading magnitudes for the next block."""
        self.gains = None if gains is None else np.asarray(gains, dtype=float)
        return self

    def process(self, received_signal):
        if self.gains is None:
            return super().process(received_signal)

        y = received_signal
        n = len(y)
        g = self.gains
        if g.size != n:
            raise ValueError(f"got {g.size} gains for {n} symbols; call "
                             "set_gains() with one gain per received symbol")

        metric = np.zeros(self.num_states)
        bp_state = np.zeros((n, self.num_states), dtype=np.int32)
        bp_input = np.zeros((n, self.num_states), dtype=np.int32)

        for i in range(n):
            # the only change from the parent: the expected observation for
            # every branch is scaled by this symbol's fading gain
            cand = metric[:, None] + np.abs(y[i] - g[i] * self.exp_y) ** 2

            new_metric = np.full(self.num_states, np.inf, dtype=float)
            bs = np.zeros(self.num_states, dtype=np.int32)
            bi = np.zeros(self.num_states, dtype=np.int32)
            for s in range(self.num_states):
                for u in range(self.M):
                    ns = self.nxt[s, u]
                    if cand[s, u] < new_metric[ns]:
                        new_metric[ns] = cand[s, u]
                        bs[ns] = s
                        bi[ns] = u
            metric = new_metric
            bp_state[i] = bs
            bp_input[i] = bi

        s = int(np.argmin(metric))
        decoded = np.empty(n, dtype=complex)
        for i in range(n - 1, -1, -1):
            u_idx = bp_input[i, s]
            decoded[i] = self.ALPHABET[u_idx]
            s = bp_state[i, s]
        return decoded
