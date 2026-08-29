"""Multi-hidden-layer MLP relay, for sweeping depth and width independently.

relaynet's MLPRelay has exactly one hidden layer, so "hidden_size" and
"neurons per layer" are the same number there and depth cannot be varied.
This class generalizes it to L hidden layers of equal width, which lets the
minimum-size study separate three things that the single-layer sweep confounds:

    window          how much of the received sequence the relay sees
    depth           how many hidden layers
    width           neurons per hidden layer

Parameter count for window w, depth L, width h, scalar output:

    params = w*h + h  +  (L-1)*(h*h + h)  +  h + 1

At L=1 this collapses to h*(w+2)+1, the formula used throughout the
single-layer study, so counts stay directly comparable across both.

DELIBERATELY A SEPARATE CLASS. relaynet/relays/mlp.py is used by the
Chapter 7 experiments whose numbers are already in the thesis. Adding depth
to it in place would put published results at risk of silently changing, so
this subclasses nothing and touches nothing -- it only mirrors MLPRelay's
conventions so that depth 1 reproduces it:

  * tanh on every hidden layer and on the output
  * He initialization from a seeded default_rng(seed)
  * Adam with beta1 0.9, beta2 0.999, eps 1e-8, the same update expression
  * batch shuffling from a fixed default_rng(42)
  * output power normalized to unity in process()
  * centred sliding window with zero padding

The depth-1 equivalence is asserted by test_deep_mlp_matches_single_layer().
"""

import numpy as np

from relaynet.relays.base import Relay


class DeepMLPRelay(Relay):
    """Feed-forward relay with `depth` hidden layers of `width` neurons.

    Parameters
    ----------
    input_size : int
        Input dimension (window length, or 2*window for joint I/Q).
    width : int
        Neurons per hidden layer.
    depth : int, optional
        Number of hidden layers (default 1, equivalent to MLPRelay).
    output_size : int, optional
        Output dimension (default 1).
    window_size : int, optional
        If given, process() extracts centred sliding windows of this length.
    seed : int, optional
        Seed for weight initialization.
    """

    def __init__(self, input_size, width, depth=1, output_size=1,
                 window_size=None, seed=0):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.input_size = input_size
        self.width = width
        self.depth = depth
        self.output_size = output_size
        self.window_size = window_size

        rng = np.random.default_rng(seed)
        sizes = [input_size] + [width] * depth + [output_size]
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He initialization, as in MLPRelay
            self.W.append(rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in))
            self.b.append(np.zeros(n_out))

        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @property
    def num_params(self):
        return sum(p.size for p in self.params)

    def _extract_windows(self, y):
        if self.window_size is None:
            return y.reshape(-1, self.input_size) if y.ndim == 1 else y
        pad = self.window_size // 2
        yp = np.pad(y, (pad, pad), mode="constant")
        return np.lib.stride_tricks.sliding_window_view(yp, self.window_size)

    def fwd(self, X):
        """Forward pass; caches activations for the backward pass."""
        self._acts = [X]
        a = X
        for W, b in zip(self.W, self.b):
            a = np.tanh(a @ W + b)
            self._acts.append(a)
        return a.ravel() if self.output_size == 1 else a

    def step(self, X, target, lr=3e-3):
        """One Adam step on MSE loss. Mirrors MLPRelay.step generalized to L."""
        output = self.fwd(X)
        batch = X.shape[0]

        # dL/d(pre-activation) at the output, through tanh
        out_a = self._acts[-1]
        if self.output_size == 1:
            delta = (2 * (output - target) / batch * (1 - output ** 2))[:, None]
        else:
            delta = 2 * (out_a - target) / batch * (1 - out_a ** 2)

        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            a_in = self._acts[i]
            gW[i] = a_in.T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1 - self._acts[i] ** 2)

        self.t += 1
        for p, g, m, v in zip(self.params, gW + gb, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

        return float(np.mean((output - target) ** 2))

    def train_on_data(self, X, target, epochs=25, batch_size=256, lr=3e-3):
        idx = np.arange(X.shape[0])
        rng = np.random.default_rng(42)
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                b = idx[i:i + batch_size]
                self.step(X[b], target[b], lr=lr)

    def process(self, received_signal):
        windows = self._extract_windows(received_signal)
        output = self.fwd(windows)
        power = np.sqrt(np.mean(output ** 2)) + 1e-12
        return output / power


def n_params(window, width, depth, input_size=None, output_size=1):
    """Closed form, for grids that need the count before building anything."""
    n_in = window if input_size is None else input_size
    p = n_in * width + width                      # input -> first hidden
    p += (depth - 1) * (width * width + width)    # hidden -> hidden
    p += width * output_size + output_size        # last hidden -> output
    return p


def test_deep_mlp_matches_single_layer(verbose=True):
    """Depth 1 must reproduce MLPRelay exactly: same init, same updates,
    same outputs. If this drifts, depth-1 rows in the sweep are no longer
    comparable with the single-layer study and the whole comparison breaks."""
    from relaynet.relays import MLPRelay

    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 5))
    T = np.sign(rng.standard_normal(512))

    a = MLPRelay(input_size=5, hidden_size=7, output_size=1,
                 window_size=5, seed=3)
    b = DeepMLPRelay(input_size=5, width=7, depth=1, output_size=1,
                     window_size=5, seed=3)

    assert np.allclose(a.W1, b.W[0]), "initial W1 differs"
    assert np.allclose(a.W2, b.W[1]), "initial W2 differs"
    assert np.allclose(a.fwd(X), b.fwd(X)), "initial forward pass differs"

    a.train_on_data(X, T, epochs=5, batch_size=64, lr=3e-3)
    b.train_on_data(X, T, epochs=5, batch_size=64, lr=3e-3)

    assert np.allclose(a.fwd(X), b.fwd(X), atol=1e-10), "trained outputs differ"
    sig = rng.standard_normal(2000)
    assert np.allclose(a.process(sig), b.process(sig), atol=1e-10), \
        "process() differs"

    p_formula = n_params(5, 7, 1)
    assert p_formula == b.num_params == sum(p.size for p in a.params), \
        f"param count mismatch: {p_formula} vs {b.num_params}"
    assert p_formula == 7 * (5 + 2) + 1, "depth-1 formula does not collapse"

    if verbose:
        print(f"depth-1 equivalence OK  ({b.num_params} params, "
              f"matches h*(w+2)+1 = {7 * (5 + 2) + 1})")
        for d in (1, 2, 3):
            r = DeepMLPRelay(input_size=5, width=7, depth=d)
            assert r.num_params == n_params(5, 7, d)
            print(f"  depth {d}: {r.num_params} params (formula agrees)")
    return True


if __name__ == "__main__":
    test_deep_mlp_matches_single_layer()
