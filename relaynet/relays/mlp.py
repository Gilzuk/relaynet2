"""Multi-Layer Perceptron relay for Chapter 7 experiments."""

import numpy as np
from .base import Relay


class MLPRelay(Relay):
    """Simple feed-forward MLP relay for signal processing.

    Supports windowed input processing with configurable architecture.
    Uses tanh activations and implements Adam optimization for training.

    Parameters
    ----------
    input_size : int
        Size of input (e.g., window length).
    hidden_size : int
        Number of hidden units.
    output_size : int, optional
        Output size (default 1).
    window_size : int, optional
        Window size for input extraction (if None, no windowing applied).
    seed : int, optional
        Random seed for weight initialization.
    """

    def __init__(self, input_size, hidden_size, output_size=1, window_size=None, seed=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = window_size

        # Initialize weights with He initialization
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

        # Adam optimizer state
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def _extract_windows(self, y):
        """Extract sliding windows from signal."""
        if self.window_size is None:
            # If no windowing, just reshape to (n_samples, input_size)
            return y.reshape(-1, self.input_size) if len(y.shape) == 1 else y

        # Pad and create sliding windows
        pad_size = self.window_size // 2
        yp = np.pad(y, (pad_size, pad_size), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(yp, self.window_size)
        return windows

    def fwd(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : numpy.ndarray
            Input array of shape (batch, input_size).

        Returns
        -------
        output : numpy.ndarray
            Output array of shape (batch, output_size).
        """
        self.h = np.tanh(X @ self.W1 + self.b1)
        self.o = np.tanh(self.h @ self.W2 + self.b2)
        return self.o.ravel() if self.output_size == 1 else self.o

    def step(self, X, target, lr=3e-3):
        """Training step with Adam optimizer.

        Parameters
        ----------
        X : numpy.ndarray
            Input batch of shape (batch, input_size).
        target : numpy.ndarray
            Target values.
        lr : float
            Learning rate.

        Returns
        -------
        loss : float
            MSE loss.
        """
        output = self.fwd(X)
        batch_size = X.shape[0]

        # Backprop through output layer
        do = 2 * (output - target) / batch_size * (1 - output ** 2)
        gW2 = self.h.T @ do[:, None]
        gb2 = do.sum(0, keepdims=True).ravel()

        # Backprop through hidden layer
        dh = do[:, None] @ self.W2.T * (1 - self.h ** 2)
        gW1 = X.T @ dh
        gb1 = dh.sum(0)

        # Adam update
        self.t += 1
        grads = [gW1, gb1, gW2, gb2]
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

        return np.mean((output - target) ** 2)

    def train_on_data(self, X, target, epochs=25, batch_size=256, lr=3e-3):
        """Train the network on provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size).
        target : numpy.ndarray
            Target values of shape (n_samples,).
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        """
        idx = np.arange(X.shape[0])
        rng = np.random.default_rng(42)
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                batch_idx = idx[i:i + batch_size]
                self.step(X[batch_idx], target[batch_idx], lr=lr)

    def process(self, received_signal):
        """Process received signal for relay transmission.

        Extracts windows (if configured), runs forward pass, and normalizes output power.

        Parameters
        ----------
        received_signal : numpy.ndarray
            Input signal.

        Returns
        -------
        output : numpy.ndarray
            Processed signal with unit output power.
        """
        windows = self._extract_windows(received_signal)
        output = self.fwd(windows)

        # Normalize output power to 1
        output_power = np.sqrt(np.mean(output ** 2)) + 1e-12
        return output / output_power
