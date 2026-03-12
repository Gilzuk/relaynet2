"""Minimal GenAI relay (169-parameter neural network, architecture 5→24→1)."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate
from relaynet.channels.awgn import awgn_channel


class _TinyNN:
    """Two-layer feedforward network with ReLU hidden and tanh output."""

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return np.tanh(self.z2)

    def train_step(self, X, y, lr=0.01):
        batch_size = X.shape[0]
        output = self.forward(X)
        loss = np.mean((output - y) ** 2)

        dz2 = 2 * (output - y) / batch_size * (1 - output ** 2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss


class MinimalGenAIRelay(Relay):
    """Minimal GenAI relay using a tiny (169-parameter) neural network.

    Default architecture: window_size=5, hidden=24, output=1  → 5*24+24+24*1+1 = 169 params.
    Uses only NumPy — no external ML framework dependency.
    """

    def __init__(self, window_size=5, hidden_size=24, target_power=1.0):
        self.window_size = window_size
        self.target_power = target_power
        self.nn = _TinyNN(window_size, hidden_size, 1)
        self.is_trained = False

    @property
    def num_params(self):
        ws = self.window_size
        hs = self.nn.W2.shape[0]
        return ws * hs + hs + hs * 1 + 1

    def train(self, training_snrs=None, num_samples=25000, epochs=100, seed=None):
        """Train the relay on simulated AWGN data.

        Parameters
        ----------
        training_snrs : list of float, optional
            SNR values (dB) used for training data generation.
            Defaults to [5, 10, 15].
        num_samples : int
            Total training samples across all SNRs.
        epochs : int
            Training epochs.
        seed : int, optional
            Random seed for reproducibility.
        """
        if training_snrs is None:
            training_snrs = [5, 10, 15]
        if seed is not None:
            np.random.seed(seed)

        samples_per_snr = num_samples // len(training_snrs)
        X_all, y_all = [], []
        pad = self.window_size // 2

        for snr in training_snrs:
            np.random.seed(42 + int(snr))
            bits = np.random.randint(0, 2, samples_per_snr)
            clean = bpsk_modulate(bits)
            noisy = awgn_channel(clean, snr)
            for i in range(pad, len(noisy) - pad):
                X_all.append(noisy[i - pad: i + pad + 1])
                y_all.append(clean[i])

        X = np.array(X_all)
        y = np.array(y_all).reshape(-1, 1)

        batch_size = 32
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                self.nn.train_step(X[idx[i: i + batch_size]], y[idx[i: i + batch_size]])

        self.is_trained = True

    def process(self, received_signal):
        if not self.is_trained:
            return received_signal * 1.5

        pad = self.window_size // 2
        padded = np.pad(received_signal, pad, mode="edge")
        processed = np.array([
            self.nn.forward(padded[i: i + self.window_size].reshape(1, -1))[0, 0]
            for i in range(len(received_signal))
        ])

        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            processed *= np.sqrt(self.target_power / current_power)
        return processed
