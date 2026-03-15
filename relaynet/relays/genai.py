"""Minimal GenAI relay (169-parameter neural network, architecture 5→24→1)."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate
from relaynet.channels.awgn import awgn_channel
from relaynet.utils.torch_compat import can_use_gpu, get_preferred_device, get_torch_module, to_numpy


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


def _build_torch_tinynn(input_size, hidden_size, device):
    """Create a tiny Torch model matching the 169-parameter network."""
    torch = get_torch_module()
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.Tanh(),
    ).to(device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


class MinimalGenAIRelay(Relay):
    """Minimal GenAI relay using a tiny (169-parameter) neural network.

    Default architecture: window_size=5, hidden=24, output=1  → 5*24+24+24*1+1 = 169 params.
    Uses only NumPy — no external ML framework dependency.
    """

    def __init__(self, window_size=5, hidden_size=24, target_power=1.0, prefer_gpu=True):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.target_power = target_power
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)
        self._use_torch = can_use_gpu(self.device)
        self.nn = _TinyNN(window_size, hidden_size, 1)
        self._torch_model = _build_torch_tinynn(window_size, hidden_size, self.device) if self._use_torch else None
        self.is_trained = False

    @property
    def num_params(self):
        ws = self.window_size
        hs = self.hidden_size
        return ws * hs + hs + hs * 1 + 1

    def train(self, training_snrs=None, num_samples=25000, epochs=100, seed=None,
              epoch_callback=None):
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
        epoch_callback : callable, optional
            Called as ``epoch_callback(epoch, epochs)`` after each epoch.
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
        if self._use_torch:
            torch = get_torch_module()
            import torch.nn as nn
            import torch.optim as optim

            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
            optimizer = optim.Adam(self._torch_model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            for _ep in range(epochs):
                idx = torch.randperm(X_t.size(0), device=self.device)
                for i in range(0, X_t.size(0), batch_size):
                    sl = idx[i: i + batch_size]
                    output = self._torch_model(X_t[sl])
                    loss = criterion(output, y_t[sl])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch_callback:
                    epoch_callback(_ep, epochs)
        else:
            for _ep in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    self.nn.train_step(X[idx[i: i + batch_size]], y[idx[i: i + batch_size]])
                if epoch_callback:
                    epoch_callback(_ep, epochs)

        self.is_trained = True

    def process(self, received_signal):
        if not self.is_trained:
            return received_signal * 1.5

        pad = self.window_size // 2
        padded = np.pad(received_signal, pad, mode="edge")
        windows = np.array([
            padded[i: i + self.window_size]
            for i in range(len(received_signal))
        ])

        if self._use_torch:
            torch = get_torch_module()
            with torch.no_grad():
                window_t = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
                processed = to_numpy(self._torch_model(window_t).flatten(), dtype=float)
        else:
            processed = self.nn.forward(windows).flatten()

        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            processed *= np.sqrt(self.target_power / current_power)
        return processed

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Save trained weights to *path*."""
        from relaynet.utils.torch_compat import save_state
        state = {
            "type": "MinimalGenAIRelay",
            "config": {"window_size": self.window_size,
                       "hidden_size": self.hidden_size},
        }
        if self._use_torch and self._torch_model is not None:
            state["torch_state_dict"] = self._torch_model.state_dict()
        else:
            state["numpy"] = {
                "W1": self.nn.W1, "b1": self.nn.b1,
                "W2": self.nn.W2, "b2": self.nn.b2,
            }
        save_state(state, path)

    def load_weights(self, path):
        """Load trained weights from *path* and mark the relay as trained."""
        from relaynet.utils.torch_compat import load_state
        state = load_state(path)
        if "torch_state_dict" in state and self._use_torch and self._torch_model is not None:
            self._torch_model.load_state_dict(state["torch_state_dict"])
        elif "numpy" in state:
            nw = state["numpy"]
            self.nn.W1, self.nn.b1 = nw["W1"], nw["b1"]
            self.nn.W2, self.nn.b2 = nw["W2"], nw["b2"]
        self.is_trained = True
