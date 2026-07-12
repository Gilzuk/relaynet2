"""Minimal MLP relay (169-parameter neural network, architecture 5→24→1)."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate
from relaynet.channels.awgn import awgn_channel
from relaynet.utils.torch_compat import can_use_gpu, get_preferred_device, get_torch_module, to_numpy
from relaynet.utils.activations import (
    apply_activation, activation_derivative, make_torch_activation,
    generate_training_targets, generate_training_targets_2d,
    get_num_classes, get_constellation_levels, symbols_to_class_indices,
    get_constellation_2d, complex_symbols_to_2d_class_indices,
)


class _TinyNN:
    """Two-layer feedforward network with ReLU hidden and configurable output."""

    def __init__(self, input_size, hidden_size, output_size, output_activation="tanh",
                 clip_range=None, classify=False):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        self.output_activation = output_activation
        self.clip_range = clip_range
        self.classify = classify

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        if self.classify:
            return self.z2  # raw logits for classification
        return apply_activation(self.z2, self.output_activation, clip_range=self.clip_range)

    def train_step(self, X, y, lr=0.01):
        batch_size = X.shape[0]
        if self.classify:
            # y is integer class indices, shape (batch,)
            logits = self.forward(X)  # (batch, num_classes)
            probs = self._softmax(logits)
            labels = y.astype(int).ravel()
            loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-10))
            dz2 = probs.copy()
            dz2[np.arange(batch_size), labels] -= 1
            dz2 /= batch_size
        else:
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            dz2 = 2 * (output - y) / batch_size * activation_derivative(output, self.z2, self.output_activation, clip_range=self.clip_range)

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


def _build_torch_tinynn(input_size, hidden_size, device, output_activation="tanh",
                       clip_range=None, num_classes=1):
    """Create a tiny Torch model matching the 169-parameter network."""
    torch = get_torch_module()
    import torch.nn as nn

    layers = [
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
    ]
    if num_classes > 1:
        layers.append(nn.Linear(hidden_size, num_classes))
    else:
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(make_torch_activation(output_activation, clip_range=clip_range))
    model = nn.Sequential(*layers).to(device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


class MinimalGenAIRelay(Relay):
    """Minimal MLP (dual-layer perceptron) relay with 169 parameters.

    Default architecture: window_size=5, hidden=24, output=1  → 5*24+24+24*1+1 = 169 params.
    Uses only NumPy — no external ML framework dependency.
    """

    def __init__(self, window_size=5, hidden_size=24, target_power=1.0, prefer_gpu=True,
                 output_activation="tanh", clip_range=None,
                 classify=False, training_modulation="bpsk",
                 classify_2d=False):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.target_power = target_power
        self.output_activation = output_activation
        self.clip_range = clip_range
        self.classify = classify
        self.classify_2d = classify_2d
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)
        self._use_torch = can_use_gpu(self.device)
        # Prefer PyTorch even on CPU when available (NumPy path has
        # numerical issues with classification and larger networks).
        if not self._use_torch:
            torch_mod = get_torch_module()
            if torch_mod is not None:
                self.device = torch_mod.device("cpu")
                self._use_torch = True

        if classify_2d:
            self.classify = True
            self.num_classes = 16
            self._constellation_2d = get_constellation_2d("qam16")
            self._constellation_levels_np = None
            self._training_modulation = training_modulation
        elif classify:
            self.num_classes = get_num_classes(training_modulation)
            self._constellation_levels_np = get_constellation_levels(training_modulation)
            self._constellation_2d = None
            self._training_modulation = training_modulation
        else:
            self.num_classes = 1
            self._constellation_levels_np = None
            self._constellation_2d = None
            self._training_modulation = training_modulation

        input_size = 2 * window_size if classify_2d else window_size
        out_size = self.num_classes if self.num_classes > 1 else 1
        self.nn = _TinyNN(input_size, hidden_size, out_size,
                          output_activation=output_activation,
                          clip_range=clip_range, classify=self.classify)
        self._torch_model = _build_torch_tinynn(
            input_size, hidden_size, self.device,
            output_activation=output_activation,
            clip_range=clip_range,
            num_classes=self.num_classes,
        ) if self._use_torch else None
        self.is_trained = False

    @property
    def num_params(self):
        ws = 2 * self.window_size if self.classify_2d else self.window_size
        hs = self.hidden_size
        out = self.num_classes if self.num_classes > 1 else 1
        return ws * hs + hs + hs * out + out

    def train(self, training_snrs=None, num_samples=25000, epochs=100, seed=None,
              epoch_callback=None, training_modulation="bpsk"):
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
        training_modulation : str, optional
            ``'bpsk'`` (default) or ``'qam16'`` — controls what target
            symbols are generated for training.
        """
        if training_snrs is None:
            training_snrs = [5, 10, 15]
        if seed is not None:
            np.random.seed(seed)

        samples_per_snr = num_samples // len(training_snrs)
        X_all, y_all = [], []
        pad = self.window_size // 2

        if self.classify_2d:
            # 16-class 2D: generate complex QAM16 data, windows = [I_win, Q_win]
            y_cls_all = []
            for snr in training_snrs:
                _clean, noisy_c, labels = generate_training_targets_2d(
                    samples_per_snr, snr, seed=42 + int(snr),
                )
                rx_I = np.pad(noisy_c.real, pad, mode="edge")
                rx_Q = np.pad(noisy_c.imag, pad, mode="edge")
                for i in range(pad, len(noisy_c) + pad):
                    i_win = rx_I[i - pad: i + pad + 1]
                    q_win = rx_Q[i - pad: i + pad + 1]
                    X_all.append(np.concatenate([i_win, q_win]))
                    y_cls_all.append(labels[i - pad])
            X = np.array(X_all, dtype=np.float32)
            y = None
            y_cls = np.array(y_cls_all, dtype=np.int64)
            use_ce = True
        else:
            for snr in training_snrs:
                clean, noisy = generate_training_targets(
                    samples_per_snr, snr,
                    training_modulation=training_modulation,
                    seed=42 + int(snr),
                )
                for i in range(pad, len(noisy) - pad):
                    X_all.append(noisy[i - pad: i + pad + 1])
                    y_all.append(clean[i])

            X = np.array(X_all)
            y = np.array(y_all).reshape(-1, 1)

            # When classifying, convert clean symbols to class indices
            use_ce = self.classify and self.num_classes > 1
            if use_ce:
                y_cls = symbols_to_class_indices(y.ravel(), training_modulation)
            else:
                y_cls = None

        batch_size = 32
        if self._use_torch:
            torch = get_torch_module()
            import torch.nn as nn
            import torch.optim as optim

            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            optimizer = optim.Adam(self._torch_model.parameters(), lr=0.01)

            if use_ce:
                y_class_t = torch.as_tensor(y_cls, dtype=torch.long, device=self.device)
                criterion = nn.CrossEntropyLoss()
                for _ep in range(epochs):
                    idx = torch.randperm(X_t.size(0), device=self.device)
                    for i in range(0, X_t.size(0), batch_size):
                        sl = idx[i: i + batch_size]
                        logits = self._torch_model(X_t[sl])  # (batch, num_classes)
                        loss = criterion(logits, y_class_t[sl])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if epoch_callback:
                        epoch_callback(_ep, epochs)
            else:
                y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
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
            if use_ce:
                for _ep in range(epochs):
                    idx = np.random.permutation(len(X))
                    for i in range(0, len(X), batch_size):
                        sl = idx[i: i + batch_size]
                        self.nn.train_step(X[sl], y_cls[sl])
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

        # ── 16-class 2D mode: handle complex signal directly ──
        if self.classify_2d and np.iscomplexobj(received_signal):
            pad = self.window_size // 2
            rx_I = np.pad(received_signal.real, pad, mode="edge")
            rx_Q = np.pad(received_signal.imag, pad, mode="edge")
            windows = np.array([
                np.concatenate([
                    rx_I[i - pad: i + pad + 1],
                    rx_Q[i - pad: i + pad + 1],
                ])
                for i in range(pad, len(received_signal) + pad)
            ], dtype=np.float32)

            if self._use_torch:
                torch = get_torch_module()
                with torch.no_grad():
                    window_t = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
                    logits = self._torch_model(window_t)
                    indices = logits.argmax(dim=-1).cpu().numpy()
            else:
                logits = self.nn.forward(windows)
                indices = np.argmax(logits, axis=-1)

            processed = self._constellation_2d[indices]  # complex
            pwr = np.mean(np.abs(processed) ** 2)
            if pwr > 0:
                processed = processed * np.sqrt(self.target_power / pwr)
            return processed

        # ── Standard per-axis mode ──
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
                out = self._torch_model(window_t)
                if self.classify and self.num_classes > 1:
                    indices = out.argmax(dim=-1).cpu().numpy()
                    processed = self._constellation_levels_np[indices]
                else:
                    processed = to_numpy(out.flatten(), dtype=float)
        else:
            out = self.nn.forward(windows)
            if self.classify and self.num_classes > 1:
                indices = np.argmax(out, axis=-1)
                processed = self._constellation_levels_np[indices]
            else:
                processed = out.flatten()

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
