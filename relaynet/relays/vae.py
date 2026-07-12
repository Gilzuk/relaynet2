"""Variational Autoencoder (VAE) relay."""

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


def _build_torch_vae(window_size, latent_size, beta, device, hidden_sizes=(32, 16),
                     output_activation="tanh", clip_range=None, num_classes=1,
                     input_dim=None):
    """Build a Torch VAE backend for optional GPU training/inference."""
    torch = get_torch_module()
    import torch.nn as nn
    import torch.optim as optim

    if input_dim is None:
        input_dim = window_size
    h1, h2 = hidden_sizes
    out_dim = num_classes if num_classes > 1 else 1
    if num_classes > 1:
        out_act = nn.Identity()
    else:
        out_act = make_torch_activation(output_activation, clip_range=clip_range)

    class TorchVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(input_dim, h1)
            self.enc2 = nn.Linear(h1, h2)
            self.mu = nn.Linear(h2, latent_size)
            self.logvar = nn.Linear(h2, latent_size)
            self.dec1 = nn.Linear(latent_size, h2)
            self.dec2 = nn.Linear(h2, h1)
            self.out = nn.Linear(h1, out_dim)
            self._out_act = out_act
            self._num_classes = num_classes

        def encode(self, x):
            h1 = torch.relu(self.enc1(x))
            h2 = torch.relu(self.enc2(h1))
            return self.mu(h2), self.logvar(h2)

        def decode(self, z):
            d1 = torch.relu(self.dec1(z))
            d2 = torch.relu(self.dec2(d1))
            return self._out_act(self.out(d2))

        def forward(self, x):
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return self.decode(z), mu, logvar

        def infer(self, x):
            mu, _ = self.encode(x)
            return self.decode(mu)

    model = TorchVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_step(X_batch, y_batch):
        X_t = torch.as_tensor(X_batch, dtype=torch.float32, device=device)
        recon, mu_v, logvar = model(X_t)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu_v.pow(2) - logvar.exp())
        if num_classes > 1:
            y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device).ravel()
            recon_loss = nn.functional.cross_entropy(recon, y_t)
        else:
            y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)
            recon_loss = torch.mean((recon - y_t) ** 2)
        loss = recon_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def infer(window_np):
        with torch.no_grad():
            X_t = torch.as_tensor(window_np, dtype=torch.float32, device=device)
            out = model.infer(X_t)
        if num_classes > 1:
            return to_numpy(out, dtype=float)  # return (N, num_classes) logits
        return to_numpy(out.flatten(), dtype=float)

    return {"train_step": train_step, "infer": infer, "model": model}


class VAERelay(Relay):
    """VAE-based relay that learns a probabilistic latent representation.

    Architecture:
        Encoder : noisy_window  → μ, log_σ²  (latent_size dims)
        Decoder : z            → clean_symbol (1 dim)

    Loss: MSE reconstruction + β * KL divergence.
    Uses only NumPy — no external ML framework dependency.
    """

    def __init__(self, window_size=7, latent_size=8, beta=0.1, target_power=1.0,
                 prefer_gpu=True, hidden_sizes=(32, 16), output_activation="tanh",
                 clip_range=None, classify=False, training_modulation="bpsk",
                 classify_2d=False):
        self.window_size = window_size
        self.latent_size = latent_size
        self.beta = beta
        self.target_power = target_power
        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.clip_range = clip_range
        self.classify = classify
        self.classify_2d = classify_2d
        self._training_modulation = training_modulation
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
        elif classify:
            self.num_classes = get_num_classes(training_modulation)
            self._constellation_levels_np = get_constellation_levels(training_modulation)
            self._constellation_2d = None
        else:
            self.num_classes = 1
            self._constellation_levels_np = None
            self._constellation_2d = None

        out_size = self.num_classes if self.num_classes > 1 else 1
        inp = 2 * window_size if classify_2d else window_size
        h1, h2 = hidden_sizes

        # Encoder
        self.W_e1 = np.random.randn(inp, h1) * np.sqrt(2 / max(inp, 1))
        self.b_e1 = np.zeros(h1)
        self.W_e2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
        self.b_e2 = np.zeros(h2)
        self.W_mu = np.random.randn(h2, latent_size) * 0.1
        self.b_mu = np.zeros(latent_size)
        self.W_lv = np.random.randn(h2, latent_size) * 0.1
        self.b_lv = np.zeros(latent_size)

        # Decoder
        self.W_d1 = np.random.randn(latent_size, h2) * np.sqrt(2 / latent_size)
        self.b_d1 = np.zeros(h2)
        self.W_d2 = np.random.randn(h2, h1) * np.sqrt(2 / h2)
        self.b_d2 = np.zeros(h1)
        self.W_out = np.random.randn(h1, out_size) * 0.1
        self.b_out = np.zeros(out_size)

        self._torch_vae = _build_torch_vae(
            window_size, latent_size, beta, self.device, hidden_sizes,
            output_activation=output_activation, clip_range=clip_range,
            num_classes=self.num_classes,
            input_dim=inp,
        ) if self._use_torch else None

        self.is_trained = False

    @property
    def num_params(self):
        h1, h2 = self.hidden_sizes
        ws = 2 * self.window_size if self.classify_2d else self.window_size
        ls = self.latent_size
        out_size = self.num_classes if self.num_classes > 1 else 1
        enc = ws * h1 + h1 + h1 * h2 + h2 + h2 * ls + ls + h2 * ls + ls
        dec = ls * h2 + h2 + h2 * h1 + h1 + h1 * out_size + out_size
        return enc + dec

    def _encode(self, X):
        h1 = np.maximum(0, X @ self.W_e1 + self.b_e1)
        h2 = np.maximum(0, h1 @ self.W_e2 + self.b_e2)
        mu = h2 @ self.W_mu + self.b_mu
        lv = h2 @ self.W_lv + self.b_lv
        return mu, lv, h1, h2

    def _decode(self, z):
        d1 = np.maximum(0, z @ self.W_d1 + self.b_d1)
        d2 = np.maximum(0, d1 @ self.W_d2 + self.b_d2)
        raw = d2 @ self.W_out + self.b_out
        if self.classify and self.num_classes > 1:
            out = raw  # raw logits for classification
        else:
            out = apply_activation(raw, self.output_activation, clip_range=self.clip_range)
        return out, d1, d2

    def _forward(self, X):
        mu, lv, h1_e, h2_e = self._encode(X)
        lv = np.clip(lv, -20, 20)
        std = np.exp(0.5 * lv)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        recon, d1, d2 = self._decode(z)
        return recon, mu, lv, z, eps, std, h1_e, h2_e, d1, d2

    def _train_step(self, X, y, lr=0.001):
        bs = X.shape[0]
        recon, mu, lv, z, eps, std, h1_e, h2_e, d1, d2 = self._forward(X)

        recon_loss = np.mean((recon - y) ** 2)
        kl_loss = -0.5 * np.mean(1 + lv - mu ** 2 - np.exp(lv))
        loss = recon_loss + self.beta * kl_loss

        # Backprop through decoder
        z_out = d2 @ self.W_out + self.b_out
        d_recon = 2 * (recon - y) / bs * activation_derivative(recon, z_out, self.output_activation, clip_range=self.clip_range)
        dW_out = d2.T @ d_recon
        db_out = np.sum(d_recon, axis=0)
        d_d2 = d_recon @ self.W_out.T * (d2 > 0)
        dW_d2 = d1.T @ d_d2
        db_d2 = np.sum(d_d2, axis=0)
        d_d1 = d_d2 @ self.W_d2.T * (d1 > 0)
        dW_d1 = z.T @ d_d1
        db_d1 = np.sum(d_d1, axis=0)

        # grad w.r.t. z
        d_z = d_d1 @ self.W_d1.T

        # KL grad w.r.t. mu and lv
        d_mu_kl = self.beta * mu / bs
        d_lv_kl = self.beta * (-0.5 + 0.5 * np.exp(np.clip(lv, -20, 20))) / bs

        # Reparameterisation
        d_mu = d_z + d_mu_kl
        d_std = d_z * eps
        d_lv = d_std * std * 0.5 + d_lv_kl

        # Encoder backprop
        dW_mu = h2_e.T @ d_mu
        db_mu = np.sum(d_mu, axis=0)
        dW_lv = h2_e.T @ d_lv
        db_lv = np.sum(d_lv, axis=0)

        d_h2_e = d_mu @ self.W_mu.T + d_lv @ self.W_lv.T
        d_h2_e = d_h2_e * (h2_e > 0)
        dW_e2 = h1_e.T @ d_h2_e
        db_e2 = np.sum(d_h2_e, axis=0)
        d_h1_e = d_h2_e @ self.W_e2.T * (h1_e > 0)
        dW_e1 = X.T @ d_h1_e
        db_e1 = np.sum(d_h1_e, axis=0)

        for (W, b, dW, db) in [
            (self.W_e1, self.b_e1, dW_e1, db_e1),
            (self.W_e2, self.b_e2, dW_e2, db_e2),
            (self.W_mu, self.b_mu, dW_mu, db_mu),
            (self.W_lv, self.b_lv, dW_lv, db_lv),
            (self.W_d1, self.b_d1, dW_d1, db_d1),
            (self.W_d2, self.b_d2, dW_d2, db_d2),
            (self.W_out, self.b_out, dW_out, db_out),
        ]:
            W -= lr * dW
            b -= lr * db
        return loss

    def train(self, training_snrs=None, num_samples=50000, epochs=100, seed=None,
              epoch_callback=None, training_modulation="bpsk"):
        """Train the VAE relay.

        Parameters
        ----------
        training_snrs : list of float, optional
            Defaults to [5, 10, 15].
        num_samples : int
        epochs : int
        seed : int, optional
        epoch_callback : callable, optional
            Called as ``epoch_callback(epoch, epochs)`` after each epoch.
        training_modulation : str, optional
            ``'bpsk'`` (default) or ``'qam16'``.
        """
        if training_snrs is None:
            training_snrs = [5, 10, 15]
        if seed is not None:
            np.random.seed(seed)

        pad = self.window_size // 2
        samples_per_snr = num_samples // len(training_snrs)
        X_all, y_all = [], []

        if self.classify_2d:
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

            use_ce = self.classify and self.num_classes > 1
            if use_ce:
                y_cls = symbols_to_class_indices(y.ravel(), training_modulation)

        batch_size = 64
        if self._use_torch:
            for _ep in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    sl = idx[i: i + batch_size]
                    self._torch_vae["train_step"](X[sl], y_cls[sl] if use_ce else y[sl])
                if epoch_callback:
                    epoch_callback(_ep, epochs)
        else:
            for _ep in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    self._train_step(X[idx[i: i + batch_size]], y[idx[i: i + batch_size]])
                if epoch_callback:
                    epoch_callback(_ep, epochs)

        self.is_trained = True

    def _infer(self, window):
        """Deterministic inference using encoder mean (no reparameterisation)."""
        mu, _, _, _ = self._encode(window)
        out, _, _ = self._decode(mu)
        return out

    def process(self, received_signal):
        if not self.is_trained:
            return received_signal

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
                out_raw = self._torch_vae["infer"](windows)
                indices = np.argmax(out_raw.reshape(-1, self.num_classes), axis=-1)
            else:
                out_raw = self._infer(windows)
                indices = np.argmax(out_raw, axis=-1)

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
            out_raw = self._torch_vae["infer"](windows)
            if self.classify and self.num_classes > 1:
                # infer returns logits; do argmax + lookup
                indices = np.argmax(out_raw.reshape(-1, self.num_classes), axis=-1)
                processed = self._constellation_levels_np[indices]
            else:
                processed = out_raw
        else:
            out_raw = self._infer(windows)
            if self.classify and self.num_classes > 1:
                indices = np.argmax(out_raw, axis=-1)
                processed = self._constellation_levels_np[indices]
            else:
                processed = out_raw.flatten()

        pwr = np.mean(np.abs(processed) ** 2)
        if pwr > 0:
            processed *= np.sqrt(self.target_power / pwr)
        return processed

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Save trained weights to *path*."""
        from relaynet.utils.torch_compat import save_state
        state = {
            "type": "VAERelay",
            "config": {"window_size": self.window_size,
                       "latent_size": self.latent_size,
                       "hidden_sizes": self.hidden_sizes},
        }
        if self._use_torch and self._torch_vae is not None:
            state["torch_state_dict"] = self._torch_vae["model"].state_dict()
        else:
            state["numpy"] = {
                "W_e1": self.W_e1, "b_e1": self.b_e1,
                "W_e2": self.W_e2, "b_e2": self.b_e2,
                "W_mu": self.W_mu, "b_mu": self.b_mu,
                "W_lv": self.W_lv, "b_lv": self.b_lv,
                "W_d1": self.W_d1, "b_d1": self.b_d1,
                "W_d2": self.W_d2, "b_d2": self.b_d2,
                "W_out": self.W_out, "b_out": self.b_out,
            }
        save_state(state, path)

    def load_weights(self, path):
        """Load trained weights from *path* and mark the relay as trained."""
        from relaynet.utils.torch_compat import load_state
        state = load_state(path)
        if "torch_state_dict" in state and self._use_torch and self._torch_vae is not None:
            self._torch_vae["model"].load_state_dict(state["torch_state_dict"])
        elif "numpy" in state:
            for key, val in state["numpy"].items():
                setattr(self, key, val)
        self.is_trained = True
