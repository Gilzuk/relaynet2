"""Variational Autoencoder (VAE) relay."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate
from relaynet.channels.awgn import awgn_channel
from relaynet.utils.torch_compat import can_use_gpu, get_preferred_device, get_torch_module, to_numpy


def _build_torch_vae(window_size, latent_size, beta, device):
    """Build a Torch VAE backend for optional GPU training/inference."""
    torch = get_torch_module()
    import torch.nn as nn
    import torch.optim as optim

    class TorchVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(window_size, 32)
            self.enc2 = nn.Linear(32, 16)
            self.mu = nn.Linear(16, latent_size)
            self.logvar = nn.Linear(16, latent_size)
            self.dec1 = nn.Linear(latent_size, 16)
            self.dec2 = nn.Linear(16, 32)
            self.out = nn.Linear(32, 1)

        def encode(self, x):
            h1 = torch.relu(self.enc1(x))
            h2 = torch.relu(self.enc2(h1))
            return self.mu(h2), self.logvar(h2)

        def decode(self, z):
            d1 = torch.relu(self.dec1(z))
            d2 = torch.relu(self.dec2(d1))
            return torch.tanh(self.out(d2))

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
        y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)
        recon, mu, logvar = model(X_t)
        recon_loss = torch.mean((recon - y_t) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def infer(window_np):
        with torch.no_grad():
            X_t = torch.as_tensor(window_np, dtype=torch.float32, device=device)
            out = model.infer(X_t)
        return to_numpy(out.flatten(), dtype=float)

    return {"train_step": train_step, "infer": infer}


class VAERelay(Relay):
    """VAE-based relay that learns a probabilistic latent representation.

    Architecture:
        Encoder : noisy_window  → μ, log_σ²  (latent_size dims)
        Decoder : z            → clean_symbol (1 dim)

    Loss: MSE reconstruction + β * KL divergence.
    Uses only NumPy — no external ML framework dependency.
    """

    def __init__(self, window_size=7, latent_size=8, beta=0.1, target_power=1.0, prefer_gpu=True):
        self.window_size = window_size
        self.latent_size = latent_size
        self.beta = beta
        self.target_power = target_power
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)
        self._use_torch = can_use_gpu(self.device)

        inp = window_size
        h1, h2 = 32, 16

        # Encoder
        self.W_e1 = np.random.randn(inp, h1) * np.sqrt(2 / inp)
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
        self.W_out = np.random.randn(h1, 1) * 0.1
        self.b_out = np.zeros(1)

        self._torch_vae = _build_torch_vae(window_size, latent_size, beta, self.device) if self._use_torch else None

        self.is_trained = False

    def _encode(self, X):
        h1 = np.maximum(0, X @ self.W_e1 + self.b_e1)
        h2 = np.maximum(0, h1 @ self.W_e2 + self.b_e2)
        mu = h2 @ self.W_mu + self.b_mu
        lv = h2 @ self.W_lv + self.b_lv
        return mu, lv, h1, h2

    def _decode(self, z):
        d1 = np.maximum(0, z @ self.W_d1 + self.b_d1)
        d2 = np.maximum(0, d1 @ self.W_d2 + self.b_d2)
        out = np.tanh(d2 @ self.W_out + self.b_out)
        return out, d1, d2

    def _forward(self, X):
        mu, lv, h1_e, h2_e = self._encode(X)
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
        d_recon = 2 * (recon - y) / bs * (1 - recon ** 2)
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
        d_lv_kl = self.beta * (-0.5 + 0.5 * np.exp(lv)) / bs

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

    def train(self, training_snrs=None, num_samples=50000, epochs=100, seed=None):
        """Train the VAE relay.

        Parameters
        ----------
        training_snrs : list of float, optional
            Defaults to [5, 10, 15].
        num_samples : int
        epochs : int
        seed : int, optional
        """
        if training_snrs is None:
            training_snrs = [5, 10, 15]
        if seed is not None:
            np.random.seed(seed)

        pad = self.window_size // 2
        samples_per_snr = num_samples // len(training_snrs)
        X_all, y_all = [], []

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

        batch_size = 64
        if self._use_torch:
            for _ in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    sl = idx[i: i + batch_size]
                    self._torch_vae["train_step"](X[sl], y[sl])
        else:
            for _ in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    self._train_step(X[idx[i: i + batch_size]], y[idx[i: i + batch_size]])

        self.is_trained = True

    def _infer(self, window):
        """Deterministic inference using encoder mean (no reparameterisation)."""
        mu, _, _, _ = self._encode(window)
        out, _, _ = self._decode(mu)
        return out

    def process(self, received_signal):
        if not self.is_trained:
            return received_signal

        pad = self.window_size // 2
        padded = np.pad(received_signal, pad, mode="edge")
        windows = np.array([
            padded[i: i + self.window_size]
            for i in range(len(received_signal))
        ])

        if self._use_torch:
            processed = self._torch_vae["infer"](windows)
        else:
            processed = np.zeros(len(received_signal))
            for i in range(len(received_signal)):
                processed[i] = self._infer(windows[i].reshape(1, -1))[0, 0]

        pwr = np.mean(np.abs(processed) ** 2)
        if pwr > 0:
            processed *= np.sqrt(self.target_power / pwr)
        return processed
