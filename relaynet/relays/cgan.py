"""
CGAN relay improved with Wasserstein GAN + Gradient Penalty (WGAN-GP).

Improvements over vanilla GAN:
- Wasserstein loss (critic outputs unbounded scores, not probabilities)
- Gradient penalty (λ=10) instead of weight clipping
- 5 critic updates per generator update
- Spectral normalisation on the critic
- Reduced L1 reconstruction weight (default 20)

Falls back to NumPy-only training when PyTorch is not available.
"""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate
from relaynet.channels.awgn import awgn_channel
from relaynet.utils.torch_compat import get_preferred_device, to_numpy


# ---------------------------------------------------------------------------
# NumPy fallback implementation
# ---------------------------------------------------------------------------

class _GenNP:
    """Small generator network (NumPy)."""

    def __init__(self, window_size=7, noise_size=8):
        inp = window_size + noise_size
        self.W1 = np.random.randn(inp, 32) * np.sqrt(2 / inp)
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 32) * np.sqrt(2 / 32)
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 16) * np.sqrt(2 / 32)
        self.b3 = np.zeros(16)
        self.W4 = np.random.randn(16, 1) * 0.1
        self.b4 = np.zeros(1)

    def forward(self, noisy, noise):
        x = np.concatenate([noisy, noise], axis=1)
        self._x = x
        self._h1 = np.maximum(0.2 * (x @ self.W1 + self.b1), x @ self.W1 + self.b1)
        self._h2 = np.maximum(0.2 * (self._h1 @ self.W2 + self.b2), self._h1 @ self.W2 + self.b2)
        self._h3 = np.maximum(0.2 * (self._h2 @ self.W3 + self.b3), self._h2 @ self.W3 + self.b3)
        out = np.tanh(self._h3 @ self.W4 + self.b4)
        self._out = out
        return out


class _CritNP:
    """Small critic network (NumPy). No sigmoid — Wasserstein critic."""

    def __init__(self, window_size=7):
        inp = 1 + window_size
        self.W1 = np.random.randn(inp, 32) * np.sqrt(2 / inp)
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 16) * np.sqrt(2 / 32)
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16, 1) * 0.1
        self.b3 = np.zeros(1)

    def forward(self, signal, condition):
        x = np.concatenate([signal, condition], axis=1)
        self._x = x
        self._h1 = np.maximum(0.2 * (x @ self.W1 + self.b1), x @ self.W1 + self.b1)
        self._h2 = np.maximum(0.2 * (self._h1 @ self.W2 + self.b2), self._h1 @ self.W2 + self.b2)
        out = self._h2 @ self.W3 + self.b3  # unbounded
        self._out = out
        return out


# ---------------------------------------------------------------------------
# PyTorch implementation (used when torch is available)
# ---------------------------------------------------------------------------

def _build_torch_cgan(window_size, noise_size, lambda_gp, lambda_l1, n_critic,
                      device, g_hidden_sizes=(32, 32, 16),
                      c_hidden_sizes=(32, 16)):
    """Return a PyTorch-based CGAN relay trainer (dict of objects)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_dim = window_size + noise_size
            for h in g_hidden_sizes:
                layers.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2)])
                in_dim = h
            layers.extend([nn.Linear(in_dim, 1), nn.Tanh()])
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, noisy, noise):
            return self.net(torch.cat([noisy, noise], dim=1))

    class Critic(nn.Module):
        """Wasserstein critic with spectral normalisation."""
        def __init__(self):
            super().__init__()
            layers = []
            in_dim = 1 + window_size
            for h in c_hidden_sizes:
                layers.extend([
                    nn.utils.spectral_norm(nn.Linear(in_dim, h)),
                    nn.LeakyReLU(0.2),
                ])
                in_dim = h
            layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, 1)))
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, signal, condition):
            return self.net(torch.cat([signal, condition], dim=1))

    G = Generator().to(device)
    C = Critic().to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def gradient_penalty(real_sig, fake_sig, condition):
        bs = real_sig.size(0)
        alpha = torch.rand(bs, 1, device=real_sig.device)
        interp = (alpha * real_sig + (1 - alpha) * fake_sig).requires_grad_(True)
        d_interp = C(interp, condition)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train_step(X_batch, y_batch):
        X_t = torch.as_tensor(X_batch, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)

        # --- Critic steps ---
        for _ in range(n_critic):
            noise = torch.randn(X_t.size(0), noise_size, device=device)
            fake = G(X_t, noise).detach()
            loss_C = (
                -C(y_t, X_t).mean()
                + C(fake, X_t).mean()
                + lambda_gp * gradient_penalty(y_t, fake, X_t)
            )
            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

        # --- Generator step ---
        noise = torch.randn(X_t.size(0), noise_size, device=device)
        fake = G(X_t, noise)
        loss_G = -C(fake, X_t).mean() + lambda_l1 * torch.mean(torch.abs(fake - y_t))
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        return loss_G.item()

    def infer(window_np):
        import torch
        G.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(window_np, dtype=torch.float32, device=device)
            noise = torch.zeros(x_t.size(0), noise_size, device=device)
            # Avoid calling Tensor.numpy(): some torch builds (compiled against
            # NumPy 1.x) can raise "RuntimeError: Numpy is not available" when
            # used with NumPy 2.x. Converting via Python lists avoids the
            # torch↔numpy bridge.
            out = G(x_t, noise)
        G.train()
        return to_numpy(out.flatten(), dtype=float)

    return {"train_step": train_step, "infer": infer}


# ---------------------------------------------------------------------------
# Public CGANRelay class
# ---------------------------------------------------------------------------

class CGANRelay(Relay):
    """CGAN relay with WGAN-GP training.

    Automatically uses PyTorch if available; otherwise falls back to a
    NumPy-based vanilla GAN (less stable but dependency-free).
    """

    def __init__(self, window_size=7, noise_size=8, target_power=1.0,
                 lambda_gp=10, lambda_l1=20, n_critic=5, prefer_gpu=True,
                 g_hidden_sizes=(32, 32, 16), c_hidden_sizes=(32, 16)):
        self.window_size = window_size
        self.noise_size = noise_size
        self.target_power = target_power
        self.lambda_gp = lambda_gp
        self.lambda_l1 = lambda_l1
        self.n_critic = n_critic
        self.g_hidden_sizes = g_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.is_trained = False
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)

        self._use_torch = False
        try:
            import torch  # noqa: F401
            self._use_torch = True
            self._torch_model = _build_torch_cgan(
                window_size, noise_size, lambda_gp, lambda_l1, n_critic,
                self.device, g_hidden_sizes, c_hidden_sizes,
            )
        except ImportError:
            self._gen = _GenNP(window_size, noise_size)
            self._crit = _CritNP(window_size)

    @property
    def num_params(self):
        g_in = self.window_size + self.noise_size
        g = 0
        for h in self.g_hidden_sizes:
            g += g_in * h + h
            g_in = h
        g += g_in * 1 + 1  # output layer
        c_in = 1 + self.window_size
        c = 0
        for h in self.c_hidden_sizes:
            c += c_in * h + h
            c_in = h
        c += c_in * 1 + 1
        return g + c

    def train(self, training_snrs=None, num_samples=50000, epochs=200, seed=None,
              epoch_callback=None):
        """Train the CGAN relay.

        Parameters
        ----------
        training_snrs : list of float, optional
            Defaults to [5, 10, 15].
        num_samples : int
        epochs : int
        seed : int, optional
        epoch_callback : callable, optional
            Called as ``epoch_callback(epoch, epochs)`` after each epoch.
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
            for _ep in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    sl = idx[i: i + batch_size]
                    self._torch_model["train_step"](X[sl], y[sl])
                if epoch_callback:
                    epoch_callback(_ep, epochs)
        else:
            # NumPy fallback — simple supervised pre-training only
            gen = self._gen
            lr = 0.001
            for _ep in range(epochs):
                idx = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    sl = idx[i: i + batch_size]
                    Xb, yb = X[sl], y[sl]
                    noise = np.random.randn(len(Xb), self.noise_size)
                    fake = gen.forward(Xb, noise)
                    # L1 reconstruction gradient: d|fake - y|/d_fake = sign(fake - y)
                    grad_out = np.sign(fake - yb) * self.lambda_l1
                    # Backprop through output tanh layer
                    d_tanh = 1 - fake ** 2
                    d4 = grad_out * d_tanh
                    gen.W4 -= lr * gen._h3.T @ d4
                    gen.b4 -= lr * np.sum(d4, axis=0)
                if epoch_callback:
                    epoch_callback(_ep, epochs)

        self.is_trained = True

    def process(self, received_signal):
        if not self.is_trained:
            return received_signal

        pad = self.window_size // 2
        padded = np.pad(received_signal, pad, mode="edge")
        n = len(received_signal)

        windows = np.array([padded[i: i + self.window_size] for i in range(n)])

        if self._use_torch:
            processed = self._torch_model["infer"](windows)
        else:
            noise = np.zeros((n, self.noise_size))
            processed = self._gen.forward(windows, noise).flatten()

        pwr = np.mean(np.abs(processed) ** 2)
        if pwr > 0:
            processed *= np.sqrt(self.target_power / pwr)
        return processed
