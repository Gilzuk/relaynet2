"""Shared output-activation helpers for relay neural networks.

Supports three output activations:

* ``"tanh"`` – standard :math:`\\tanh` (default, bounded to [-1, +1]).
* ``"linear"`` – identity / no activation (unbounded).
* ``"hardtanh"`` – clipped linear in
  :math:`[-3/\\sqrt{10},\\; +3/\\sqrt{10}]` matching the 16-QAM
  per-axis range.
"""

import numpy as np

# 16-QAM outer level: 3 / sqrt(10) ≈ 0.9487
QAM16_CLIP = 3.0 / np.sqrt(10.0)


# ── NumPy helpers ─────────────────────────────────────────────────

def apply_activation(z, activation="tanh"):
    """Apply the chosen output activation (NumPy)."""
    if activation == "tanh":
        return np.tanh(z)
    if activation == "hardtanh":
        return np.clip(z, -QAM16_CLIP, QAM16_CLIP)
    # "linear"
    return z


def activation_derivative(output, z, activation="tanh"):
    """Element-wise derivative of the output activation (NumPy)."""
    if activation == "tanh":
        return 1 - output ** 2
    if activation == "hardtanh":
        return ((z > -QAM16_CLIP) & (z < QAM16_CLIP)).astype(float)
    # "linear"
    return np.ones_like(output)


# ── PyTorch helper ────────────────────────────────────────────────

def make_torch_activation(activation="tanh"):
    """Return a ``torch.nn.Module`` for the chosen output activation."""
    import torch.nn as nn

    if activation == "tanh":
        return nn.Tanh()
    if activation == "hardtanh":
        return nn.Hardtanh(-QAM16_CLIP, QAM16_CLIP)
    # "linear"
    return nn.Identity()


# ── Training-data generator ──────────────────────────────────────

def generate_training_targets(num_samples, snr_db, training_modulation="bpsk",
                              seed=None):
    """Generate *clean* and *noisy* 1-D real training symbols.

    Parameters
    ----------
    num_samples : int
    snr_db : float
    training_modulation : str
        ``"bpsk"`` → targets ∈ {-1, +1}.
        ``"qam16"`` → targets ∈ {-3, -1, +1, +3}/√10 (one I/Q axis).
    seed : int, optional

    Returns
    -------
    clean : ndarray, shape (num_samples,)
    noisy : ndarray, shape (num_samples,)
    """
    from relaynet.modulation.bpsk import bpsk_modulate
    from relaynet.channels.awgn import awgn_channel

    if seed is not None:
        np.random.seed(seed)

    if training_modulation == "qam16":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        indices = np.random.randint(0, 4, num_samples)
        clean = levels[indices]
    else:  # bpsk
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits)

    noisy = awgn_channel(clean, snr_db)
    return clean, noisy
