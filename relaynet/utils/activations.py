"""Shared output-activation helpers for relay neural networks.

Supports five output activations:

* ``"tanh"`` – standard :math:`\\tanh` (default, bounded to [-1, +1]).
* ``"linear"`` – identity / no activation (unbounded).
* ``"hardtanh"`` – clipped linear in :math:`[-c,\\; +c]` where *c* is
  the constellation-specific clip range.
* ``"sigmoid"`` – smooth S-curve scaled to the same
  :math:`[-c,\\; +c]` range as ``hardtanh``.
* ``"scaled_tanh"`` – :math:`c\\,\\tanh(z)`, smooth and
  bounded to :math:`\\pm c`.

The clip range *c* defaults to ``QAM16_CLIP`` for backward
compatibility but can be set per constellation via the ``clip_range``
parameter or the :func:`get_clip_range` helper.
"""

import numpy as np

# 16-QAM outer level: 3 / sqrt(10) ≈ 0.9487
QAM16_CLIP = 3.0 / np.sqrt(10.0)

# Per-constellation outer symbol range (per I/Q axis).
CONSTELLATION_CLIP = {
    "bpsk":  1.0,
    "qpsk":  1.0 / np.sqrt(2.0),
    "qam16": QAM16_CLIP,
}


def get_clip_range(constellation):
    """Return the per-axis clip range for *constellation* (str or float).

    Accepts a constellation name (``"bpsk"``, ``"qpsk"``, ``"qam16"``)
    or a numeric value which is returned as-is.
    """
    if isinstance(constellation, (int, float)):
        return float(constellation)
    return CONSTELLATION_CLIP[constellation.lower()]


# ── NumPy helpers ─────────────────────────────────────────────────

def apply_activation(z, activation="tanh", clip_range=None):
    """Apply the chosen output activation (NumPy).

    Parameters
    ----------
    clip_range : float or None
        Clip / scale value for bounded activations (hardtanh, sigmoid,
        scaled_tanh).  ``None`` → ``QAM16_CLIP`` (backward compat).
    """
    if clip_range is None:
        clip_range = QAM16_CLIP
    if activation == "tanh":
        return np.tanh(z)
    if activation == "hardtanh":
        return np.clip(z, -clip_range, clip_range)
    if activation == "sigmoid":
        return clip_range * (2.0 / (1.0 + np.exp(-z)) - 1.0)
    if activation == "scaled_tanh":
        return clip_range * np.tanh(z)
    # "linear"
    return z


def activation_derivative(output, z, activation="tanh", clip_range=None):
    """Element-wise derivative of the output activation (NumPy)."""
    if clip_range is None:
        clip_range = QAM16_CLIP
    if activation == "tanh":
        return 1 - output ** 2
    if activation == "hardtanh":
        return ((z > -clip_range) & (z < clip_range)).astype(float)
    if activation == "sigmoid":
        s = (output / clip_range + 1.0) / 2.0  # recover sigmoid(z)
        return 2.0 * clip_range * s * (1.0 - s)
    if activation == "scaled_tanh":
        return clip_range * (1.0 - (output / clip_range) ** 2)
    # "linear"
    return np.ones_like(output)


# ── PyTorch helper ────────────────────────────────────────────────

def make_torch_activation(activation="tanh", clip_range=None):
    """Return a ``torch.nn.Module`` for the chosen output activation.

    Parameters
    ----------
    clip_range : float or None
        Scale / clip value for bounded activations.
        ``None`` → ``QAM16_CLIP`` (backward compat).
    """
    import torch
    import torch.nn as nn

    if clip_range is None:
        clip_range = QAM16_CLIP

    if activation == "tanh":
        return nn.Tanh()
    if activation == "hardtanh":
        return nn.Hardtanh(-clip_range, clip_range)
    if activation == "sigmoid":
        class ScaledSigmoid(nn.Module):
            def __init__(self, cr):
                super().__init__()
                self.clip_range = cr
            def forward(self, x):
                return self.clip_range * (2.0 * torch.sigmoid(x) - 1.0)
        return ScaledSigmoid(clip_range)
    if activation == "scaled_tanh":
        class ScaledTanh(nn.Module):
            def __init__(self, cr):
                super().__init__()
                self.clip_range = cr
            def forward(self, x):
                return self.clip_range * torch.tanh(x)
        return ScaledTanh(clip_range)
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
        ``"qpsk"`` → targets ∈ {-1, +1}/√2 (one I/Q axis).
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
    elif training_modulation == "qpsk":
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits) / np.sqrt(2.0)
    else:  # bpsk
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits)

    noisy = awgn_channel(clean, snr_db)
    return clean, noisy
