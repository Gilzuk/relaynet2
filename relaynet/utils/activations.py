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
    "psk16": 1.0,
}

# ── Per-axis constellation levels for classification-based relaying ──

CONSTELLATION_LEVELS = {
    "bpsk":  np.array([-1.0, 1.0]),
    "qpsk":  np.array([-1.0, 1.0]) / np.sqrt(2.0),
    "qam16": np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0),
    "psk16": np.sort(np.unique(np.round(
        np.cos(2.0 * np.pi * np.arange(16) / 16.0), decimals=10,
    ))),  # 9 unique I-axis projections
}

# ── Full 2-D constellation for 16-class QAM16 classification ────
_QAM16_LEVELS_1D = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
CONSTELLATION_2D_QAM16 = np.array([
    i_val + 1j * q_val
    for i_val in _QAM16_LEVELS_1D
    for q_val in _QAM16_LEVELS_1D
])  # (16,) complex – I varies slow, Q varies fast


def get_num_classes(modulation, classify_2d=False):
    """Return the number of constellation classes for *modulation*.

    When *classify_2d* is True and modulation is ``"qam16"``, returns 16
    (full 2-D constellation) instead of 4 (per-axis levels).
    """
    if classify_2d and modulation.lower() == "qam16":
        return 16
    levels = CONSTELLATION_LEVELS.get(modulation.lower())
    if levels is None:
        return 1
    return len(levels)


def get_constellation_levels(modulation):
    """Return sorted per-axis constellation levels as a 1-D numpy array."""
    return CONSTELLATION_LEVELS[modulation.lower()]


def get_constellation_2d(modulation):
    """Return the full 2-D complex constellation as a 1-D complex array."""
    if modulation.lower() == "qam16":
        return CONSTELLATION_2D_QAM16.copy()
    raise ValueError(f"2-D constellation not defined for {modulation}")


def symbols_to_class_indices(symbols, modulation):
    """Map 1-D real symbols to nearest constellation-level class index."""
    levels = CONSTELLATION_LEVELS[modulation.lower()]
    syms = np.asarray(symbols).reshape(-1, 1)
    diffs = np.abs(syms - levels.reshape(1, -1))
    return np.argmin(diffs, axis=1).astype(np.int64)


def complex_symbols_to_2d_class_indices(symbols):
    """Map complex QAM16 symbols to nearest of 16 constellation points -> index 0..15."""
    syms = np.asarray(symbols).ravel()
    dist = np.abs(syms[:, None] - CONSTELLATION_2D_QAM16[None, :])  # (N, 16)
    return np.argmin(dist, axis=1).astype(np.int64)


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
                              seed=None, use_rayleigh=False, return_csi=False):
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
    use_rayleigh : bool, optional
        If True, apply Rayleigh fading channel instead of AWGN.
    return_csi : bool, optional
        If True and use_rayleigh is True, returns (clean, noisy, h).

    Returns
    -------
    clean : ndarray, shape (num_samples,)
    noisy : ndarray, shape (num_samples,)
    h : ndarray, shape (num_samples,), optional (if return_csi is True)
    """
    from relaynet.modulation.bpsk import bpsk_modulate
    from relaynet.channels.awgn import awgn_channel
    from relaynet.channels.fading import rayleigh_fading_channel

    if seed is not None:
        np.random.seed(seed)

    if training_modulation == "qam16":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        indices = np.random.randint(0, 4, num_samples)
        clean = levels[indices]
    elif training_modulation == "psk16":
        # 16-PSK: 16 equally-spaced points on the unit circle.
        # For 1-D per-axis training, project onto I or Q axis.
        angles = 2.0 * np.pi * np.arange(16) / 16.0
        cos_vals = np.cos(angles)  # I-axis projections
        indices = np.random.randint(0, 16, num_samples)
        clean = cos_vals[indices]
    elif training_modulation == "qpsk":
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits) / np.sqrt(2.0)
    else:  # bpsk
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits)

    if use_rayleigh:
        noisy, h = rayleigh_fading_channel(clean, snr_db, return_channel=True)
        if return_csi:
            return clean, noisy, h
    else:
        noisy = awgn_channel(clean, snr_db)
        if return_csi:
            return clean, noisy, np.ones_like(noisy, dtype=np.complex128)
            
    return clean, noisy


def generate_training_targets_2d(num_samples, snr_db, seed=None):
    """Generate complex QAM16 training data for 16-class classification.

    Returns
    -------
    clean : ndarray, shape (num_samples,), complex
        Clean complex QAM16 symbols.
    noisy : ndarray, shape (num_samples,), complex
        Noisy complex QAM16 symbols after AWGN.
    labels : ndarray, shape (num_samples,), int64
        Class indices 0..15 for 16-class classification.
    """
    from relaynet.channels.awgn import awgn_channel

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.randint(0, 16, num_samples)
    clean = CONSTELLATION_2D_QAM16[indices]  # complex
    noisy = awgn_channel(clean, snr_db)

    return clean, noisy, indices.astype(np.int64)
