"""
MIMO Channel Implementation (2×2 Spatial Multiplexing)

Implements a 2×2 MIMO Rayleigh flat-fading channel with two receiver
equalization options:

* **Zero-Forcing (ZF)** — inverts the channel; simple but amplifies
  noise when H is ill-conditioned.
* **MMSE (Minimum Mean Square Error)** — regularised inverse that
  trades a small amount of residual interference for a large noise
  reduction, yielding lower BER than ZF at every SNR.

System model
------------
The input BPSK stream of length *N* (must be even) is demultiplexed
into two spatial streams of length *N/2*:

    x = [x₁, x₂]ᵀ          (2×1 per symbol interval)

Each symbol interval uses an independent 2×2 complex channel matrix
whose entries are i.i.d. CN(0, 1):

    H = [[h₁₁, h₁₂],
         [h₂₁, h₂₂]]

The received vector is:

    y = H x + n,    n ~ CN(0, σ²I)

Equalization:

    ZF:    x̂ = H⁻¹ y
    MMSE:  x̂ = (H^H H + σ²I)⁻¹ H^H y

The two estimated streams are re-interleaved into a single real-valued
output of the same length as the input.

Notes
-----
* **Spatial multiplexing** doubles the spectral efficiency compared with
  SISO, but ZF equalization amplifies noise (especially when H is
  ill-conditioned).  MMSE avoids the worst noise enhancement at the
  cost of a small bias, making it the preferred linear equalizer.
* The channel assumes perfect CSI at the receiver (no channel
  estimation errors).
* Each symbol interval has an independently drawn H (block-fading with
  block length = 1 symbol).

Author: GitHub Copilot
"""

import numpy as np


# ── Shared helpers ──────────────────────────────────────────────────

def _prepare_mimo(signal):
    """Prepare signal for 2×2 MIMO: ensure even length, demux, compute power."""
    N = len(signal)
    if N % 2 != 0:
        signal = signal[:-1]
        N -= 1
    n_sym = N // 2
    x1 = signal[0::2].astype(np.complex128)
    x2 = signal[1::2].astype(np.complex128)
    signal_power = np.mean(np.abs(signal) ** 2)
    return signal, N, n_sym, x1, x2, signal_power


def _generate_symbol(x1, x2, k, n_rx, n_tx, noise_power):
    """Generate H, x_vec, y_vec for one symbol interval."""
    H = (np.random.randn(n_rx, n_tx) +
         1j * np.random.randn(n_rx, n_tx)) / np.sqrt(2)
    x_vec = np.array([x1[k], x2[k]])
    noise_std = np.sqrt(noise_power / 2)
    n_vec = noise_std * (np.random.randn(n_rx) +
                         1j * np.random.randn(n_rx))
    y_vec = H @ x_vec + n_vec
    return H, y_vec


# ── ZF equalizer ────────────────────────────────────────────────────

def mimo_2x2_channel(signal, snr_db):
    """Apply a 2×2 MIMO Rayleigh channel with ZF equalization.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real-valued BPSK symbols ±1).  Length must be even;
        if odd, the last symbol is dropped.
    snr_db : float
        Target Signal-to-Noise Ratio per receive antenna in dB.

    Returns
    -------
    equalized : numpy.ndarray
        ZF-equalized real-valued output, same length as (the possibly
        truncated) input.
    """
    n_tx, n_rx = 2, 2
    signal, N, n_sym, x1, x2, sig_pow = _prepare_mimo(signal)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_pow / snr_lin

    equalized = np.zeros(N)
    for k in range(n_sym):
        H, y_vec = _generate_symbol(x1, x2, k, n_rx, n_tx, noise_power)
        try:
            x_hat = np.linalg.solve(H, y_vec)
        except np.linalg.LinAlgError:
            x_hat = np.linalg.lstsq(H, y_vec, rcond=None)[0]
        equalized[2 * k] = np.real(x_hat[0])
        equalized[2 * k + 1] = np.real(x_hat[1])

    return equalized


# ── MMSE equalizer ──────────────────────────────────────────────────

def mimo_2x2_mmse_channel(signal, snr_db):
    """Apply a 2×2 MIMO Rayleigh channel with MMSE equalization.

    The MMSE linear filter is:

        W = (H^H H + σ² I)^{-1} H^H

    where σ² = signal_power / SNR_linear.  Compared with ZF, MMSE adds
    a noise-variance regularization term that prevents excessive noise
    amplification when H is ill-conditioned.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real-valued BPSK symbols ±1).  Length must be even;
        if odd, the last symbol is dropped.
    snr_db : float
        Target Signal-to-Noise Ratio per receive antenna in dB.

    Returns
    -------
    equalized : numpy.ndarray
        MMSE-equalized real-valued output, same length as (the possibly
        truncated) input.
    """
    n_tx, n_rx = 2, 2
    signal, N, n_sym, x1, x2, sig_pow = _prepare_mimo(signal)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_pow / snr_lin

    eye = np.eye(n_tx, dtype=np.complex128)
    equalized = np.zeros(N)

    for k in range(n_sym):
        H, y_vec = _generate_symbol(x1, x2, k, n_rx, n_tx, noise_power)

        # W_mmse = (H^H H + σ² I)^{-1} H^H
        HH = H.conj().T
        gram = HH @ H + noise_power * eye
        x_hat = np.linalg.solve(gram, HH @ y_vec)

        equalized[2 * k] = np.real(x_hat[0])
        equalized[2 * k + 1] = np.real(x_hat[1])

    return equalized
