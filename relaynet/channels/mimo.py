"""
MIMO Channel Implementation (2×2 Spatial Multiplexing)

Implements a 2×2 MIMO Rayleigh flat-fading channel with Zero-Forcing
(ZF) equalization at the receiver.

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

Zero-Forcing equalization inverts the channel:

    x̂ = H⁻¹ y = x + H⁻¹ n

The two estimated streams are re-interleaved into a single real-valued
output of the same length as the input.

Notes
-----
* **Spatial multiplexing** doubles the spectral efficiency compared with
  SISO, but ZF equalization amplifies noise (especially when H is
  ill-conditioned).  This makes relay processing more important —
  a good relay can partially compensate for the ZF noise enhancement.
* The channel assumes perfect CSI at the receiver (no channel
  estimation errors).
* Each symbol interval has an independently drawn H (block-fading with
  block length = 1 symbol).

Author: GitHub Copilot
"""

import numpy as np


def mimo_2x2_channel(signal, snr_db):
    """Apply a 2×2 MIMO Rayleigh channel with ZF equalization.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real-valued BPSK symbols ±1).  Length must be even;
        if odd, the last symbol is dropped and a warning is printed.
    snr_db : float
        Target Signal-to-Noise Ratio per receive antenna in dB.

    Returns
    -------
    equalized : numpy.ndarray
        ZF-equalized real-valued output, same length as (the possibly
        truncated) input.
    """
    n_tx = 2
    n_rx = 2

    # Ensure even length
    N = len(signal)
    if N % 2 != 0:
        signal = signal[:-1]
        N -= 1

    n_sym = N // n_tx  # number of symbol intervals

    # Demux into two spatial streams
    x1 = signal[0::2].astype(np.complex128)  # stream 1
    x2 = signal[1::2].astype(np.complex128)  # stream 2

    # Noise power per receive antenna
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    equalized = np.zeros(N)

    for k in range(n_sym):
        # Independent 2×2 complex Rayleigh channel
        H = (np.random.randn(n_rx, n_tx) +
             1j * np.random.randn(n_rx, n_tx)) / np.sqrt(2)

        x_vec = np.array([x1[k], x2[k]])

        # Complex AWGN at each receive antenna
        noise_std = np.sqrt(noise_power / 2)
        n_vec = noise_std * (np.random.randn(n_rx) +
                             1j * np.random.randn(n_rx))

        y_vec = H @ x_vec + n_vec

        # ZF equalization: x_hat = H^{-1} y
        try:
            x_hat = np.linalg.solve(H, y_vec)
        except np.linalg.LinAlgError:
            # Singular channel (extremely rare) — fall back to
            # pseudo-inverse
            x_hat = np.linalg.lstsq(H, y_vec, rcond=None)[0]

        equalized[2 * k] = np.real(x_hat[0])
        equalized[2 * k + 1] = np.real(x_hat[1])

    return equalized
