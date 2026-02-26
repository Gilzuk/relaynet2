"""
Fading Channel Implementations

Implements Rayleigh and Rician fading channels for digital
communication simulation.
"""

import numpy as np


def rayleigh_fading_channel(signal, snr_db, return_channel=False):
    """
    Apply Rayleigh fading channel: y = h * x + n, where h ~ CN(0, 1).

    Perfect channel state information (CSI) equalization is applied at
    the receiver, i.e. the output is y / h.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real-valued BPSK symbols).
    snr_db : float
        Target Signal-to-Noise Ratio in dB (post-equalization).
    return_channel : bool, optional
        If True, also return the channel coefficients *h*.

    Returns
    -------
    equalized : numpy.ndarray
        Equalized received signal.
    h : numpy.ndarray, optional
        Channel coefficients (only when *return_channel* is True).
    """
    n = len(signal)

    # Complex Rayleigh fading coefficients: h ~ CN(0, 1)
    h = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(n) + 1j * np.random.randn(n))

    received = h * signal + noise

    # Perfect CSI equalization: divide by h
    equalized = np.real(received / h)

    if return_channel:
        return equalized, h
    return equalized


def rician_fading_channel(signal, snr_db, k_factor=1.0, return_channel=False):
    """
    Apply Rician fading channel with K-factor.

    The channel coefficient is h = h_los + h_scatter where:
      h_los   is the line-of-sight (deterministic) component,
      h_scatter ~ CN(0, sigma^2) is the scattered component.

    The K-factor is defined as K = |h_los|^2 / (2 * sigma^2).

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real-valued BPSK symbols).
    snr_db : float
        Target Signal-to-Noise Ratio in dB (post-equalization).
    k_factor : float, optional
        Rician K-factor (ratio of LOS power to scatter power). Default 1.0.
    return_channel : bool, optional
        If True, also return the channel coefficients *h*.

    Returns
    -------
    equalized : numpy.ndarray
        Equalized received signal.
    h : numpy.ndarray, optional
        Channel coefficients (only when *return_channel* is True).
    """
    n = len(signal)

    # Decompose: total power = 1
    # LOS component magnitude
    los_amplitude = np.sqrt(k_factor / (k_factor + 1))
    # Scatter std (per real/imag component)
    scatter_std = np.sqrt(1.0 / (2 * (k_factor + 1)))

    h_los = los_amplitude + 0j  # deterministic, constant phase = 0
    h_scatter = scatter_std * (np.random.randn(n) + 1j * np.random.randn(n))
    h = h_los + h_scatter

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(n) + 1j * np.random.randn(n))

    received = h * signal + noise

    # Perfect CSI equalization
    equalized = np.real(received / h)

    if return_channel:
        return equalized, h
    return equalized
