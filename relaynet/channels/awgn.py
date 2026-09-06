"""
AWGN Channel Implementation

Implements an Additive White Gaussian Noise (AWGN) channel
for digital communication simulation.
"""

import numpy as np


def awgn_channel(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (real or complex)
    snr_db : float
        Target Signal-to-Noise Ratio in decibels

    Returns
    -------
    noisy_signal : numpy.ndarray
        Signal with added AWGN
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Noise variance is N0/2 per real dimension, so ``snr_db`` is Eb/N0 and
    # BPSK obeys the textbook Pb = Q(sqrt(2*Eb/N0)). The real-valued branch
    # previously used the whole of N0 in its single dimension, which is a
    # 3 dB pessimistic axis (Pb = Q(sqrt(Eb/N0))) and put this channel on a
    # different footing from the Rayleigh channel in fading.py, even though
    # the two are compared against each other in the calibration table.
    if np.iscomplexobj(signal):
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
    else:
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * np.random.randn(len(signal))

    return signal + noise


def calculate_snr(signal, noisy_signal):
    """
    Calculate the actual SNR between a clean signal and a noisy signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Original clean signal
    noisy_signal : numpy.ndarray
        Signal with noise added

    Returns
    -------
    snr_db : float
        Measured SNR in decibels
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise = noisy_signal - signal
    noise_power = np.mean(np.abs(noise) ** 2)

    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    return np.inf
