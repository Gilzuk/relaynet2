"""
BPSK Modulation / Demodulation

Implements Binary Phase Shift Keying modulation, demodulation, and
Bit Error Rate calculation.
"""

import numpy as np


def bpsk_modulate(bits):
    """
    Modulate binary bits using BPSK.

    Mapping: 0 → -1,  1 → +1.

    Parameters
    ----------
    bits : array-like
        Binary input bits (0s and 1s).

    Returns
    -------
    symbols : numpy.ndarray
        BPSK modulated symbols (-1.0 and +1.0).
    """
    bits = np.asarray(bits)
    return (2 * bits - 1).astype(float)


def bpsk_demodulate(symbols, decision_threshold=0.0):
    """
    Demodulate BPSK symbols to binary bits (hard-decision).

    Parameters
    ----------
    symbols : array-like
        Received BPSK symbols (real-valued).
    decision_threshold : float, optional
        Decision threshold (default 0.0).

    Returns
    -------
    bits : numpy.ndarray
        Demodulated binary bits (0s and 1s).
    """
    symbols = np.asarray(symbols)
    return (symbols >= decision_threshold).astype(int)


def calculate_ber(transmitted_bits, received_bits):
    """
    Calculate Bit Error Rate (BER).

    Parameters
    ----------
    transmitted_bits : array-like
        Original transmitted bits.
    received_bits : array-like
        Received / decoded bits.

    Returns
    -------
    ber : float
        Bit Error Rate (0 to 1).
    num_errors : int
        Number of bit errors.
    """
    tx = np.asarray(transmitted_bits)
    rx = np.asarray(received_bits)
    if len(tx) != len(rx):
        raise ValueError("Transmitted and received bit arrays must have the same length.")
    num_errors = int(np.sum(tx != rx))
    ber = num_errors / len(tx) if len(tx) > 0 else 0.0
    return ber, num_errors
