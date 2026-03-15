"""
QPSK Modulation / Demodulation

Implements Quadrature Phase Shift Keying with Gray coding.

Constellation (unit average power):
    bits  →  symbol
    00    →  (+1+j) / √2     (45°)
    01    →  (+1−j) / √2     (315°)
    10    →  (−1+j) / √2     (135°)
    11    →  (−1−j) / √2     (225°)

First bit  → I component:  0 → +1,  1 → −1
Second bit → Q component:  0 → +1,  1 → −1

Adjacent symbols (in Gray code sense) differ in exactly one bit,
minimising BER at moderate-to-high SNR.

Theoretical BER (AWGN):  P_b = Q(√(2 E_b / N_0))  — identical to BPSK per bit.
"""

import numpy as np


def qpsk_modulate(bits):
    """
    Modulate binary bits using QPSK with Gray coding.

    Parameters
    ----------
    bits : array-like
        Binary input bits (0s and 1s).  Length must be even.

    Returns
    -------
    symbols : numpy.ndarray (complex128)
        QPSK modulated symbols with unit average power.

    Raises
    ------
    ValueError
        If the number of bits is odd.
    """
    bits = np.asarray(bits)
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK modulation.")

    bit_pairs = bits.reshape(-1, 2)

    # Gray mapping:  bit 0 → +1,  bit 1 → −1
    I = (1 - 2 * bit_pairs[:, 0]).astype(float)
    Q = (1 - 2 * bit_pairs[:, 1]).astype(float)

    # Normalise to unit average power:  E[|s|²] = 1
    return (I + 1j * Q) / np.sqrt(2)


def qpsk_demodulate(symbols):
    """
    Demodulate QPSK symbols to binary bits (hard-decision).

    Decision rule (Gray decoded):
        I ≥ 0 → first bit = 0,   I < 0 → first bit = 1
        Q ≥ 0 → second bit = 0,  Q < 0 → second bit = 1

    Parameters
    ----------
    symbols : array-like
        Received (possibly noisy) complex QPSK symbols.

    Returns
    -------
    bits : numpy.ndarray
        Demodulated binary bits (0s and 1s).  Length = 2 × len(symbols).
    """
    symbols = np.asarray(symbols)

    I_bits = (symbols.real < 0).astype(int)
    Q_bits = (symbols.imag < 0).astype(int)

    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = I_bits
    bits[1::2] = Q_bits

    return bits
