"""
16-PSK Modulation / Demodulation

Implements 16-point Phase Shift Keying with Gray coding.

Each symbol carries 4 bits.  The 16 constellation points are
uniformly spaced on the unit circle at angles

    θ_k = 2π · k / 16,   k = 0, 1, …, 15

with Gray-coded symbol indexing so that adjacent constellation
points differ in exactly one bit (minimising BER).

All symbols lie on the unit circle, so E[|x|²] = 1. The maximum
per-axis extent is cos(π/16) ≈ 0.9808 → clip_range = 1.0 (radius).

Theoretical BER (AWGN, Gray-coded, high SNR):
    P_b ≈ (1/4) erfc(√(E_b / N_0) sin(π/16))
"""

import numpy as np


# ── Gray code mapping ──────────────────────────────────────────────

# Standard 4-bit Gray code order: adjacent indices differ by 1 bit.
#   decimal  →  Gray
#     0      →  0000
#     1      →  0001
#     2      →  0011
#     3      →  0010
#     4      →  0110
#     5      →  0111
#     6      →  0101
#     7      →  0100
#     8      →  1100
#     9      →  1101
#    10      →  1111
#    11      →  1110
#    12      →  1010
#    13      →  1011
#    14      →  1001
#    15      →  1000

# Map 4-bit decimal value (0-15) → Gray-coded constellation index
_GRAY_CODE = np.array([0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8])

# Inverse: constellation index → 4-bit decimal value
_GRAY_INV = np.empty(16, dtype=int)
_GRAY_INV[_GRAY_CODE] = np.arange(16)

# Constellation angles and symbols (16 points on unit circle)
PSK16_ANGLES = 2.0 * np.pi * np.arange(16) / 16.0
PSK16_SYMBOLS = np.exp(1j * PSK16_ANGLES)

# Pre-computed 4-bit patterns for each decimal 0-15
_BITS_TABLE = np.array([[(d >> 3) & 1, (d >> 2) & 1, (d >> 1) & 1, d & 1]
                        for d in range(16)])


# ── Public API ──────────────────────────────────────────────────────

def psk16_modulate(bits):
    """
    Modulate binary bits using 16-PSK with Gray coding.

    Parameters
    ----------
    bits : array-like
        Binary input bits (0s and 1s).  Length must be divisible by 4.

    Returns
    -------
    symbols : numpy.ndarray (complex128)
        16-PSK modulated symbols on the unit circle.

    Raises
    ------
    ValueError
        If the number of bits is not divisible by 4.
    """
    bits = np.asarray(bits)
    if len(bits) % 4 != 0:
        raise ValueError("Number of bits must be divisible by 4 for 16-PSK.")

    groups = bits.reshape(-1, 4)
    # Convert 4-bit groups to decimal (0-15)
    decimal = groups[:, 0] * 8 + groups[:, 1] * 4 + groups[:, 2] * 2 + groups[:, 3]
    # Gray-code mapping: decimal → constellation index
    indices = _GRAY_CODE[decimal]
    return PSK16_SYMBOLS[indices]


def psk16_demodulate(symbols):
    """
    Demodulate 16-PSK symbols to binary bits (hard-decision).

    Decision rule: nearest constellation point by angle.

    Parameters
    ----------
    symbols : array-like
        Received (possibly noisy) complex 16-PSK symbols.

    Returns
    -------
    bits : numpy.ndarray
        Demodulated binary bits (0s and 1s).  Length = 4 × len(symbols).
    """
    symbols = np.asarray(symbols)

    # Compute angles in [0, 2π)
    angles = np.angle(symbols)
    angles = np.where(angles < 0, angles + 2.0 * np.pi, angles)

    # Quantise to nearest of the 16 uniformly-spaced angles
    indices = np.round(angles * 16.0 / (2.0 * np.pi)).astype(int) % 16

    # Inverse Gray map: constellation index → decimal
    decimal = _GRAY_INV[indices]

    # Decimal → 4-bit pattern
    bits_out = _BITS_TABLE[decimal]  # (N, 4)

    return bits_out.flatten()
