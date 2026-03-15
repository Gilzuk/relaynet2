"""
16-QAM Modulation / Demodulation

Implements 16-point Quadrature Amplitude Modulation with Gray coding.

Each symbol carries 4 bits.  The first two bits select the in-phase (I)
amplitude and the last two bits select the quadrature (Q) amplitude,
both from the set {−3, −1, +1, +3} (before normalisation).

Gray-coded PAM-4 mapping (per axis, 2 bits → level):
    00 → +3
    01 → +1
    11 → −1
    10 → −3

Average symbol energy (before normalisation):
    E[I²] = (9 + 1 + 1 + 9) / 4 = 5
    E[|x|²] = E[I²] + E[Q²] = 10

Symbols are normalised by √10 so that E[|x|²] = 1.

Theoretical BER (AWGN, Gray-coded):
    P_b ≈ (3/8) erfc(√(2 E_b / (5 N_0)))     (approximate)
"""

import numpy as np


# ── Look-up tables ──────────────────────────────────────────────────

# Forward map:  index = b0*2 + b1  →  PAM-4 level
#   00 (0) → +3,   01 (1) → +1,   10 (2) → −3,   11 (3) → −1
_IDX_TO_LEVEL = np.array([3.0, 1.0, -3.0, -1.0])

# Ordered constellation levels for nearest-neighbour quantisation
_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])

# Inverse map:  quantised-level index (sorted order) → Gray-coded bits
#   −3 (idx 0) → (1, 0)
#   −1 (idx 1) → (1, 1)
#   +1 (idx 2) → (0, 1)
#   +3 (idx 3) → (0, 0)
_LEVEL_IDX_TO_BITS = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=int)

# Normalisation factor for unit average symbol power
_QAM16_NORM = np.sqrt(10.0)


# ── Public API ──────────────────────────────────────────────────────

def qam16_modulate(bits):
    """
    Modulate binary bits using 16-QAM with Gray coding.

    Parameters
    ----------
    bits : array-like
        Binary input bits (0s and 1s).  Length must be divisible by 4.

    Returns
    -------
    symbols : numpy.ndarray (complex128)
        16-QAM symbols with unit average power.

    Raises
    ------
    ValueError
        If the number of bits is not divisible by 4.
    """
    bits = np.asarray(bits)
    if len(bits) % 4 != 0:
        raise ValueError(
            "Number of bits must be divisible by 4 for 16-QAM modulation."
        )

    bit_groups = bits.reshape(-1, 4)

    # First 2 bits → I level,  last 2 bits → Q level
    I_idx = bit_groups[:, 0] * 2 + bit_groups[:, 1]
    Q_idx = bit_groups[:, 2] * 2 + bit_groups[:, 3]

    I = _IDX_TO_LEVEL[I_idx]
    Q = _IDX_TO_LEVEL[Q_idx]

    return (I + 1j * Q) / _QAM16_NORM


def qam16_demodulate(symbols):
    """
    Demodulate 16-QAM symbols to binary bits (hard-decision).

    Uses nearest-constellation-point detection followed by Gray decoding.

    Parameters
    ----------
    symbols : array-like
        Received (possibly noisy) complex 16-QAM symbols.

    Returns
    -------
    bits : numpy.ndarray
        Demodulated binary bits (0s and 1s).  Length = 4 × len(symbols).
    """
    symbols = np.asarray(symbols)

    # Un-normalise
    I = symbols.real * _QAM16_NORM
    Q = symbols.imag * _QAM16_NORM

    # Quantise to nearest of {−3, −1, +1, +3}
    I_idx = np.argmin(np.abs(I[:, None] - _LEVELS[None, :]), axis=1)
    Q_idx = np.argmin(np.abs(Q[:, None] - _LEVELS[None, :]), axis=1)

    # Map quantised level indices to Gray-coded bit pairs
    I_bits = _LEVEL_IDX_TO_BITS[I_idx]   # (N, 2)
    Q_bits = _LEVEL_IDX_TO_BITS[Q_idx]   # (N, 2)

    # Combine:  [I_b0, I_b1, Q_b0, Q_b1] per symbol
    result = np.empty((len(symbols), 4), dtype=int)
    result[:, :2] = I_bits
    result[:, 2:] = Q_bits

    return result.flatten()
