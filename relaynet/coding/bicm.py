"""Bit-interleaved coded modulation (BICM) helpers for rate adaptation.

Rate adaptation by puncturing deletes individual *coded bits*, which does
not compose with a metric defined on whole symbols: the joint PAM-4
branch metric of :mod:`relaynet.coding.convolutional_qam16` assumes both
bits of a trellis step survive to form one axis level, and puncturing
breaks exactly that. BICM sidesteps it -- map the (possibly punctured)
coded bit stream onto symbols independently of the trellis, then demap
back to *per-bit* soft values at the receiver, so puncturing, modulation
and decoding are three separable stages.

Soft values follow the convention the Viterbi/BCJR decoders already use:
positive means bit 0, negative means bit 1, magnitude means confidence.
Because those decoders minimize a squared distance to expected symbols
+-1, and

    sum_i (y_i - e_i)^2 = sum_i y_i^2 - 2 sum_i y_i e_i + sum_i e_i^2,

with the first and third terms constant across branches, the metric is
equivalent to a correlation. Any positive monotone scaling of the soft
value therefore leaves the Viterbi decision unchanged, which is why the
max-log LLRs below can be fed in directly without calibration. (BCJR is
*not* scale-invariant in this way -- see the note in
:mod:`relaynet.coding.bcjr` -- so this shortcut is for the hard decoder.)
"""

import numpy as np

from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.modulation.qam import qam16_modulate

BITS_PER_SYMBOL = {"qpsk": 2, "qam16": 4}


def _constellation(modulation):
    """All constellation points with their bit labels, in index order."""
    k = BITS_PER_SYMBOL[modulation]
    bits = np.array([[(i >> (k - 1 - j)) & 1 for j in range(k)]
                     for i in range(2 ** k)], dtype=int)
    mod = qpsk_modulate if modulation == "qpsk" else qam16_modulate
    pts = np.array([mod(b)[0] for b in bits])
    return bits, pts


def modulate_bits(bits, modulation):
    """Map a bit stream to symbols, zero-padding the final partial symbol.

    Returns (symbols, n_pad) so the receiver can discard the padding.
    """
    k = BITS_PER_SYMBOL[modulation]
    bits = np.asarray(bits, dtype=int)
    n_pad = (-len(bits)) % k
    if n_pad:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    mod = qpsk_modulate if modulation == "qpsk" else qam16_modulate
    return mod(bits), n_pad


def soft_demap(symbols, modulation, n_bits, noise_var=1.0):
    """Per-bit max-log soft values from received symbols.

    Parameters
    ----------
    symbols : ndarray, complex
        Received (equalized) symbols.
    modulation : {"qpsk", "qam16"}
    n_bits : int
        Number of real coded bits to return; trailing padding is dropped.
    noise_var : float
        Scales the output. Irrelevant to Viterbi (see the module docstring)
        but kept so callers can pass a calibrated value if they feed the
        result to a metric that is not scale-invariant.

    Returns
    -------
    soft : ndarray of float, length n_bits
        Positive => bit 0, negative => bit 1.
    """
    bits, pts = _constellation(modulation)
    k = BITS_PER_SYMBOL[modulation]
    y = np.asarray(symbols).reshape(-1, 1)
    d2 = np.abs(y - pts.reshape(1, -1)) ** 2          # (n_sym, 2^k)

    out = np.empty((len(y), k))
    for j in range(k):
        zero = d2[:, bits[:, j] == 0].min(axis=1)
        one = d2[:, bits[:, j] == 1].min(axis=1)
        out[:, j] = (one - zero) / (2.0 * noise_var)   # +ve => bit 0
    return out.reshape(-1)[:n_bits]


def hard_remap(soft_bits, modulation):
    """Hard-decide a soft bit stream and re-modulate it.

    This is the denoise-only relay operation at the bit level: it cleans
    each bit to the nearest decision but never invokes the code, so the
    code's redundancy reaches the destination intact.
    """
    hard = (np.asarray(soft_bits) < 0).astype(int)
    return modulate_bits(hard, modulation)
