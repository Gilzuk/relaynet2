"""Modulation implementations for relaynet."""

from .bpsk import bpsk_modulate, bpsk_demodulate, calculate_ber
from .qpsk import qpsk_modulate, qpsk_demodulate
from .qam import qam16_modulate, qam16_demodulate

__all__ = [
    "bpsk_modulate",
    "bpsk_demodulate",
    "calculate_ber",
    "qpsk_modulate",
    "qpsk_demodulate",
    "qam16_modulate",
    "qam16_demodulate",
    "get_modulation_functions",
]


def get_modulation_functions(modulation):
    """Return ``(modulate_fn, demodulate_fn, bits_per_symbol)`` for a scheme.

    Parameters
    ----------
    modulation : str
        One of ``'bpsk'``, ``'qpsk'``, ``'qam16'``.

    Returns
    -------
    modulate : callable
        ``bits → symbols``
    demodulate : callable
        ``symbols → bits``
    bits_per_symbol : int
        Number of bits encoded in each symbol.
    """
    if modulation == "bpsk":
        return bpsk_modulate, bpsk_demodulate, 1
    if modulation == "qpsk":
        return qpsk_modulate, qpsk_demodulate, 2
    if modulation == "qam16":
        return qam16_modulate, qam16_demodulate, 4
    raise ValueError(f"Unknown modulation scheme: {modulation!r}")
