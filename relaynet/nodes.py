"""
Communication Node classes for relaynet.

Supports BPSK (default), QPSK, and 16-QAM modulation schemes.
"""

import numpy as np

from relaynet.modulation import get_modulation_functions


class Source:
    """Source node that generates and modulates binary data.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    modulation : str, optional
        Modulation scheme: ``'bpsk'``, ``'qpsk'``, or ``'qam16'``.
        Default ``'bpsk'``.
    """

    def __init__(self, seed=None, modulation="bpsk"):
        self.seed = seed
        self.modulation = modulation
        if seed is not None:
            np.random.seed(seed)

    def generate_bits(self, num_bits):
        """Generate random binary bits."""
        return np.random.randint(0, 2, num_bits)

    def transmit(self, num_bits):
        """Generate bits and modulate them.

        The number of bits is truncated to the nearest multiple of
        *bits_per_symbol* so that modulation always succeeds.

        Returns
        -------
        bits : numpy.ndarray
        symbols : numpy.ndarray
        """
        modulate, _, bps = get_modulation_functions(self.modulation)
        usable = (num_bits // bps) * bps
        bits = self.generate_bits(usable)
        symbols = modulate(bits)
        return bits, symbols


class Destination:
    """Destination node that demodulates received symbols.

    Parameters
    ----------
    modulation : str, optional
        Modulation scheme: ``'bpsk'``, ``'qpsk'``, or ``'qam16'``.
        Default ``'bpsk'``.
    """

    def __init__(self, modulation="bpsk"):
        self.modulation = modulation

    def receive(self, received_signal):
        """Hard-decision demodulate the received signal."""
        _, demodulate, _ = get_modulation_functions(self.modulation)
        return demodulate(received_signal)
