"""
Communication Node classes for relaynet.
"""

import numpy as np

from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate


class Source:
    """Source node that generates and modulates binary data."""

    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_bits(self, num_bits):
        """Generate random binary bits."""
        return np.random.randint(0, 2, num_bits)

    def transmit(self, num_bits):
        """Generate bits and BPSK-modulate them.

        Returns
        -------
        bits : numpy.ndarray
        symbols : numpy.ndarray
        """
        bits = self.generate_bits(num_bits)
        symbols = bpsk_modulate(bits)
        return bits, symbols


class Destination:
    """Destination node that demodulates received symbols."""

    def receive(self, received_signal):
        """Hard-decision demodulate the received signal."""
        return bpsk_demodulate(received_signal)
