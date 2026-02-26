"""Decode-and-Forward relay."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate


class DecodeAndForwardRelay(Relay):
    """Decode-and-Forward (DF) relay.

    Demodulates the received signal to recover bits, then re-modulates and
    forwards a clean signal to the destination.
    """

    def __init__(self, target_power=1.0):
        self.target_power = target_power

    def process(self, received_signal):
        decoded_bits = bpsk_demodulate(received_signal)
        clean_symbols = bpsk_modulate(decoded_bits)
        current_power = np.mean(np.abs(clean_symbols) ** 2)
        if current_power > 0:
            clean_symbols = clean_symbols * np.sqrt(self.target_power / current_power)
        return clean_symbols
