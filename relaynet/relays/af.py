"""Amplify-and-Forward relay."""

import numpy as np

from .base import Relay


class AmplifyAndForwardRelay(Relay):
    """Amplify-and-Forward (AF) relay.

    Amplifies the received noisy signal and forwards it to the destination
    with power normalisation.
    """

    def __init__(self, target_power=1.0):
        self.target_power = target_power

    def process(self, received_signal):
        current_power = np.mean(np.abs(received_signal) ** 2)
        if current_power > 0:
            gain = np.sqrt(self.target_power / current_power)
            return gain * received_signal
        return received_signal
