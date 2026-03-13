"""Amplify-and-Forward relay."""

import numpy as np

from .base import Relay
from relaynet.utils.torch_compat import can_use_gpu, get_preferred_device, get_torch_module, to_numpy


class AmplifyAndForwardRelay(Relay):
    """Amplify-and-Forward (AF) relay.

    Amplifies the received noisy signal and forwards it to the destination
    with power normalisation.
    """

    def __init__(self, target_power=1.0, prefer_gpu=True):
        self.target_power = target_power
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)

    def process(self, received_signal):
        if can_use_gpu(self.device):
            torch = get_torch_module()
            signal_t = torch.as_tensor(received_signal, dtype=torch.float32, device=self.device)
            current_power = torch.mean(torch.abs(signal_t) ** 2)
            if float(current_power.item()) > 0:
                gain = torch.sqrt(torch.tensor(self.target_power, dtype=torch.float32, device=self.device) / current_power)
                return to_numpy(gain * signal_t, dtype=float)
            return to_numpy(signal_t, dtype=float)

        current_power = np.mean(np.abs(received_signal) ** 2)
        if current_power > 0:
            gain = np.sqrt(self.target_power / current_power)
            return gain * received_signal
        return received_signal
