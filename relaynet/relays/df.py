"""Decode-and-Forward relay."""

import numpy as np

from .base import Relay
from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate
from relaynet.utils.torch_compat import can_use_gpu, get_preferred_device, get_torch_module, to_numpy


class DecodeAndForwardRelay(Relay):
    """Decode-and-Forward (DF) relay.

    Demodulates the received signal to recover bits, then re-modulates and
    forwards a clean signal to the destination.
    """

    def __init__(self, target_power=1.0, prefer_gpu=True):
        self.target_power = target_power
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)

    def process(self, received_signal):
        if can_use_gpu(self.device):
            torch = get_torch_module()
            rx_t = torch.as_tensor(received_signal, dtype=torch.float32, device=self.device)
            clean_symbols = (rx_t >= 0).to(dtype=torch.float32) * 2.0 - 1.0
            current_power = torch.mean(torch.abs(clean_symbols) ** 2)
            if float(current_power.item()) > 0:
                clean_symbols = clean_symbols * torch.sqrt(
                    torch.tensor(self.target_power, dtype=torch.float32, device=self.device) / current_power
                )
            return to_numpy(clean_symbols, dtype=float)

        decoded_bits = bpsk_demodulate(received_signal)
        clean_symbols = bpsk_modulate(decoded_bits)
        current_power = np.mean(np.abs(clean_symbols) ** 2)
        if current_power > 0:
            clean_symbols = clean_symbols * np.sqrt(self.target_power / current_power)
        return clean_symbols
