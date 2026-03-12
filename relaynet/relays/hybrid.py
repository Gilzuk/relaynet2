"""
Hybrid SNR-Adaptive Relay

Dynamically switches between Minimal GenAI and Decode-and-Forward based
on an estimated SNR at the relay.

Strategy:
  - If estimated SNR < threshold  → use Minimal GenAI relay
  - Otherwise                     → use classical DF relay
"""

import numpy as np

from .base import Relay
from .df import DecodeAndForwardRelay
from .genai import MinimalGenAIRelay


def estimate_snr(received_signal, known_tx_power=1.0):
    """Estimate SNR from the received signal power.

    Assumes BPSK with known transmitted power ``known_tx_power``.
    The received power is P_rx = P_tx + P_noise, so
    P_noise ≈ P_rx - P_tx  (only valid when P_rx > P_tx).

    Parameters
    ----------
    received_signal : numpy.ndarray
        Received noisy signal.
    known_tx_power : float
        Known transmitted signal power (default 1.0 for normalised BPSK).

    Returns
    -------
    snr_db : float
        Estimated SNR in dB.
    """
    rx_power = np.mean(np.abs(received_signal) ** 2)
    # Noise power estimate (floor at a small positive value for stability)
    noise_power = max(rx_power - known_tx_power, 1e-10)
    snr_linear = known_tx_power / noise_power
    return 10 * np.log10(snr_linear)


class HybridRelay(Relay):
    """Hybrid SNR-adaptive relay.

    Uses a Minimal GenAI relay at low SNR and a DF relay at medium/high SNR.
    An SNR estimator based on received signal power selects the strategy.

    Parameters
    ----------
    snr_threshold : float
        SNR threshold in dB. Below this the GenAI relay is used;
        at or above it the DF relay is used. Default 5 dB.
    target_power : float
        Target transmitted power for both sub-relays.
    genai_window_size : int
        Window size for the Minimal GenAI sub-relay.
    genai_hidden_size : int
        Hidden layer size for the Minimal GenAI sub-relay.
    """

    def __init__(self, snr_threshold=5.0, target_power=1.0,
                 genai_window_size=5, genai_hidden_size=24):
        self.snr_threshold = snr_threshold
        self.target_power = target_power

        self.genai_relay = MinimalGenAIRelay(
            window_size=genai_window_size,
            hidden_size=genai_hidden_size,
            target_power=target_power,
        )
        self.df_relay = DecodeAndForwardRelay(target_power=target_power)

        self.is_trained = False

    def train(self, training_snrs=None, num_samples=25000, epochs=100, seed=None):
        """Train the internal GenAI sub-relay.

        Parameters
        ----------
        training_snrs : list of float, optional
            Defaults to [2, 4, 6] (covers the low-SNR operating region).
        num_samples : int
        epochs : int
        seed : int, optional
        """
        if training_snrs is None:
            training_snrs = [2, 4, 6]
        self.genai_relay.train(
            training_snrs=training_snrs,
            num_samples=num_samples,
            epochs=epochs,
            seed=seed,
        )
        self.is_trained = True

    def process(self, received_signal):
        """Process signal using the SNR-adaptive strategy.

        Estimates the SNR from the received signal power and routes to
        the appropriate sub-relay.
        """
        est_snr = estimate_snr(received_signal)

        if est_snr < self.snr_threshold:
            return self.genai_relay.process(received_signal)
        return self.df_relay.process(received_signal)
