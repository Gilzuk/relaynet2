"""
Hybrid SNR-Adaptive Relay

Dynamically switches between Minimal MLP and Decode-and-Forward based
on an estimated SNR at the relay.

Strategy:
  - If estimated SNR < threshold  → use Minimal MLP relay
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

    Uses a Minimal MLP relay at low SNR and a DF relay at medium/high SNR.
    An SNR estimator based on received signal power selects the strategy.

    Parameters
    ----------
    snr_threshold : float
        SNR threshold in dB. Below this the MLP relay is used;
        at or above it the DF relay is used. Default 5 dB.
    target_power : float
        Target transmitted power for both sub-relays.
    mlp_window_size : int
        Window size for the Minimal MLP sub-relay.
    mlp_hidden_size : int
        Hidden layer size for the Minimal MLP sub-relay.
    """

    def __init__(self, snr_threshold=5.0, target_power=1.0,
                 mlp_window_size=5, mlp_hidden_size=24, prefer_gpu=True,
                 output_activation="tanh", clip_range=None,
                 # Legacy aliases
                 genai_window_size=None, genai_hidden_size=None):
        self.snr_threshold = snr_threshold
        self.target_power = target_power

        # Support legacy parameter names
        ws = genai_window_size if genai_window_size is not None else mlp_window_size
        hs = genai_hidden_size if genai_hidden_size is not None else mlp_hidden_size

        self.mlp_relay = MinimalGenAIRelay(
            window_size=ws,
            hidden_size=hs,
            target_power=target_power,
            prefer_gpu=prefer_gpu,
            output_activation=output_activation,
            clip_range=clip_range,
        )
        self.df_relay = DecodeAndForwardRelay(target_power=target_power, prefer_gpu=prefer_gpu)
        self.device = getattr(self.mlp_relay, "device", None)

        self.is_trained = False

    @property
    def genai_relay(self):
        """Legacy alias for mlp_relay."""
        return self.mlp_relay

    @property
    def num_params(self):
        return self.mlp_relay.num_params

    def train(self, training_snrs=None, num_samples=25000, epochs=100, seed=None,
              epoch_callback=None, training_modulation="bpsk"):
        """Train the internal MLP sub-relay.

        Parameters
        ----------
        training_snrs : list of float, optional
            Defaults to [2, 4, 6] (covers the low-SNR operating region).
        num_samples : int
        epochs : int
        seed : int, optional
        epoch_callback : callable, optional
            Called as ``epoch_callback(epoch, epochs)`` after each epoch.
        training_modulation : str, optional
            ``'bpsk'`` (default) or ``'qam16'``.
        """
        if training_snrs is None:
            training_snrs = [2, 4, 6]
        self.mlp_relay.train(
            training_snrs=training_snrs,
            num_samples=num_samples,
            epochs=epochs,
            seed=seed,
            epoch_callback=epoch_callback,
            training_modulation=training_modulation,
        )
        self.is_trained = True

    def process(self, received_signal):
        """Process signal using the SNR-adaptive strategy.

        Estimates the SNR from the received signal power and routes to
        the appropriate sub-relay.
        """
        est_snr = estimate_snr(received_signal)

        if est_snr < self.snr_threshold:
            return self.mlp_relay.process(received_signal)
        return self.df_relay.process(received_signal)

    # ------------------------------------------------------------------
    # Weight persistence (delegates to MLP sub-relay)
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Save the internal MLP sub-relay weights to *path*."""
        self.mlp_relay.save_weights(path)

    def load_weights(self, path):
        """Load weights into the MLP sub-relay and mark as trained."""
        self.mlp_relay.load_weights(path)
        self.is_trained = True
