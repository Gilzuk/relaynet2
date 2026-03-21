"""End-to-End Autoencoder relay.

Wraps a pre-trained E2E transmitter/receiver pair as a :class:`Relay`
so it can be plugged into the ``relaynet`` simulation pipeline alongside
AF, DF, GenAI, VAE, CGAN, Transformer, and Mamba relays.

The relay processes received signals by:
1. Quantising each I/Q sample to the nearest learned constellation point
   (using the trained transmitter's codebook).
2. Power-normalising the output.

For BPSK (real-valued) signals the I and Q components are processed
independently — consistent with the convention used by all other
AI-based relays in the project.
"""

import numpy as np

from .base import Relay
from relaynet.utils.torch_compat import (
    get_torch_module,
    get_preferred_device,
    can_use_gpu,
    to_numpy,
    save_state,
    load_state,
)


class E2ERelay(Relay):
    """End-to-End Autoencoder relay.

    After training the autoencoder (checkpoint 24) the relay uses the
    learned constellation (transmitter codebook) as a nearest-neighbour
    lookup for re-modulating received symbols — analogous to DF with a
    learned constellation instead of a standard one.

    Parameters
    ----------
    M : int
        Constellation size (must match the trained model).
    hidden_dim : int
        Hidden-layer width of the transmitter / receiver.
    target_power : float
        Desired average output power.
    prefer_gpu : bool
        Try to use CUDA if available.
    """

    def __init__(self, M=16, hidden_dim=64, target_power=1.0, prefer_gpu=True):
        self.M = M
        self.hidden_dim = hidden_dim
        self.target_power = target_power
        self.device = get_preferred_device(prefer_gpu=prefer_gpu)
        self._use_torch = can_use_gpu(self.device) or get_torch_module() is not None

        # Lazily built when weights are loaded
        self._transmitter = None
        self._codebook = None  # (M, 2) numpy array of constellation points
        self.is_trained = False

    # ------------------------------------------------------------------
    # Model construction helpers
    # ------------------------------------------------------------------

    def _build_models(self):
        """Construct Torch model instances (mirrors checkpoint_24)."""
        torch = get_torch_module()
        if torch is None:
            raise RuntimeError("PyTorch is required for the E2E relay.")

        # Import from the checkpoint to reuse the exact architecture
        from checkpoints.checkpoint_24_e2e_transmitter import E2ETransmitter

        device = self.device if self.device is not None else "cpu"
        self._transmitter = E2ETransmitter(
            M=self.M, hidden_dim=self.hidden_dim
        ).to(device)

    def _refresh_codebook(self):
        """Extract the (M, 2) constellation from the trained transmitter."""
        torch = get_torch_module()
        device = next(self._transmitter.parameters()).device

        self._transmitter.eval()
        with torch.no_grad():
            indices = torch.arange(self.M, device=device)
            points = self._transmitter(indices)
        self._codebook = to_numpy(points, dtype=float)
        self._transmitter.train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_snrs=None, num_samples=25_000, epochs=10_000,
              seed=42, epoch_callback=None, curriculum=True,
              label_smoothing=0.1, **kwargs):
        """Train the E2E autoencoder from scratch.

        Parameters
        ----------
        training_snrs : ignored (SNR is sampled uniformly during training).
        num_samples : ignored (batch_size x epochs determines total data).
        epochs : int
        seed : int
        epoch_callback : callable, optional
        curriculum : bool
            Enable curriculum SNR schedule (high SNR first).
        label_smoothing : float
            Label smoothing factor for CrossEntropyLoss.
        """
        from checkpoints.checkpoint_24_e2e_transmitter import (
            E2ETransmitter, DifferentiableRayleighChannel, E2EReceiver,
            train_e2e,
        )

        torch = get_torch_module()
        device = self.device if self.device is not None else torch.device("cpu")

        self._transmitter = E2ETransmitter(M=self.M, hidden_dim=self.hidden_dim).to(device)
        channel = DifferentiableRayleighChannel(perfect_csi=True).to(device)
        receiver = E2EReceiver(M=self.M, hidden_dim=self.hidden_dim).to(device)

        train_e2e(
            self._transmitter, channel, receiver,
            epochs=epochs, seed=seed, device=device,
            curriculum=curriculum,
            label_smoothing=label_smoothing,
            verbose=False,
        )

        self._refresh_codebook()
        self.is_trained = True

    # ------------------------------------------------------------------
    # Relay processing (compatible with simulation pipeline)
    # ------------------------------------------------------------------

    def process(self, received_signal):
        """Process received signal via nearest-constellation-point mapping.

        For real-valued inputs (BPSK), the signal is treated as a
        1-D projection and mapped to the nearest constellation I-component.

        Parameters
        ----------
        received_signal : numpy.ndarray

        Returns
        -------
        forwarded : numpy.ndarray
        """
        if not self.is_trained or self._codebook is None:
            # Untrained fallback: simple amplification (like AF)
            current_power = np.mean(np.abs(received_signal) ** 2)
            if current_power > 0:
                return received_signal * np.sqrt(self.target_power / current_power)
            return received_signal

        codebook = self._codebook  # shape (M, 2)

        if np.iscomplexobj(received_signal):
            # Complex signal: find nearest constellation point for each sample
            rx_points = np.column_stack([received_signal.real, received_signal.imag])
            dists = np.linalg.norm(
                rx_points[:, None, :] - codebook[None, :, :], axis=2
            )  # (N, M)
            nearest_idx = np.argmin(dists, axis=1)
            clean_points = codebook[nearest_idx]
            forwarded = clean_points[:, 0] + 1j * clean_points[:, 1]
        else:
            # Real signal (BPSK): nearest I-component
            codebook_i = codebook[:, 0]
            dists = np.abs(received_signal[:, None] - codebook_i[None, :])
            nearest_idx = np.argmin(dists, axis=1)
            forwarded = codebook_i[nearest_idx]

        # Power normalisation
        pwr = np.mean(np.abs(forwarded) ** 2)
        if pwr > 0:
            forwarded = forwarded * np.sqrt(self.target_power / pwr)

        return forwarded

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    @property
    def num_params(self):
        """Total number of trainable parameters in the transmitter."""
        if self._transmitter is not None:
            return sum(p.numel() for p in self._transmitter.parameters())
        # Estimate from architecture
        h = self.hidden_dim
        M = self.M
        return M * h + h * h + h + h * h + h + h * 2 + 2  # embedding + 3-layer MLP

    def save_weights(self, path):
        """Save trained weights to *path*."""
        state = {
            "type": "E2ERelay",
            "config": {"M": self.M, "hidden_dim": self.hidden_dim},
        }
        if self._transmitter is not None:
            state["transmitter_state_dict"] = self._transmitter.state_dict()
        if self._codebook is not None:
            state["codebook"] = self._codebook
        save_state(state, path)

    def load_weights(self, path):
        """Load trained weights from *path*."""
        state = load_state(path)

        config = state.get("config", {})
        self.M = config.get("M", self.M)
        self.hidden_dim = config.get("hidden_dim", self.hidden_dim)

        if "transmitter_state_dict" in state:
            self._build_models()
            self._transmitter.load_state_dict(state["transmitter_state_dict"])
            self._refresh_codebook()
        elif "codebook" in state:
            self._codebook = np.asarray(state["codebook"])

        self.is_trained = True
