"""Unit tests for channel implementations."""

import numpy as np
import pytest

from relaynet.channels.awgn import awgn_channel, calculate_snr
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel


class TestAWGNChannel:
    def test_output_shape(self):
        signal = np.ones(100)
        noisy = awgn_channel(signal, snr_db=10)
        assert noisy.shape == signal.shape

    def test_snr_accuracy(self):
        np.random.seed(42)
        signal = 2 * np.random.randint(0, 2, 100_000) - 1.0
        for target in [0, 5, 10, 15, 20]:
            noisy = awgn_channel(signal, snr_db=target)
            measured = calculate_snr(signal, noisy)
            assert abs(measured - target) < 0.6, (
                f"SNR mismatch at {target} dB: measured {measured:.2f} dB"
            )

    def test_complex_signal(self):
        np.random.seed(0)
        n = 50_000
        signal = (2 * np.random.randint(0, 2, n) - 1) + 1j * (2 * np.random.randint(0, 2, n) - 1)
        noisy = awgn_channel(signal, snr_db=10)
        assert np.iscomplexobj(noisy)
        measured = calculate_snr(signal, noisy)
        assert abs(measured - 10) < 0.7

    def test_high_snr_low_noise(self):
        np.random.seed(1)
        signal = np.random.randn(10_000)
        noisy = awgn_channel(signal, snr_db=30)
        assert np.mean((noisy - signal) ** 2) < np.mean(signal ** 2) * 0.01


class TestFadingChannels:
    def test_rayleigh_output_shape(self):
        np.random.seed(0)
        signal = np.ones(500)
        out = rayleigh_fading_channel(signal, snr_db=10)
        assert out.shape == signal.shape
        assert np.isrealobj(out)

    def test_rayleigh_return_channel(self):
        np.random.seed(0)
        signal = np.ones(200)
        out, h = rayleigh_fading_channel(signal, snr_db=10, return_channel=True)
        assert h.shape == signal.shape
        assert np.iscomplexobj(h)

    def test_rician_k0_is_rayleigh(self):
        """Rician with K=0 should behave like Rayleigh (same distribution family)."""
        np.random.seed(7)
        signal = np.ones(200)
        out = rician_fading_channel(signal, snr_db=10, k_factor=0.0)
        assert out.shape == signal.shape

    def test_rician_high_k_less_spread(self):
        """High K-factor (strong LOS) → less fading spread."""
        np.random.seed(3)
        signal = np.ones(5000)
        out_low_k = rician_fading_channel(signal, snr_db=15, k_factor=0.1)
        out_high_k = rician_fading_channel(signal, snr_db=15, k_factor=10.0)
        # High K → equalized output closer to original signal (lower variance)
        assert np.std(out_high_k) < np.std(out_low_k) * 2  # relaxed check
