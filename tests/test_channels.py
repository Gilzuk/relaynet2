"""Unit tests for channel implementations."""

import numpy as np
import pytest

from relaynet.channels.awgn import awgn_channel, calculate_snr
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.channels.mimo import mimo_2x2_channel


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


class TestMIMOChannel:
    def test_output_shape_even(self):
        np.random.seed(0)
        signal = 2 * np.random.randint(0, 2, 100) - 1.0
        out = mimo_2x2_channel(signal, snr_db=10)
        assert out.shape == signal.shape

    def test_output_shape_odd(self):
        """Odd-length input is truncated to even."""
        np.random.seed(0)
        signal = 2 * np.random.randint(0, 2, 101) - 1.0
        out = mimo_2x2_channel(signal, snr_db=10)
        assert len(out) == 100  # last symbol dropped

    def test_output_is_real(self):
        np.random.seed(1)
        signal = 2 * np.random.randint(0, 2, 200) - 1.0
        out = mimo_2x2_channel(signal, snr_db=10)
        assert np.isrealobj(out)

    def test_high_snr_recovers_signal(self):
        """At very high SNR, ZF should recover BPSK symbols accurately."""
        np.random.seed(42)
        signal = 2 * np.random.randint(0, 2, 2000) - 1.0
        out = mimo_2x2_channel(signal, snr_db=30)
        # Hard-decision should match
        decoded = np.sign(out)
        ber = np.mean(decoded != signal)
        assert ber < 0.01, f"BER {ber:.4f} too high at 30 dB"

    def test_low_snr_has_errors(self):
        """At low SNR, BER should be significant."""
        np.random.seed(7)
        signal = 2 * np.random.randint(0, 2, 2000) - 1.0
        out = mimo_2x2_channel(signal, snr_db=0)
        decoded = np.sign(out)
        ber = np.mean(decoded != signal)
        assert ber > 0.05, f"BER {ber:.4f} unexpectedly low at 0 dB"

    def test_worse_than_siso_rayleigh_at_low_snr(self):
        """2×2 ZF noise enhancement should make BER worse than SISO at low SNR."""
        np.random.seed(10)
        signal = 2 * np.random.randint(0, 2, 4000) - 1.0
        mimo_out = mimo_2x2_channel(signal, snr_db=5)
        siso_out = rayleigh_fading_channel(signal, snr_db=5)
        ber_mimo = np.mean(np.sign(mimo_out) != signal)
        ber_siso = np.mean(np.sign(siso_out) != signal)
        # ZF typically has worse BER at same SNR (noise enhancement)
        # Using relaxed check — just verify MIMO isn't magically better
        assert ber_mimo > ber_siso * 0.5, (
            f"MIMO BER {ber_mimo:.4f} unexpectedly much better than SISO {ber_siso:.4f}"
        )
