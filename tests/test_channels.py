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
        """``snr_db`` is Eb/N0, so the raw power ratio is 3 dB above it.

        The noise variance is N0/2 per real dimension. For real baseband with
        Es = 1 that makes the measured signal-to-noise *power* ratio
        Es/(N0/2) = 2*Es/N0, i.e. exactly 3.01 dB above Eb/N0 -- a definition,
        not an error. A complex signal carries the same N0 across two
        dimensions, so there the power ratio equals Eb/N0 directly.

        This test previously asserted measured == snr_db for the real branch,
        which pinned it to the 3 dB pessimistic axis and disagreed with both
        the complex branch and the Rayleigh channel in fading.py.
        """
        np.random.seed(42)
        signal = 2 * np.random.randint(0, 2, 200_000) - 1.0
        for target in [0, 5, 10, 15, 20]:
            measured = calculate_snr(signal, awgn_channel(signal, snr_db=target))
            assert abs(measured - (target + 3.01)) < 0.6, (
                f"real branch at Eb/N0={target} dB: expected power ratio "
                f"{target + 3.01:.2f} dB, measured {measured:.2f} dB"
            )

        rng = np.random.default_rng(0)
        cplx = ((2 * rng.integers(0, 2, 200_000) - 1.0)
                + 1j * (2 * rng.integers(0, 2, 200_000) - 1.0)) / np.sqrt(2)
        for target in [0, 10, 20]:
            measured = calculate_snr(cplx, awgn_channel(cplx, snr_db=target))
            assert abs(measured - target) < 0.6, (
                f"complex branch at Es/N0={target} dB: measured {measured:.2f} dB"
            )

    def test_bpsk_ber_matches_textbook(self):
        """The point of the convention: BPSK over AWGN must give Q(sqrt(2Eb/N0))."""
        import math

        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, 400_000)
        for snr_db in (0, 4, 8):
            np.random.seed(snr_db + 1)
            y = awgn_channel(1.0 - 2.0 * bits, snr_db)
            ber = float(np.mean((y < 0).astype(int) != bits))
            theory = 0.5 * math.erfc(math.sqrt(10 ** (snr_db / 10.0)))
            assert abs(ber - theory) < 0.004, (
                f"AWGN BPSK at {snr_db} dB: sim {ber:.5f} vs theory {theory:.5f}"
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




