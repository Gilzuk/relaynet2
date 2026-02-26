"""Unit tests for simulation runner."""

import numpy as np
import pytest

from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.simulation.runner import simulate_transmission, run_monte_carlo


class TestSimulateTransmission:
    def test_ber_range(self):
        np.random.seed(0)
        relay = DecodeAndForwardRelay()
        ber, errors = simulate_transmission(relay, num_bits=1000, snr_db=10, seed=0)
        assert 0.0 <= ber <= 1.0
        assert 0 <= errors <= 1000

    def test_high_snr_low_ber(self):
        np.random.seed(0)
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(relay, num_bits=5000, snr_db=20, seed=1)
        assert ber < 0.01

    def test_seed_reproducibility(self):
        relay = DecodeAndForwardRelay()
        ber1, _ = simulate_transmission(relay, 1000, 10, seed=42)
        ber2, _ = simulate_transmission(relay, 1000, 10, seed=42)
        assert ber1 == ber2


class TestRunMonteCarlo:
    def test_output_shapes(self):
        relay = AmplifyAndForwardRelay()
        snr = [0, 5, 10]
        snr_vals, ber_vals, ber_trials = run_monte_carlo(
            relay, snr, num_bits_per_trial=500, num_trials=3
        )
        assert len(snr_vals) == 3
        assert len(ber_vals) == 3
        assert ber_trials.shape == (3, 3)

    def test_ber_decreases_with_snr(self):
        np.random.seed(7)
        relay = DecodeAndForwardRelay()
        snr = [0, 10, 20]
        _, ber, _ = run_monte_carlo(
            relay, snr, num_bits_per_trial=2000, num_trials=5, seed_offset=0
        )
        # BER should generally decrease as SNR increases
        assert ber[2] <= ber[0]

    def test_custom_channel(self):
        from relaynet.channels.fading import rayleigh_fading_channel
        relay = DecodeAndForwardRelay()
        snr_vals, ber_vals, _ = run_monte_carlo(
            relay, [5, 10], num_bits_per_trial=500, num_trials=2,
            channel_fn=rayleigh_fading_channel,
        )
        assert all(0 <= b <= 1 for b in ber_vals)
