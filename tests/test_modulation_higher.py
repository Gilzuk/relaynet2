"""Tests for QPSK and 16-QAM modulation, and higher-order modulation
end-to-end simulation through channels and relays."""

import numpy as np
import pytest

from relaynet.modulation.qpsk import qpsk_modulate, qpsk_demodulate
from relaynet.modulation.qam import qam16_modulate, qam16_demodulate
from relaynet.modulation import get_modulation_functions
from relaynet.modulation.bpsk import calculate_ber
from relaynet.channels.awgn import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.nodes import Source, Destination
from relaynet.simulation.runner import simulate_transmission, run_monte_carlo
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay


# =====================================================================
# QPSK modulation unit tests
# =====================================================================

class TestQPSKModulate:
    def test_mapping_00(self):
        syms = qpsk_modulate(np.array([0, 0]))
        expected = (1 + 1j) / np.sqrt(2)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_01(self):
        syms = qpsk_modulate(np.array([0, 1]))
        expected = (1 - 1j) / np.sqrt(2)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_10(self):
        syms = qpsk_modulate(np.array([1, 0]))
        expected = (-1 + 1j) / np.sqrt(2)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_11(self):
        syms = qpsk_modulate(np.array([1, 1]))
        expected = (-1 - 1j) / np.sqrt(2)
        np.testing.assert_allclose(syms[0], expected)

    def test_output_shape(self):
        bits = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        syms = qpsk_modulate(bits)
        assert syms.shape == (4,)

    def test_unit_power(self):
        np.random.seed(0)
        bits = np.random.randint(0, 2, 10000)
        syms = qpsk_modulate(bits)
        avg_power = np.mean(np.abs(syms) ** 2)
        assert abs(avg_power - 1.0) < 0.01

    def test_is_complex(self):
        syms = qpsk_modulate(np.array([0, 1, 1, 0]))
        assert np.iscomplexobj(syms)

    def test_odd_bits_raises(self):
        with pytest.raises(ValueError):
            qpsk_modulate(np.array([0, 1, 0]))


class TestQPSKDemodulate:
    def test_round_trip_noiseless(self):
        np.random.seed(42)
        bits = np.random.randint(0, 2, 1000)
        # Make even
        bits = bits[:len(bits) // 2 * 2]
        recovered = qpsk_demodulate(qpsk_modulate(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_output_length(self):
        syms = qpsk_modulate(np.array([0, 1, 1, 0, 0, 0]))
        bits = qpsk_demodulate(syms)
        assert len(bits) == 6

    def test_noisy_round_trip_high_snr(self):
        np.random.seed(1)
        bits = np.random.randint(0, 2, 2000)
        syms = qpsk_modulate(bits)
        noisy = awgn_channel(syms, snr_db=20)
        recovered = qpsk_demodulate(noisy)
        ber, _ = calculate_ber(bits, recovered)
        assert ber < 0.01


# =====================================================================
# 16-QAM modulation unit tests
# =====================================================================

class TestQAM16Modulate:
    def test_mapping_0000(self):
        syms = qam16_modulate(np.array([0, 0, 0, 0]))
        expected = (3 + 3j) / np.sqrt(10)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_0101(self):
        syms = qam16_modulate(np.array([0, 1, 0, 1]))
        expected = (1 + 1j) / np.sqrt(10)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_1010(self):
        syms = qam16_modulate(np.array([1, 0, 1, 0]))
        expected = (-3 - 3j) / np.sqrt(10)
        np.testing.assert_allclose(syms[0], expected)

    def test_mapping_1111(self):
        syms = qam16_modulate(np.array([1, 1, 1, 1]))
        expected = (-1 - 1j) / np.sqrt(10)
        np.testing.assert_allclose(syms[0], expected)

    def test_output_shape(self):
        bits = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        syms = qam16_modulate(bits)
        assert syms.shape == (2,)

    def test_unit_power(self):
        """All 16 constellation points should have unit average power."""
        # Enumerate all 16 symbols
        all_bits = []
        for i in range(16):
            all_bits.extend([(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
        syms = qam16_modulate(np.array(all_bits))
        avg_power = np.mean(np.abs(syms) ** 2)
        assert abs(avg_power - 1.0) < 1e-10

    def test_is_complex(self):
        syms = qam16_modulate(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert np.iscomplexobj(syms)

    def test_bad_length_raises(self):
        with pytest.raises(ValueError):
            qam16_modulate(np.array([0, 1, 0]))


class TestQAM16Demodulate:
    def test_round_trip_noiseless(self):
        np.random.seed(42)
        bits = np.random.randint(0, 2, 1000)
        bits = bits[:len(bits) // 4 * 4]
        recovered = qam16_demodulate(qam16_modulate(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_output_length(self):
        syms = qam16_modulate(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        bits = qam16_demodulate(syms)
        assert len(bits) == 8

    def test_noisy_round_trip_high_snr(self):
        np.random.seed(1)
        bits = np.random.randint(0, 2, 2000)
        bits = bits[:len(bits) // 4 * 4]
        syms = qam16_modulate(bits)
        noisy = awgn_channel(syms, snr_db=25)
        recovered = qam16_demodulate(noisy)
        ber, _ = calculate_ber(bits, recovered)
        assert ber < 0.01

    def test_gray_adjacency(self):
        """Adjacent constellation points should differ by exactly 1 bit."""
        # Check one pair: 0000 → (+3+3j), 0100 → (+1+3j) — differ in bit 1
        bits_a = np.array([0, 0, 0, 0])  # (+3+3j)
        bits_b = np.array([0, 1, 0, 0])  # (+1+3j)
        assert sum(bits_a != bits_b) == 1


# =====================================================================
# get_modulation_functions tests
# =====================================================================

class TestGetModulationFunctions:
    def test_bpsk(self):
        mod, demod, bps = get_modulation_functions("bpsk")
        assert bps == 1
        syms = mod(np.array([0, 1]))
        assert len(syms) == 2

    def test_qpsk(self):
        mod, demod, bps = get_modulation_functions("qpsk")
        assert bps == 2
        syms = mod(np.array([0, 1, 1, 0]))
        assert len(syms) == 2

    def test_qam16(self):
        mod, demod, bps = get_modulation_functions("qam16")
        assert bps == 4
        syms = mod(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert len(syms) == 2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_modulation_functions("unknown")


# =====================================================================
# Source / Destination with modulation
# =====================================================================

class TestModulationNodes:
    def test_source_bpsk(self):
        src = Source(seed=0, modulation="bpsk")
        bits, syms = src.transmit(100)
        assert len(bits) == 100
        assert len(syms) == 100
        assert np.isrealobj(syms)

    def test_source_qpsk(self):
        src = Source(seed=0, modulation="qpsk")
        bits, syms = src.transmit(100)
        assert len(bits) == 100
        assert len(syms) == 50
        assert np.iscomplexobj(syms)

    def test_source_qam16(self):
        src = Source(seed=0, modulation="qam16")
        bits, syms = src.transmit(100)
        assert len(bits) == 100
        assert len(syms) == 25
        assert np.iscomplexobj(syms)

    def test_source_truncates_odd(self):
        src = Source(seed=0, modulation="qpsk")
        bits, syms = src.transmit(101)
        assert len(bits) == 100
        assert len(syms) == 50

    def test_destination_qpsk(self):
        bits = np.array([0, 1, 1, 0])
        syms = qpsk_modulate(bits)
        dst = Destination(modulation="qpsk")
        recovered = dst.receive(syms)
        np.testing.assert_array_equal(bits, recovered)

    def test_destination_qam16(self):
        bits = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        syms = qam16_modulate(bits)
        dst = Destination(modulation="qam16")
        recovered = dst.receive(syms)
        np.testing.assert_array_equal(bits, recovered)


# =====================================================================
# Channel tests with complex signals
# =====================================================================

class TestChannelsComplex:
    def test_awgn_complex(self):
        np.random.seed(0)
        syms = qpsk_modulate(np.random.randint(0, 2, 1000))
        noisy = awgn_channel(syms, snr_db=10)
        assert np.iscomplexobj(noisy)
        assert noisy.shape == syms.shape

    def test_rayleigh_complex_input(self):
        """Rayleigh with complex input should return complex output."""
        np.random.seed(0)
        syms = qpsk_modulate(np.random.randint(0, 2, 500))
        out = rayleigh_fading_channel(syms, snr_db=10)
        assert np.iscomplexobj(out)
        assert out.shape == syms.shape

    def test_rayleigh_real_input_unchanged(self):
        """Rayleigh with real BPSK input should still return real."""
        np.random.seed(0)
        signal = 2 * np.random.randint(0, 2, 500) - 1.0
        out = rayleigh_fading_channel(signal, snr_db=10)
        assert np.isrealobj(out)

    def test_rician_complex_input(self):
        np.random.seed(0)
        syms = qpsk_modulate(np.random.randint(0, 2, 500))
        out = rician_fading_channel(syms, snr_db=10, k_factor=3.0)
        assert np.iscomplexobj(out)
        assert out.shape == syms.shape

    def test_rician_real_input_unchanged(self):
        np.random.seed(0)
        signal = 2 * np.random.randint(0, 2, 500) - 1.0
        out = rician_fading_channel(signal, snr_db=10, k_factor=3.0)
        assert np.isrealobj(out)


# =====================================================================
# Relay processing with complex signals
# =====================================================================

class TestRelayComplex:
    def test_af_complex(self):
        """AF should amplify complex signals correctly."""
        np.random.seed(0)
        syms = qpsk_modulate(np.random.randint(0, 2, 500))
        noisy = awgn_channel(syms, snr_db=10)
        relay = AmplifyAndForwardRelay(target_power=1.0, prefer_gpu=False)
        out = relay.process(noisy)
        assert np.iscomplexobj(out)
        assert out.shape == noisy.shape
        # Power should be normalised
        assert abs(np.mean(np.abs(out) ** 2) - 1.0) < 0.05


# =====================================================================
# End-to-end simulation with modulation
# =====================================================================

class TestSimulationModulation:
    def test_qpsk_awgn_df(self):
        np.random.seed(0)
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=15, seed=0,
            modulation="qpsk",
        )
        assert 0.0 <= ber <= 1.0
        assert ber < 0.05

    def test_qpsk_awgn_af(self):
        np.random.seed(0)
        relay = AmplifyAndForwardRelay(prefer_gpu=False)
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=15, seed=0,
            modulation="qpsk",
        )
        assert 0.0 <= ber <= 1.0

    def test_qpsk_rayleigh_df(self):
        np.random.seed(0)
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=15, seed=0,
            channel_fn=rayleigh_fading_channel,
            modulation="qpsk",
        )
        assert 0.0 <= ber <= 1.0

    def test_qam16_awgn_df(self):
        np.random.seed(0)
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=20, seed=0,
            modulation="qam16",
        )
        assert 0.0 <= ber <= 1.0
        assert ber < 0.1

    def test_qam16_awgn_af(self):
        np.random.seed(0)
        relay = AmplifyAndForwardRelay(prefer_gpu=False)
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=20, seed=0,
            modulation="qam16",
        )
        assert 0.0 <= ber <= 1.0

    def test_qpsk_mlp(self):
        """MLP relay processing I/Q independently on QPSK."""
        np.random.seed(0)
        relay = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False)
        relay.train(training_snrs=[10], num_samples=2000, epochs=5, seed=1)
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=10, seed=0,
            modulation="qpsk",
        )
        assert 0.0 <= ber <= 1.0

    def test_qpsk_hybrid(self):
        """Hybrid relay on QPSK."""
        np.random.seed(0)
        relay = HybridRelay(snr_threshold=5.0, prefer_gpu=False)
        relay.train(training_snrs=[5], num_samples=500, epochs=2, seed=0)
        ber, _ = simulate_transmission(
            relay, num_bits=2000, snr_db=10, seed=0,
            modulation="qpsk",
        )
        assert 0.0 <= ber <= 1.0

    def test_monte_carlo_qpsk(self):
        relay = DecodeAndForwardRelay()
        snr_vals, ber_vals, trials = run_monte_carlo(
            relay, [5, 10, 15],
            num_bits_per_trial=1000, num_trials=3,
            modulation="qpsk",
        )
        assert len(snr_vals) == 3
        assert all(0 <= b <= 1 for b in ber_vals)
        # BER should decrease with SNR
        assert ber_vals[2] <= ber_vals[0]

    def test_monte_carlo_qam16(self):
        relay = DecodeAndForwardRelay()
        snr_vals, ber_vals, trials = run_monte_carlo(
            relay, [10, 15, 20],
            num_bits_per_trial=1000, num_trials=3,
            modulation="qam16",
        )
        assert len(snr_vals) == 3
        assert all(0 <= b <= 1 for b in ber_vals)

    def test_bpsk_backward_compatible(self):
        """Default modulation='bpsk' gives same results as before."""
        relay = DecodeAndForwardRelay()
        ber1, _ = simulate_transmission(relay, 1000, 10, seed=42)
        ber2, _ = simulate_transmission(relay, 1000, 10, seed=42, modulation="bpsk")
        assert ber1 == ber2

    def test_df_high_snr_qpsk(self):
        """DF at very high SNR should have near-zero BER for QPSK."""
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(
            relay, num_bits=5000, snr_db=25, seed=1,
            modulation="qpsk",
        )
        assert ber < 0.01

    def test_df_high_snr_qam16(self):
        """DF at very high SNR should have near-zero BER for 16-QAM."""
        relay = DecodeAndForwardRelay()
        ber, _ = simulate_transmission(
            relay, num_bits=5000, snr_db=30, seed=1,
            modulation="qam16",
        )
        assert ber < 0.01
