"""Unit tests for BPSK modulation."""

import numpy as np
import pytest

from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate, calculate_ber


class TestBPSKModulate:
    def test_mapping(self):
        bits = np.array([0, 1, 0, 1])
        syms = bpsk_modulate(bits)
        np.testing.assert_array_equal(syms, [-1, 1, -1, 1])

    def test_dtype(self):
        syms = bpsk_modulate(np.array([0, 1]))
        assert syms.dtype == float

    def test_large(self):
        np.random.seed(0)
        bits = np.random.randint(0, 2, 10_000)
        syms = bpsk_modulate(bits)
        assert set(syms).issubset({-1.0, 1.0})


class TestBPSKDemodulate:
    def test_clean_symbols(self):
        syms = np.array([-1.0, 1.0, -1.0, 1.0])
        bits = bpsk_demodulate(syms)
        np.testing.assert_array_equal(bits, [0, 1, 0, 1])

    def test_threshold_edge(self):
        # symbols exactly at threshold (0.0) → bit 1
        syms = np.array([0.0, 0.0001, -0.0001])
        bits = bpsk_demodulate(syms)
        np.testing.assert_array_equal(bits, [1, 1, 0])

    def test_round_trip_noiseless(self):
        np.random.seed(42)
        bits = np.random.randint(0, 2, 1000)
        recovered = bpsk_demodulate(bpsk_modulate(bits))
        np.testing.assert_array_equal(bits, recovered)


class TestCalculateBER:
    def test_no_errors(self):
        bits = np.array([0, 1, 0, 1])
        ber, errors = calculate_ber(bits, bits)
        assert ber == 0.0
        assert errors == 0

    def test_all_errors(self):
        tx = np.array([0, 0, 0])
        rx = np.array([1, 1, 1])
        ber, errors = calculate_ber(tx, rx)
        assert ber == pytest.approx(1.0)
        assert errors == 3

    def test_half_errors(self):
        tx = np.array([0, 1, 0, 1])
        rx = np.array([1, 1, 1, 1])
        ber, errors = calculate_ber(tx, rx)
        assert ber == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_ber(np.array([0, 1]), np.array([1]))
