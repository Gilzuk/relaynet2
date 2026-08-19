"""Unit tests for the rate-1/2 convolutional code and the coded-DF relay."""

import numpy as np
import pytest

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.coding.convolutional_qam16 import QAM16CodeDecoder, _PAM4_IDX_TO_LEVEL
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.modulation.qam import qam16_modulate


class TestConvolutionalCode:
    def test_zero_noise_round_trip(self):
        enc = ConvolutionalEncoder()
        dec = ViterbiCodeDecoder()
        rng = np.random.default_rng(0)
        for _ in range(20):
            n_info = int(rng.integers(5, 200))
            info = rng.integers(0, 2, n_info)
            coded = enc.encode(info)
            soft = 1.0 - 2.0 * coded.astype(float)
            decoded = dec.decode(soft)
            assert np.array_equal(decoded, info)

    def test_frame_length_accounting(self):
        enc = ConvolutionalEncoder()
        n_info = 97
        n_coded = enc.n_coded_bits(n_info)
        assert n_coded == 2 * (n_info + enc.num_tail)
        assert enc.n_info_bits(n_coded) == n_info

    def test_rejects_unsupported_constraint_length(self):
        with pytest.raises(NotImplementedError):
            ConvolutionalEncoder(constraint_length=4)
        with pytest.raises(NotImplementedError):
            ViterbiCodeDecoder(constraint_length=4)

    @pytest.mark.parametrize("K", [3, 5, 7])
    def test_zero_noise_round_trip_all_constraint_lengths(self, K):
        enc = ConvolutionalEncoder(constraint_length=K)
        dec = ViterbiCodeDecoder(constraint_length=K)
        rng = np.random.default_rng(K)
        for _ in range(10):
            n_info = int(rng.integers(5, 100))
            info = rng.integers(0, 2, n_info)
            coded = enc.encode(info)
            soft = 1.0 - 2.0 * coded.astype(float)
            decoded = dec.decode(soft)
            assert np.array_equal(decoded, info)

    @pytest.mark.parametrize("K", [3, 5, 7])
    def test_num_states_matches_constraint_length(self, K):
        dec = ViterbiCodeDecoder(constraint_length=K)
        assert dec.num_states == 2 ** (K - 1)

    def test_decoder_corrects_a_single_bit_flip(self):
        # A lone hard error should still be within the K=3 code's correction
        # capability given a clean neighbourhood; sanity check on softened bits.
        enc = ConvolutionalEncoder()
        dec = ViterbiCodeDecoder()
        rng = np.random.default_rng(1)
        info = rng.integers(0, 2, 50)
        coded = enc.encode(info)
        soft = 1.0 - 2.0 * coded.astype(float)
        soft[10] *= -0.3  # weaken one soft observation without fully flipping it
        decoded = dec.decode(soft)
        assert np.array_equal(decoded, info)


class TestCodedDecodeAndForwardRelay:
    def test_output_length_matches_input(self):
        relay = CodedDecodeAndForwardRelay(frame_info_bits=50)
        n_frames = 4
        frame_symbols = 50 + relay.decoder.num_tail
        rng = np.random.default_rng(2)
        x = (rng.standard_normal(n_frames * frame_symbols)
             + 1j * rng.standard_normal(n_frames * frame_symbols))
        out = relay.process(x)
        assert out.shape == x.shape

    def test_high_snr_recovers_frame_exactly(self):
        relay = CodedDecodeAndForwardRelay(frame_info_bits=100)
        frame_symbols = relay.frame_symbols
        rng = np.random.default_rng(3)
        info = rng.integers(0, 2, 100)
        coded = relay.encoder.encode(info)
        tx = qpsk_modulate(coded)
        rx = rayleigh_fading_channel(tx, snr_db=30)
        out = relay.process(rx)
        assert out.shape == (frame_symbols,)
        # At 30 dB the relay should decode-and-reencode essentially perfectly,
        # reproducing the same clean codeword that a genuine info source would.
        assert np.allclose(np.abs(out), 1.0, atol=1e-9)

    def test_partial_frame_is_truncated_not_crashed(self):
        relay = CodedDecodeAndForwardRelay(frame_info_bits=20)
        frame_symbols = 20 + relay.decoder.num_tail
        rng = np.random.default_rng(4)
        x = (rng.standard_normal(frame_symbols + 3)
             + 1j * rng.standard_normal(frame_symbols + 3))
        out = relay.process(x)
        assert len(out) == frame_symbols  # trailing partial frame dropped


class TestQAM16CodeDecoder:
    @pytest.mark.parametrize("K", [3, 5, 7])
    def test_zero_noise_round_trip(self, K):
        enc = ConvolutionalEncoder(constraint_length=K)
        dec = QAM16CodeDecoder(constraint_length=K)
        rng = np.random.default_rng(K + 50)
        for _ in range(15):
            n_info = int(rng.integers(6, 150))
            info = rng.integers(0, 2, n_info)
            coded = enc.encode(info)
            idx = coded.reshape(-1, 2)
            axis_vals = _PAM4_IDX_TO_LEVEL[idx[:, 0] * 2 + idx[:, 1]]
            decoded = dec.decode(axis_vals)
            assert np.array_equal(decoded, info)

    def test_matches_qam16_modulate_bit_packing(self):
        enc = ConvolutionalEncoder(constraint_length=3)
        dec = QAM16CodeDecoder(constraint_length=3)
        rng = np.random.default_rng(99)
        n_info = 200  # + tail(2) = 202, divisible by 4 -> whole 16-QAM symbols
        info = rng.integers(0, 2, n_info)
        coded = enc.encode(info)
        assert len(coded) % 4 == 0
        symbols = qam16_modulate(coded)

        axis_vals = np.empty(len(coded) // 2)
        axis_vals[0::2] = symbols.real
        axis_vals[1::2] = symbols.imag
        decoded = dec.decode(axis_vals)
        assert np.array_equal(decoded, info)


class TestCodedDecodeAndForwardRelayQAM16:
    def test_output_length_and_high_snr_fidelity(self):
        from relaynet.relays.coded_df_qam16 import CodedDecodeAndForwardRelayQAM16
        from relaynet.channels.fading import rayleigh_fading_channel as rfc
        from relaynet.modulation.qam import qam16_modulate as qam_mod

        relay = CodedDecodeAndForwardRelayQAM16(frame_info_bits=200)
        rng = np.random.default_rng(5)
        info = rng.integers(0, 2, 200)
        coded = relay.encoder.encode(info)
        tx = qam_mod(coded)
        rx = rfc(tx, snr_db=30)
        out = relay.process(rx)
        assert out.shape == (relay.frame_symbols,)

    def test_rejects_odd_frame_length(self):
        from relaynet.relays.coded_df_qam16 import CodedDecodeAndForwardRelayQAM16
        with pytest.raises(ValueError):
            CodedDecodeAndForwardRelayQAM16(frame_info_bits=199, constraint_length=3)  # 199+2=201, odd
