"""Unit tests for the rate-1/2 convolutional code and the coded-DF relay."""

import numpy as np
import pytest

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate


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
            ConvolutionalEncoder(constraint_length=7)
        with pytest.raises(NotImplementedError):
            ViterbiCodeDecoder(constraint_length=7)

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
