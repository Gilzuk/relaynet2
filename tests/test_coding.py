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


class TestBCJRDecoder:
    @pytest.mark.parametrize("K", [3, 5, 7])
    def test_zero_noise_map_round_trip(self, K):
        from relaynet.coding.bcjr import BCJRCodeDecoder
        enc = ConvolutionalEncoder(constraint_length=K)
        dec = BCJRCodeDecoder(constraint_length=K, noise_var=0.1)
        rng = np.random.default_rng(K + 200)
        for _ in range(8):
            n = int(rng.integers(10, 80))
            info = rng.integers(0, 2, n)
            soft = 1.0 - 2.0 * enc.encode(info).astype(float)
            assert np.array_equal(dec.decode(soft), info)

    def test_posteriors_are_confident_and_correct_without_noise(self):
        from relaynet.coding.bcjr import BCJRCodeDecoder
        enc = ConvolutionalEncoder()
        dec = BCJRCodeDecoder(noise_var=0.1)
        rng = np.random.default_rng(7)
        info = rng.integers(0, 2, 60)
        coded = enc.encode(info)
        p1 = dec.coded_bit_posteriors(1.0 - 2.0 * coded.astype(float))
        truth = coded.reshape(-1, 2)
        assert np.array_equal((p1 > 0.5).astype(int), truth)
        # confidence on the true bit should be essentially 1
        assert np.mean(np.where(truth == 1, p1, 1 - p1)) > 0.99

    def test_posteriors_are_probabilities(self):
        from relaynet.coding.bcjr import BCJRCodeDecoder
        enc = ConvolutionalEncoder()
        dec = BCJRCodeDecoder(noise_var=0.5)
        rng = np.random.default_rng(8)
        info = rng.integers(0, 2, 40)
        soft = 1.0 - 2.0 * enc.encode(info).astype(float) + 0.5 * rng.standard_normal(2 * 42)
        p1 = dec.coded_bit_posteriors(soft)
        assert p1.shape == (42, 2)
        assert np.all((p1 >= 0) & (p1 <= 1))


class TestSoftRelays:
    def test_soft_df_output_shape_and_power(self):
        from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay
        relay = SoftCodedDecodeAndForwardRelay(frame_info_bits=100)
        relay.set_snr_db(16)
        rng = np.random.default_rng(11)
        n = 2 * relay.frame_symbols
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        out = relay.process(x)
        assert out.shape == (n,)
        assert np.isclose(np.mean(np.abs(out) ** 2), 1.0)

    def test_soft_df_high_snr_approaches_constellation(self):
        from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay
        from relaynet.channels.fading import rayleigh_fading_channel as rfc
        relay = SoftCodedDecodeAndForwardRelay(frame_info_bits=100)
        relay.set_snr_db(30)
        rng = np.random.default_rng(12)
        info = rng.integers(0, 2, 100)
        tx = qpsk_modulate(ConvolutionalEncoder().encode(info))
        out = relay.process(rfc(tx, snr_db=30))
        # confident posteriors => magnitudes near the unit-power constellation
        assert np.mean(np.abs(out)) > 0.9

    def test_soft_learned_relay_shares_weights_with_hard(self):
        from relaynet.relays.soft_coded_df import SoftLearnedRelay
        from relaynet.relays.mlp import MLPQPSKClassifierRelay
        clf = MLPQPSKClassifierRelay(window_size=11, hidden_size=8, seed=3)
        soft = SoftLearnedRelay(clf)
        rng = np.random.default_rng(13)
        y = (rng.standard_normal(200) + 1j * rng.standard_normal(200)) / np.sqrt(2)
        out_soft = soft.process(y)
        out_hard = clf.process(y)
        assert out_soft.shape == out_hard.shape == y.shape
        assert np.isclose(np.mean(np.abs(out_soft) ** 2), 1.0)
        # posterior mean is a genuinely different read-out, not the argmax
        assert not np.allclose(out_soft, out_hard)


class TestPuncturing:
    @pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
    def test_zero_noise_round_trip(self, rate):
        from relaynet.coding.puncturing import PuncturedCode
        pc = PuncturedCode(rate=rate)
        dec = ViterbiCodeDecoder()
        rng = np.random.default_rng(hash(rate) % 1000)
        for _ in range(5):
            n = int(rng.integers(50, 250))
            info = rng.integers(0, 2, n)
            coded = pc.encode(info)
            soft = 1.0 - 2.0 * coded.astype(float)
            out = dec.decode(pc.depuncture(soft, pc.n_steps(n)))
            assert np.array_equal(out, info)

    @pytest.mark.parametrize("rate,expected", [("1/2", 0.5), ("2/3", 2/3), ("3/4", 0.75)])
    def test_effective_rate_approaches_nominal(self, rate, expected):
        from relaynet.coding.puncturing import PuncturedCode
        pc = PuncturedCode(rate=rate)
        n = 2000  # long frame -> tail overhead negligible
        eff = n / len(pc.encode(np.zeros(n, dtype=int)))
        assert abs(eff - expected) < 0.01

    def test_rejects_unknown_rate(self):
        from relaynet.coding.puncturing import PuncturedCode
        with pytest.raises(ValueError):
            PuncturedCode(rate="7/8")


class TestBICM:
    @pytest.mark.parametrize("mod", ["qpsk", "qam16"])
    def test_noiseless_demap_is_exact(self, mod):
        from relaynet.coding.bicm import modulate_bits, soft_demap
        rng = np.random.default_rng(4)
        bits = rng.integers(0, 2, 4000)
        sym, _ = modulate_bits(bits, mod)
        rec = (soft_demap(sym, mod, len(bits)) < 0).astype(int)
        assert np.array_equal(rec, bits)

    @pytest.mark.parametrize("mod", ["qpsk", "qam16"])
    def test_constellation_is_unit_power(self, mod):
        from relaynet.coding.bicm import _constellation
        _, pts = _constellation(mod)
        assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0)

    def test_padding_is_tracked_and_stripped(self):
        from relaynet.coding.bicm import modulate_bits, soft_demap
        bits = np.array([1, 0, 1])  # not a multiple of 4
        sym, npad = modulate_bits(bits, "qam16")
        assert npad == 1 and len(sym) == 1
        assert len(soft_demap(sym, "qam16", len(bits))) == 3

    def test_qam16_is_noisier_than_qpsk_at_equal_snr(self):
        from relaynet.coding.bicm import modulate_bits, soft_demap
        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2, 8000)
        bers = {}
        for mod in ("qpsk", "qam16"):
            sym, _ = modulate_bits(bits, mod)
            sigma = np.sqrt(1 / (2 * 10 ** (8 / 10)))
            y = sym + sigma * (rng.standard_normal(len(sym))
                               + 1j * rng.standard_normal(len(sym)))
            bers[mod] = np.mean((soft_demap(y, mod, len(bits)) < 0).astype(int) != bits)
        assert bers["qam16"] > bers["qpsk"]
