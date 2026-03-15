"""Unit tests for relay implementations."""

import numpy as np
import pytest

from relaynet.channels.awgn import awgn_channel
from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay, estimate_snr
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay


@pytest.fixture
def bpsk_signal():
    np.random.seed(0)
    bits = np.random.randint(0, 2, 500)
    return bpsk_modulate(bits), bits


class TestAFRelay:
    def test_output_shape(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = AmplifyAndForwardRelay()
        out = relay.process(awgn_channel(symbols, 10))
        assert out.shape == symbols.shape

    def test_power_normalisation(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = AmplifyAndForwardRelay(target_power=1.0)
        noisy = awgn_channel(symbols, 5)
        out = relay.process(noisy)
        assert abs(np.mean(out ** 2) - 1.0) < 0.05


class TestDFRelay:
    def test_output_shape(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = DecodeAndForwardRelay()
        out = relay.process(awgn_channel(symbols, 10))
        assert out.shape == symbols.shape

    def test_high_snr_perfect(self, bpsk_signal):
        """At very high SNR DF should reconstruct the signal perfectly."""
        symbols, bits = bpsk_signal
        relay = DecodeAndForwardRelay()
        noisy = awgn_channel(symbols, 40)
        out = relay.process(noisy)
        recovered = bpsk_demodulate(out)
        assert np.array_equal(bits, recovered)

    def test_power_normalisation(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = DecodeAndForwardRelay(target_power=1.0)
        out = relay.process(awgn_channel(symbols, 10))
        assert abs(np.mean(out ** 2) - 1.0) < 0.01


class TestMinimalGenAIRelay:
    def test_untrained_passthrough(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = MinimalGenAIRelay()
        out = relay.process(symbols)
        assert out.shape == symbols.shape

    def test_train_and_process(self, bpsk_signal):
        np.random.seed(1)
        symbols, _ = bpsk_signal
        relay = MinimalGenAIRelay(window_size=5, hidden_size=24)
        relay.train(training_snrs=[10], num_samples=2000, epochs=5, seed=1)
        assert relay.is_trained
        out = relay.process(awgn_channel(symbols, 10))
        assert out.shape == symbols.shape

    def test_param_count(self):
        relay = MinimalGenAIRelay(window_size=5, hidden_size=24)
        # 5*24 + 24 + 24*1 + 1 = 169
        assert relay.num_params == 169


class TestHybridRelay:
    def test_estimate_snr_high(self):
        """At high SNR the estimator should give a positive result."""
        np.random.seed(0)
        signal = bpsk_modulate(np.random.randint(0, 2, 5000))
        noisy = awgn_channel(signal, 15)
        est = estimate_snr(noisy)
        assert est > 0  # should detect a positive SNR

    def test_routes_to_df_above_threshold(self):
        """Above threshold the hybrid relay should behave like DF (output ±1.0)."""
        from relaynet.channels.awgn import awgn_channel
        np.random.seed(5)
        bits = np.random.randint(0, 2, 1000)
        signal = bpsk_modulate(bits)
        # Nearly noiseless signal (30 dB >> 5 dB threshold).
        # The SNR estimator computes snr ≈ tx_power / (rx_power - tx_power)
        # which will be >> 5 dB for a nearly noiseless signal.
        nearly_noiseless = awgn_channel(signal, snr_db=30)
        relay = HybridRelay(snr_threshold=5.0)
        out = relay.process(nearly_noiseless)
        # DF relay always outputs normalised ±1.0 BPSK symbols
        np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-9)

    def test_train_sets_flag(self):
        relay = HybridRelay()
        relay.train(training_snrs=[5], num_samples=500, epochs=2, seed=0)
        assert relay.is_trained
        assert relay.genai_relay.is_trained


class TestVAERelay:
    def test_train_and_process(self, bpsk_signal):
        np.random.seed(2)
        symbols, _ = bpsk_signal
        relay = VAERelay(window_size=7, latent_size=4)
        relay.train(training_snrs=[10], num_samples=500, epochs=2, seed=2)
        assert relay.is_trained
        out = relay.process(awgn_channel(symbols, 10))
        assert out.shape == symbols.shape

    def test_untrained_passthrough(self, bpsk_signal):
        symbols, _ = bpsk_signal
        relay = VAERelay()
        out = relay.process(symbols)
        assert out.shape == symbols.shape


class TestCGANRelay:
    def test_train_and_process(self, bpsk_signal):
        np.random.seed(3)
        symbols, _ = bpsk_signal
        relay = CGANRelay(window_size=7)
        relay.train(training_snrs=[10], num_samples=500, epochs=2, seed=3)
        assert relay.is_trained
        out = relay.process(awgn_channel(symbols, 10))
        assert out.shape == symbols.shape


# ======================================================================
# Weight save / load round-trip tests
# ======================================================================


class TestWeightSaveLoad:
    """Verify that trained relay weights survive a save→load round-trip."""

    def test_genai_save_load(self, bpsk_signal, tmp_path):
        symbols, _ = bpsk_signal
        relay = MinimalGenAIRelay(window_size=5, hidden_size=24)
        relay.train(training_snrs=[10], num_samples=2000, epochs=5, seed=1)
        noisy = awgn_channel(symbols, 10)
        out_before = relay.process(noisy)

        path = str(tmp_path / "genai.pt")
        relay.save_weights(path)

        relay2 = MinimalGenAIRelay(window_size=5, hidden_size=24)
        relay2.load_weights(path)
        assert relay2.is_trained
        out_after = relay2.process(noisy)
        np.testing.assert_array_almost_equal(out_before, out_after)

    def test_vae_save_load(self, bpsk_signal, tmp_path):
        symbols, _ = bpsk_signal
        relay = VAERelay(window_size=7, latent_size=4)
        relay.train(training_snrs=[10], num_samples=500, epochs=2, seed=2)
        noisy = awgn_channel(symbols, 10)
        out_before = relay.process(noisy)

        path = str(tmp_path / "vae.pt")
        relay.save_weights(path)

        relay2 = VAERelay(window_size=7, latent_size=4)
        relay2.load_weights(path)
        assert relay2.is_trained
        out_after = relay2.process(noisy)
        np.testing.assert_array_almost_equal(out_before, out_after)

    def test_cgan_save_load(self, bpsk_signal, tmp_path):
        symbols, _ = bpsk_signal
        relay = CGANRelay(window_size=7)
        relay.train(training_snrs=[10], num_samples=500, epochs=2, seed=3)
        noisy = awgn_channel(symbols, 10)
        out_before = relay.process(noisy)

        path = str(tmp_path / "cgan.pt")
        relay.save_weights(path)

        relay2 = CGANRelay(window_size=7)
        relay2.load_weights(path)
        assert relay2.is_trained
        out_after = relay2.process(noisy)
        np.testing.assert_array_almost_equal(out_before, out_after)

    def test_hybrid_save_load(self, bpsk_signal, tmp_path):
        relay = HybridRelay()
        relay.train(training_snrs=[5], num_samples=500, epochs=2, seed=0)
        symbols, _ = bpsk_signal
        noisy = awgn_channel(symbols, 10)
        out_before = relay.process(noisy)

        path = str(tmp_path / "hybrid.pt")
        relay.save_weights(path)

        relay2 = HybridRelay()
        relay2.load_weights(path)
        assert relay2.is_trained
        out_after = relay2.process(noisy)
        np.testing.assert_array_almost_equal(out_before, out_after)

    def test_checkpoint_manager_round_trip(self, tmp_path):
        from relaynet.utils.checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(str(tmp_path / "weights"))

        relay = MinimalGenAIRelay(window_size=5, hidden_size=24)
        relay.train(training_snrs=[10], num_samples=2000, epochs=5, seed=42)
        relays = {"GenAI (169p)": relay}
        saved = mgr.save_all(relays, seed=42)
        assert "GenAI (169p)" in saved
        assert mgr.has_checkpoint(42)
        assert 42 in mgr.list_checkpoints()

        relay2 = MinimalGenAIRelay(window_size=5, hidden_size=24)
        relays2 = {"GenAI (169p)": relay2}
        loaded, skipped = mgr.load_all(relays2, seed=42)
        assert "GenAI (169p)" in loaded
        assert relay2.is_trained

        meta = mgr.get_metadata(42)
        assert meta is not None
        assert meta["seed"] == 42
        assert "GenAI (169p)" in meta["relays"]
