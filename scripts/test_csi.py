import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from relaynet.channels.fading import rayleigh_fading_channel
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from relaynet.simulation.runner import run_monte_carlo

print("Building Relay w/ CSI...")
r = MambaRelayWrapper(in_channels=2, use_input_norm=True, output_activation="scaled_tanh")
r.train(training_snrs=[15], num_samples=1000, epochs=2, training_modulation="qam16", use_rayleigh=True)

print("Evaluating...")
channel_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)
snr_range = [15]
_, ber_mean, _ = run_monte_carlo(r, snr_range, num_bits_per_trial=1000, num_trials=2, channel_fn=channel_fn, modulation="qam16")
print(f"BER: {ber_mean}")
