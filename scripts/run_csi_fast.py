import argparse, os, sys
from time import perf_counter
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.utils.activations import get_clip_range
import matplotlib.pyplot as plt
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper

def main():
    snr_range = np.arange(0, 22, 4)
    samples = 15000
    epochs = 15
    relays = {
        "AF System": AmplifyAndForwardRelay(target_power=1.0),
        "DF System": DecodeAndForwardRelay(target_power=1.0)
    }
    clip = get_clip_range("qam16")
    base_kw = dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2, clip_range=clip, prefer_gpu=True)
    
    r_base = MambaRelayWrapper(**base_kw, in_channels=1, use_input_norm=False, output_activation="tanh")
    r_base.train(training_snrs=[15], num_samples=samples, epochs=epochs, training_modulation="qam16", use_rayleigh=True)
    relays["Mamba (Blind)"] = r_base
    
    r_csi = MambaRelayWrapper(**base_kw, in_channels=2, use_input_norm=True, output_activation="scaled_tanh")
    r_csi.train(training_snrs=[15], num_samples=samples, epochs=epochs, training_modulation="qam16", use_rayleigh=True)
    relays["Mamba (+CSI)"] = r_csi

    results = {}
    channel_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)
    for name, r in relays.items():
        _, mean, _ = run_monte_carlo(r, snr_range, num_bits_per_trial=5000, num_trials=5, channel_fn=channel_fn, modulation="qam16")
        results[name] = mean

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, mean in results.items():
        ax.semilogy(snr_range, mean, marker="o", label=name)
    ax.grid(True)
    ax.legend()
    os.makedirs("results/csi", exist_ok=True)
    plt.savefig("results/csi/csi_experiment_qam16_rayleigh.png", dpi=300)
    print("Done")

if __name__ == '__main__': main()