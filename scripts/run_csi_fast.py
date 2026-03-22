import argparse, os, sys
from time import perf_counter
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.utils.activations import get_clip_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper


def plot_training_history(all_histories, out_dir):
    """Plot train loss, train accuracy and val accuracy for each model."""
    os.makedirs(out_dir, exist_ok=True)
    for name, hist in all_histories.items():
        epochs_range = range(1, len(hist['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(name, fontsize=13)

        ax1.plot(epochs_range, hist['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, hist['train_acc'], 'g-', label='Train Accuracy')
        ax2.plot(epochs_range, hist['val_acc'], 'r--', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
        path = os.path.join(out_dir, f"training_{safe_name}.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved training chart: {path}")


def main():
    snr_range = np.arange(0, 22, 4)
    samples = 15000
    epochs = 15
    relays = {
        "AF System": AmplifyAndForwardRelay(target_power=1.0),
        "DF System": DecodeAndForwardRelay(target_power=1.0)
    }
    all_histories = {}
    clip = get_clip_range("qam16")
    base_kw = dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2, clip_range=clip, prefer_gpu=True)
    train_kw = dict(training_snrs=[15], num_samples=samples, epochs=epochs,
                    training_modulation="qam16", use_rayleigh=True,
                    patience=5, min_delta=1e-5)
    
    r_base = MambaRelayWrapper(**base_kw, in_channels=1, use_input_norm=False, output_activation="tanh")
    all_histories["Mamba (Blind)"] = r_base.train(**train_kw)
    relays["Mamba (Blind)"] = r_base
    
    r_csi = MambaRelayWrapper(**base_kw, in_channels=2, use_input_norm=True, output_activation="scaled_tanh")
    all_histories["Mamba (+CSI)"] = r_csi.train(**train_kw)
    relays["Mamba (+CSI)"] = r_csi

    results = {}
    channel_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)
    for name, r in relays.items():
        _, mean, _ = run_monte_carlo(r, snr_range, num_bits_per_trial=5000, num_trials=5, channel_fn=channel_fn, modulation="qam16")
        results[name] = mean

    # Plot BER
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, mean in results.items():
        ax.semilogy(snr_range, mean, marker="o", label=name)
    ax.grid(True)
    ax.legend()
    os.makedirs("results/csi", exist_ok=True)
    plt.savefig("results/csi/csi_experiment_qam16_rayleigh.png", dpi=300)
    plt.close()

    # Training history charts
    plot_training_history(all_histories, "results/csi")
    print("Done")

if __name__ == '__main__': main()