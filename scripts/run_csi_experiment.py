"""
run_csi_experiment.py
============================
Experiment: Evaluate whether introducing Channel State Information (CSI) 
as an explicit secondary input feature helps Sequence Realy Models 
exceed Amplify-and-Forward (AF) performance for 16-QAM in Rayleigh.

Usage::
    python scripts/run_csi_experiment.py --quick
"""

import argparse
import os
import sys
from time import perf_counter

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.utils.activations import get_clip_range

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper


def plot_training_history(all_histories, out_dir):
    """Plot train loss, train accuracy and val accuracy for each model."""
    if not _HAS_PLT:
        return
    os.makedirs(out_dir, exist_ok=True)
    for name, hist in all_histories.items():
        epochs_range = range(1, len(hist['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(name, fontsize=13)

        # Loss
        ax1.plot(epochs_range, hist['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stopping patience (0=disabled)")
    p.add_argument("--min-delta", type=float, default=1e-5,
                   help="Min loss improvement for early stopping")
    return p.parse_args()

def evaluate(relays, snr_range, args, modulation="qam16"):
    results = {}
    channel_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)
    for name, relay in relays.items():
        print(f"    {name} ...", end=" ", flush=True)
        t0 = perf_counter()
        _, ber_mean, ber_trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            seed_offset=args.seed,
            modulation=modulation,
        )
        print(f"done ({perf_counter()-t0:.1f}s)")
        results[name] = (ber_mean, ber_trials)
    return results

def plot_csi_results(results, snr_range, out_path):
    if not _HAS_PLT: return
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {
        "AF": ("#999999", "x", "--"),
        "DF": ("#555555", "+", "-."),
        "Mamba S6 (Baseline)": ("#0072b2", "^", "dotted"),
        "Mamba S6 (+CSI)": ("#009e73", "*", "-")
    }
    for name, (ber_mean, _) in results.items():
        color, marker, ls = styles.get(name, ("k", "o", "-"))
        ax.semilogy(snr_range, ber_mean, marker=marker, color=color, linestyle=ls, label=name, markersize=8)
    
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("16-QAM in Rayleigh Fading: Baseline vs CSI Injection")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
    
    samples = 1000 if args.quick else 20000
    epochs = 5 if args.quick else 25
    
    print("--- CSI Injection Experiment (16-QAM / Rayleigh) ---")
    clip = get_clip_range("qam16")
    
    relays = {
        "AF": AmplifyAndForwardRelay(target_power=1.0),
        "DF": DecodeAndForwardRelay(target_power=1.0)
    }
    all_histories = {}
    
    base_kw = dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2, clip_range=clip, prefer_gpu=args.gpu)
    train_kw = dict(training_snrs=[5, 10, 15], num_samples=samples, epochs=epochs,
                    training_modulation="qam16", use_rayleigh=True,
                    patience=args.patience, min_delta=args.min_delta)
    
    print("\nBuilding Mamba S6 (Baseline)...")
    r_base = MambaRelayWrapper(**base_kw, in_channels=1, use_input_norm=False, output_activation="hardtanh")
    all_histories["Mamba S6 (Baseline)"] = r_base.train(**train_kw)
    relays["Mamba S6 (Baseline)"] = r_base
    
    print("\nBuilding Mamba S6 (+CSI)...")
    r_csi = MambaRelayWrapper(**base_kw, in_channels=2, use_input_norm=False, output_activation="hardtanh")
    all_histories["Mamba S6 (+CSI)"] = r_csi.train(**train_kw)
    relays["Mamba S6 (+CSI)"] = r_csi
    
    print("\nEvaluating...")
    res = evaluate(relays, snr_range, args, "qam16")
    
    # print BER table
    print("\n--- BER Summary ---")
    print(f"{'SNR':>5} |", end="")
    for name in relays.keys():
        print(f" {name[:12]:>12} |", end="")
    print()
    for i, snr in enumerate(snr_range):
        print(f"{snr:5.1f} |", end="")
        for name in relays.keys():
            ber = res[name][0][i]
            print(f" {ber:12.4e} |", end="")
        print()
    
    out_path = "results/csi/csi_experiment_qam16_rayleigh.png"
    plot_csi_results(res, snr_range, out_path)
    print(f"\nSaved plot to {out_path}")

    # Training history charts
    plot_training_history(all_histories, "results/csi")
    print("DONE.")

if __name__ == "__main__":
    main()