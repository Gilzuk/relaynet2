"""
run_full_comparison.py
======================
Reproduces the full BER comparison across all relay methods on both AWGN
and fading channels, including:
  - Hybrid SNR-adaptive relay
  - 95% confidence intervals
  - Statistical significance testing
  - Comparison plots

Usage::

    python scripts/run_full_comparison.py

Output plots are saved to the ``results/`` directory.
"""

import argparse
import os
import sys

import numpy as np

# Allow running from the repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import (
    compute_confidence_interval,
    wilcoxon_test,
    significance_table,
)
from relaynet.visualization.plots import plot_ber_curves, plot_ber_with_ci
from relaynet.channels.awgn import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel


def parse_args():
    p = argparse.ArgumentParser(description="Full relay comparison")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_models(seed):
    print("\n=== Training AI relay models ===")

    print("  Training Minimal GenAI relay (169 params) …")
    genai = MinimalGenAIRelay(window_size=5, hidden_size=24)
    genai.train(training_snrs=[5, 10, 15], num_samples=25_000, epochs=100, seed=seed)

    print("  Training Hybrid relay …")
    hybrid = HybridRelay(snr_threshold=5.0)
    hybrid.train(training_snrs=[2, 4, 6], num_samples=25_000, epochs=100, seed=seed)

    print("  Training VAE relay …")
    vae = VAERelay(window_size=7, latent_size=8, beta=0.1)
    vae.train(training_snrs=[5, 10, 15], num_samples=50_000, epochs=100, seed=seed)

    print("  Training CGAN relay (WGAN-GP) …")
    cgan = CGANRelay(window_size=7, noise_size=8, lambda_gp=10, lambda_l1=20, n_critic=5)
    cgan.train(training_snrs=[5, 10, 15], num_samples=50_000, epochs=200, seed=seed)

    return {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": genai,
        "Hybrid": hybrid,
        "VAE": vae,
        "CGAN (WGAN-GP)": cgan,
    }


def run_awgn_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== AWGN Channel Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        print(f"done (mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_fading_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== Rayleigh Fading Channel Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=rayleigh_fading_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        print(f"done (mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def print_significance(snr_range, all_ber, all_trials, baseline="DF"):
    methods = [k for k in all_ber if k != baseline]
    print(f"\n=== Statistical Significance vs {baseline} ===")
    trials_dict = {k: all_trials[k] for k in all_trials}
    significance_table(snr_range, methods, trials_dict, baseline=baseline)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)
    os.makedirs("results", exist_ok=True)

    relays = train_models(args.seed)

    # AWGN comparison
    awgn_ber, awgn_trials = run_awgn_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, awgn_ber, awgn_trials)

    if not args.no_plots:
        ci_dict = {}
        for name, trials in awgn_trials.items():
            lo, hi = compute_confidence_interval(trials)
            ci_dict[name] = (lo, hi)
        plot_ber_with_ci(
            snr_range, awgn_ber, ci_dict,
            title="AWGN Channel – All Relay Methods (95% CI)",
            save_path="results/awgn_comparison_ci.png",
        )

    # Fading comparison
    fading_ber, fading_trials = run_fading_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, fading_ber, fading_trials)

    if not args.no_plots:
        plot_ber_curves(
            snr_range, fading_ber,
            title="Rayleigh Fading Channel – All Relay Methods",
            save_path="results/fading_comparison.png",
        )

    print("\nDone. Results saved to results/")


if __name__ == "__main__":
    main()
