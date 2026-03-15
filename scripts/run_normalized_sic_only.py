#!/usr/bin/env python
"""Train normalized 3K models and evaluate only on MIMO SIC channel.

This is a one-off script to fill the missing SIC data in the thesis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from time import perf_counter

from checkpoints.checkpoint_22_normalized_3k import build_all_3k
from relaynet.channels.mimo import mimo_2x2_sic_channel
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import compute_confidence_interval
from relaynet.visualization.plots import plot_ber_with_ci

SNR_RANGE = np.arange(0, 21, 2)
BITS_PER_TRIAL = 10_000
NUM_TRIALS = 10
SEED = 42


def main():
    t0 = perf_counter()

    # ── Build & train all six 3K models ──────────────────────────────
    relays = build_all_3k(prefer_gpu=True, include_sequence_models=True)

    train_cfg = {
        "GenAI-3K":       dict(training_snrs=[5, 10, 15], num_samples=50_000, epochs=100),
        "Hybrid-3K":      dict(training_snrs=[2, 4, 6],   num_samples=50_000, epochs=100),
        "VAE-3K":         dict(training_snrs=[5, 10, 15], num_samples=50_000, epochs=100),
        "CGAN-3K":        dict(training_snrs=[5, 10, 15], num_samples=50_000, epochs=200),
        "Transformer-3K": dict(training_snrs=[5, 10, 15], num_samples=50_000, epochs=100, lr=0.001),
        "Mamba-3K":       dict(training_snrs=[5, 10, 15], num_samples=50_000, epochs=100, lr=0.001),
    }

    for name, relay in relays.items():
        cfg = train_cfg[name]
        print(f"Training {name} ({relay.num_params}p) …", flush=True)
        ts = perf_counter()
        relay.train(**cfg, seed=SEED)
        print(f"  done in {perf_counter()-ts:.1f}s")

    # ── Evaluate on MIMO SIC only ────────────────────────────────────
    print(f"\n=== Normalized 3K: 2×2 MIMO SIC ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        ts = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, SNR_RANGE,
            num_bits_per_trial=BITS_PER_TRIAL,
            num_trials=NUM_TRIALS,
            channel_fn=mimo_2x2_sic_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        print(f"done ({perf_counter()-ts:.1f}s)  BER=[{ber.min():.2e}, {ber.max():.2e}]")

    # ── Print table ──────────────────────────────────────────────────
    header = f"{'SNR':>8}" + "".join(f"{n:>18}" for n in relays)
    print("\n" + header)
    for i, snr in enumerate(SNR_RANGE):
        row = f"{snr:>6}dB"
        for name in relays:
            row += f"  {all_ber[name][i]:>14.4e}"
        print(row)

    # ── Plot ─────────────────────────────────────────────────────────
    ci_dict = {}
    for name, trials in all_trials.items():
        lo, hi = compute_confidence_interval(trials)
        ci_dict[name] = (lo, hi)
    plot_ber_with_ci(
        SNR_RANGE, all_ber, ci_dict,
        title="Normalized ~3K params – 2×2 MIMO SIC (95% CI)",
        save_path="results/normalized_mimo_sic_3k_ci.png",
    )

    print(f"\nTotal time: {perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
