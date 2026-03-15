"""
run_modulation_comparison.py
============================
Compare relay strategies across BPSK, QPSK, and 16-QAM modulations
on AWGN and Rayleigh fading channels.

This script tests whether the thesis hypotheses (H1–H6) generalise
beyond BPSK to higher-order modulations.

Usage::

    python scripts/run_modulation_comparison.py
    python scripts/run_modulation_comparison.py --quick
    python scripts/run_modulation_comparison.py --include-sequence-models

Output plots are saved to ``results/modulation/``.
"""

import argparse
import os
import sys
from time import perf_counter

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

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
    significance_table,
)
from relaynet.channels.fading import rayleigh_fading_channel

try:
    from relaynet.visualization.plots import plot_ber_with_ci
    _HAS_PLOTS = True
except Exception:
    _HAS_PLOTS = False

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper
    _HAS_SEQ = True
except Exception:
    _HAS_SEQ = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Relay comparison across BPSK / QPSK / 16-QAM modulations"
    )
    p.add_argument("--quick", action="store_true",
                   help="Reduced training and Monte Carlo effort.")
    p.add_argument("--include-sequence-models", action="store_true",
                   help="Include Transformer and Mamba relays.")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-timings", action="store_true")
    return p.parse_args()


def train_relays(args):
    """Train all relay models once (on BPSK AWGN data)."""
    print("\n=== Training relay models (BPSK data) ===")

    samples = 25_000 if not args.quick else 5_000
    epochs = 100 if not args.quick else 20
    vae_samples = 25_000 if not args.quick else 5_000
    cgan_samples = 10_000 if not args.quick else 5_000
    cgan_epochs = 50 if not args.quick else 20

    relays = {
        "AF": AmplifyAndForwardRelay(prefer_gpu=False),
        "DF": DecodeAndForwardRelay(prefer_gpu=False),
    }

    print("  GenAI …", end=" ", flush=True)
    t0 = perf_counter()
    genai = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False)
    genai.train(training_snrs=[5, 10, 15], num_samples=samples,
                epochs=epochs, seed=args.seed)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["GenAI (169p)"] = genai

    print("  Hybrid …", end=" ", flush=True)
    t0 = perf_counter()
    hybrid = HybridRelay(snr_threshold=5.0, prefer_gpu=False)
    hybrid.train(training_snrs=[2, 4, 6], num_samples=samples,
                 epochs=epochs, seed=args.seed)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["Hybrid"] = hybrid

    print("  VAE …", end=" ", flush=True)
    t0 = perf_counter()
    vae = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False)
    vae.train(training_snrs=[5, 10, 15], num_samples=vae_samples,
              epochs=epochs, seed=args.seed)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["VAE"] = vae

    print("  CGAN …", end=" ", flush=True)
    t0 = perf_counter()
    cgan = CGANRelay(window_size=7, noise_size=8, lambda_gp=10,
                     lambda_l1=20, n_critic=5, prefer_gpu=args.gpu)
    cgan.train(training_snrs=[5, 10, 15], num_samples=cgan_samples,
               epochs=cgan_epochs, seed=args.seed)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["CGAN (WGAN-GP)"] = cgan

    if args.include_sequence_models and _HAS_SEQ:
        seq_samples = 50_000 if not args.quick else 3_000
        seq_epochs = 100 if not args.quick else 10

        for name, cls in [("Transformer", TransformerRelayWrapper),
                          ("Mamba S6", MambaRelayWrapper),
                          ("Mamba2 (SSD)", Mamba2RelayWrapper)]:
            print(f"  {name} …", end=" ", flush=True)
            t0 = perf_counter()
            kw = dict(target_power=1.0, window_size=11, d_model=32,
                      num_layers=2, prefer_gpu=args.gpu)
            if "Mamba" in name:
                kw["d_state"] = 16
            else:
                kw["num_heads"] = 4
            r = cls(**kw)
            r.train(training_snrs=[5, 10, 15], num_samples=seq_samples,
                    epochs=seq_epochs, lr=0.001)
            print(f"done ({perf_counter()-t0:.1f}s)")
            relays[name] = r

    return relays


def run_comparison(relays, snr_range, args, modulation, channel_name, channel_fn=None):
    """Run BER comparison for a given modulation and channel."""
    print(f"\n=== {modulation.upper()} – {channel_name} ===")
    all_ber, all_trials = {}, {}

    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        t0 = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - t0
        print(f"done ({elapsed:.1f}s, BER [{ber.min():.2e}, {ber.max():.2e}])")

    # Significance
    if "DF" in all_ber:
        methods = [k for k in all_ber if k != "DF"]
        print(f"\n  Significance vs DF:")
        significance_table(
            snr_range, methods,
            {k: all_trials[k] for k in all_trials},
            baseline="DF",
        )

    return all_ber, all_trials


def plot_results(snr_range, all_ber, all_trials, modulation, channel_name,
                 out_dir="results/modulation"):
    """Generate and save BER plot."""
    if not _HAS_PLOTS:
        print("  [WARNING] matplotlib/plots not available — skipping plot.")
        return
    ci_dict = {}
    for name, trials in all_trials.items():
        lo, hi = compute_confidence_interval(trials)
        ci_dict[name] = (lo, hi)

    safe_mod = modulation.replace("16", "16_")
    safe_ch = channel_name.lower().replace(" ", "_").replace("×", "x")
    fname = f"{safe_mod}_{safe_ch}_ci.png"

    plot_ber_with_ci(
        snr_range, all_ber, ci_dict,
        title=f"{modulation.upper()} – {channel_name} – All Relay Methods (95% CI)",
        save_path=os.path.join(out_dir, fname),
    )


def print_summary_table(results, snr_range):
    """Print a summary comparison table at selected SNR points."""
    snr_pts = [0, 4, 10, 16]
    snr_list = list(np.asarray(snr_range).tolist())

    print("\n" + "=" * 80)
    print("SUMMARY: BER at selected SNR points across modulations")
    print("=" * 80)

    for snr_pt in snr_pts:
        if snr_pt not in snr_list:
            continue
        idx = snr_list.index(float(snr_pt))
        print(f"\n--- SNR = {snr_pt} dB ---")
        header = f"  {'Relay':<18}"
        for (mod, ch), (ber_dict, _) in results.items():
            header += f"  {mod.upper()}/{ch[:5]:>5}"
        print(header)

        relay_names = list(next(iter(results.values()))[0].keys())
        for rname in relay_names:
            row = f"  {rname:<18}"
            for (mod, ch), (ber_dict, _) in results.items():
                val = ber_dict[rname][idx]
                row += f"  {val:>10.4f}"
            print(row)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    total_start = perf_counter()

    if args.quick:
        args.bits_per_trial = min(args.bits_per_trial, 2_000)
        args.num_trials = min(args.num_trials, 3)

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)
    out_dir = "results/modulation"
    os.makedirs(out_dir, exist_ok=True)

    # Train once on BPSK
    relays = train_relays(args)

    # Configurations: (modulation, channel_name, channel_fn)
    configs = [
        ("bpsk",  "AWGN",     None),
        ("bpsk",  "Rayleigh", rayleigh_fading_channel),
        ("qpsk",  "AWGN",     None),
        ("qpsk",  "Rayleigh", rayleigh_fading_channel),
        ("qam16", "AWGN",     None),
        ("qam16", "Rayleigh", rayleigh_fading_channel),
    ]

    results = {}
    for mod, ch_name, ch_fn in configs:
        ber_dict, trial_dict = run_comparison(
            relays, snr_range, args, mod, ch_name, ch_fn
        )
        results[(mod, ch_name)] = (ber_dict, trial_dict)

        if not args.no_plots:
            plot_results(snr_range, ber_dict, trial_dict, mod, ch_name, out_dir)

    # Summary table
    print_summary_table(results, snr_range)

    elapsed = perf_counter() - total_start
    print(f"\nDone. Total time: {elapsed:.1f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
