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
from time import perf_counter

# Ensure UTF-8 output on Windows (avoids cp1252 crashes when redirecting)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    significance_table,
)
from relaynet.visualization.plots import plot_ber_curves, plot_ber_with_ci
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.channels.mimo import mimo_2x2_channel, mimo_2x2_mmse_channel

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    _HAS_SEQUENCE_MODELS = True
except Exception:
    TransformerRelayWrapper = None
    MambaRelayWrapper = None
    _HAS_SEQUENCE_MODELS = False


def parse_args():
    p = argparse.ArgumentParser(description="Full relay comparison")
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a much faster (lower-fidelity) comparison. "
            "Generates the same plots but uses fewer training samples/epochs "
            "and fewer Monte Carlo bits/trials."
        ),
    )
    p.add_argument(
        "--include-sequence-models",
        action="store_true",
        help="Include Transformer and Mamba relays from the checkpoint implementations.",
    )
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)

    # Training hyperparameters (defaults reproduce the PR's intended settings)
    p.add_argument("--genai-samples", type=int, default=25_000)
    p.add_argument("--genai-epochs", type=int, default=100)
    p.add_argument("--hybrid-samples", type=int, default=25_000)
    p.add_argument("--hybrid-epochs", type=int, default=100)
    p.add_argument("--vae-samples", type=int, default=50_000)
    p.add_argument("--vae-epochs", type=int, default=100)
    p.add_argument("--cgan-samples", type=int, default=50_000)
    p.add_argument("--cgan-epochs", type=int, default=200)
    p.add_argument("--transformer-samples", type=int, default=50_000)
    p.add_argument("--transformer-epochs", type=int, default=100)
    p.add_argument("--mamba-samples", type=int, default=50_000)
    p.add_argument("--mamba-epochs", type=int, default=100)

    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--log-timings",
        action="store_true",
        help="Print elapsed times and device selection for each training/evaluation stage.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _relay_device(relay):
    device = getattr(relay, "device", None)
    if device is None:
        return "cpu"
    return str(device)


def _timed(label, fn, *args, log_timings=False, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    if log_timings:
        print(f"    [time] {label}: {elapsed:.2f}s")
    return result, elapsed


def train_models(args):
    print("\n=== Training AI relay models ===")

    timing_summary = {}

    print("  Training Minimal GenAI relay (169 params) …")
    # Tiny models are typically faster on CPU than GPU due to kernel-launch overhead.
    genai = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False)
    print(f"    device: {_relay_device(genai)}")
    _, elapsed = _timed(
        "train GenAI (169p)",
        genai.train,
        training_snrs=[5, 10, 15],
        num_samples=args.genai_samples,
        epochs=args.genai_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
    )
    timing_summary["GenAI (169p)"] = (_relay_device(genai), elapsed)

    print("  Training Hybrid relay …")
    hybrid = HybridRelay(snr_threshold=5.0, prefer_gpu=False)
    print(f"    device: {_relay_device(hybrid)}")
    _, elapsed = _timed(
        "train Hybrid",
        hybrid.train,
        training_snrs=[2, 4, 6],
        num_samples=args.hybrid_samples,
        epochs=args.hybrid_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
    )
    timing_summary["Hybrid"] = (_relay_device(hybrid), elapsed)

    print("  Training VAE relay …")
    vae = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False)
    print(f"    device: {_relay_device(vae)}")
    _, elapsed = _timed(
        "train VAE",
        vae.train,
        training_snrs=[5, 10, 15],
        num_samples=args.vae_samples,
        epochs=args.vae_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
    )
    timing_summary["VAE"] = (_relay_device(vae), elapsed)

    print("  Training CGAN relay (WGAN-GP) …")
    # CGAN networks are ~3K params total — too small for GPU benefit.
    cgan = CGANRelay(
        window_size=7,
        noise_size=8,
        lambda_gp=10,
        lambda_l1=20,
        n_critic=5,
        prefer_gpu=False,
    )
    print(f"    device: {_relay_device(cgan)}")
    _, elapsed = _timed(
        "train CGAN (WGAN-GP)",
        cgan.train,
        training_snrs=[5, 10, 15],
        num_samples=args.cgan_samples,
        epochs=args.cgan_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
    )
    timing_summary["CGAN (WGAN-GP)"] = (_relay_device(cgan), elapsed)

    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": genai,
        "Hybrid": hybrid,
        "VAE": vae,
        "CGAN (WGAN-GP)": cgan,
    }

    if args.include_sequence_models:
        if not _HAS_SEQUENCE_MODELS:
            print("  [WARNING] Transformer/Mamba checkpoints could not be imported; skipping.")
        else:
            print("  Training Transformer relay …")
            transformer = TransformerRelayWrapper(
                target_power=1.0,
                window_size=11,
                d_model=32,
                num_heads=4,
                num_layers=2,
            )
            print(f"    device: {_relay_device(transformer)}")
            _, elapsed = _timed(
                "train Transformer",
                transformer.train,
                training_snrs=[5, 10, 15],
                num_samples=args.transformer_samples,
                epochs=args.transformer_epochs,
                lr=0.001,
                log_timings=args.log_timings,
            )
            timing_summary["Transformer"] = (_relay_device(transformer), elapsed)

            print("  Training Mamba S6 relay …")
            mamba = MambaRelayWrapper(
                target_power=1.0,
                window_size=11,
                d_model=32,
                d_state=16,
                num_layers=2,
            )
            print(f"    device: {_relay_device(mamba)}")
            _, elapsed = _timed(
                "train Mamba S6",
                mamba.train,
                training_snrs=[5, 10, 15],
                num_samples=args.mamba_samples,
                epochs=args.mamba_epochs,
                lr=0.001,
                log_timings=args.log_timings,
            )
            timing_summary["Mamba S6"] = (_relay_device(mamba), elapsed)

            relays["Transformer"] = transformer
            relays["Mamba S6"] = mamba

    if args.log_timings:
        print("\n=== Training Time Summary ===")
        for name, (device, elapsed) in timing_summary.items():
            print(f"  {name:<18} {device:<6} {elapsed:>8.2f}s")

    return relays


def _parameter_count(name, relay):
    if name in {"AF", "DF"}:
        return 0
    if hasattr(relay, "num_params"):
        return int(relay.num_params)
    if hasattr(relay, "model"):
        try:
            return int(sum(p.numel() for p in relay.model.parameters()))
        except Exception:
            return 0
    return 0


def plot_complexity_comparison(relays, awgn_ber, snr_range, save_path="results/complexity_comparison_all_relays.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping complexity plot.")
        return None

    params = []
    ber0 = []
    ber4 = []
    ber8 = []
    labels = []
    colors = []
    color_map = {
        "AF": "gray",
        "DF": "black",
        "GenAI (169p)": "magenta",
        "Hybrid": "red",
        "VAE": "cyan",
        "CGAN (WGAN-GP)": "orange",
        "Transformer": "green",
        "Mamba S6": "blue",
    }

    snr_list = list(np.asarray(snr_range).tolist())
    idx0 = snr_list.index(0.0) if 0.0 in snr_list else 0
    idx4 = snr_list.index(4.0) if 4.0 in snr_list else min(len(snr_list) - 1, 2)
    idx8 = snr_list.index(8.0) if 8.0 in snr_list else min(len(snr_list) - 1, 4)

    for name, relay in relays.items():
        labels.append(name)
        params.append(max(_parameter_count(name, relay), 1))
        ber_curve = np.asarray(awgn_ber[name])
        ber0.append(float(ber_curve[idx0]))
        ber4.append(float(ber_curve[idx4]))
        ber8.append(float(ber_curve[idx8]))
        colors.append(color_map.get(name, "purple"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for x, y, label, color in zip(params, ber0, labels, colors):
        ax1.scatter(x, y, s=180, c=color, edgecolors="black", linewidth=1.2)
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Number of Parameters (log scale)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("BER at 0 dB", fontsize=12, fontweight="bold")
    ax1.set_title("Complexity vs Low-SNR Performance", fontsize=13, fontweight="bold")

    for x, y, label, color in zip(ber0, ber8, labels, colors):
        ax2.scatter(x, y, s=180, c=color, edgecolors="black", linewidth=1.2)
        ax2.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.set_xlabel("BER at 0 dB", fontsize=12, fontweight="bold")
    ax2.set_ylabel("BER at 8 dB", fontsize=12, fontweight="bold")
    ax2.set_title("Low-SNR vs High-SNR Trade-off", fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {save_path}")
    return save_path


def run_awgn_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== AWGN Channel Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_fading_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== Rayleigh Fading Channel Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=rayleigh_fading_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_rician_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== Rician Fading Channel (K=3) Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0),
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_mimo_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== 2\u00d72 MIMO Rayleigh Channel (ZF) Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_mimo_mmse_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== 2\u00d72 MIMO Rayleigh Channel (MMSE) Comparison ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_mmse_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def print_significance(snr_range, all_ber, all_trials, baseline="DF"):
    methods = [k for k in all_ber if k != baseline]
    print(f"\n=== Statistical Significance vs {baseline} ===")
    trials_dict = {k: all_trials[k] for k in all_trials}
    significance_table(snr_range, methods, trials_dict, baseline=baseline)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    total_start = perf_counter()

    if args.quick:
        # Keep defaults for SNR sweep; reduce training and Monte Carlo effort.
        args.bits_per_trial = min(args.bits_per_trial, 2_000)
        args.num_trials = min(args.num_trials, 3)
        args.genai_samples = min(args.genai_samples, 5_000)
        args.hybrid_samples = min(args.hybrid_samples, 5_000)
        args.vae_samples = min(args.vae_samples, 5_000)
        args.cgan_samples = min(args.cgan_samples, 5_000)
        args.genai_epochs = min(args.genai_epochs, 20)
        args.hybrid_epochs = min(args.hybrid_epochs, 20)
        args.vae_epochs = min(args.vae_epochs, 20)
        args.cgan_epochs = min(args.cgan_epochs, 20)
        args.transformer_samples = min(args.transformer_samples, 3_000)
        args.mamba_samples = min(args.mamba_samples, 3_000)
        args.transformer_epochs = min(args.transformer_epochs, 10)
        args.mamba_epochs = min(args.mamba_epochs, 10)

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)
    os.makedirs("results", exist_ok=True)

    relays = train_models(args)

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
        plot_complexity_comparison(relays, awgn_ber, snr_range)

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
    # Rician fading comparison
    rician_ber, rician_trials = run_rician_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, rician_ber, rician_trials)

    if not args.no_plots:
        rician_ci = {}
        for name, trials in rician_trials.items():
            lo, hi = compute_confidence_interval(trials)
            rician_ci[name] = (lo, hi)
        plot_ber_with_ci(
            snr_range, rician_ber, rician_ci,
            title="Rician Fading Channel (K=3) \u2013 All Relay Methods (95% CI)",
            save_path="results/rician_comparison_ci.png",
        )

    # 2×2 MIMO comparison
    mimo_ber, mimo_trials = run_mimo_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, mimo_ber, mimo_trials)

    if not args.no_plots:
        mimo_ci = {}
        for name, trials in mimo_trials.items():
            lo, hi = compute_confidence_interval(trials)
            mimo_ci[name] = (lo, hi)
        plot_ber_with_ci(
            snr_range, mimo_ber, mimo_ci,
            title="2\u00d72 MIMO Rayleigh (ZF) \u2013 All Relay Methods (95% CI)",
            save_path="results/mimo_2x2_comparison_ci.png",
        )

    # 2×2 MIMO MMSE comparison
    mimo_mmse_ber, mimo_mmse_trials = run_mimo_mmse_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, mimo_mmse_ber, mimo_mmse_trials)

    if not args.no_plots:
        mimo_mmse_ci = {}
        for name, trials in mimo_mmse_trials.items():
            lo, hi = compute_confidence_interval(trials)
            mimo_mmse_ci[name] = (lo, hi)
        plot_ber_with_ci(
            snr_range, mimo_mmse_ber, mimo_mmse_ci,
            title="2\u00d72 MIMO Rayleigh (MMSE) \u2013 All Relay Methods (95% CI)",
            save_path="results/mimo_2x2_mmse_comparison_ci.png",
        )

    print("\nDone. Results saved to results/")
    if args.log_timings:
        print(f"Total elapsed time: {perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
