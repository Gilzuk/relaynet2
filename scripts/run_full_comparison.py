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
from relaynet.channels.mimo import mimo_2x2_channel, mimo_2x2_mmse_channel, mimo_2x2_sic_channel
from relaynet.utils.checkpoint_manager import CheckpointManager

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper
    _HAS_SEQUENCE_MODELS = True
except Exception:
    TransformerRelayWrapper = None
    MambaRelayWrapper = None
    Mamba2RelayWrapper = None
    _HAS_SEQUENCE_MODELS = False

try:
    from checkpoints.checkpoint_22_normalized_3k import build_all_3k
    _HAS_NORMALIZED = True
except Exception:
    build_all_3k = None
    _HAS_NORMALIZED = False


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
    p.add_argument(
        "--include-normalized",
        action="store_true",
        help="Run an extra apples-to-apples comparison with all AI models at ~3K params.",
    )
    p.add_argument(
        "--include-cgan",
        action="store_true",
        help=(
            "Include CGAN (WGAN-GP) in the normalized 3K comparison. "
            "Disabled by default due to ~12x training overhead."
        ),
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
    p.add_argument("--mamba2-samples", type=int, default=50_000)
    p.add_argument("--mamba2-epochs", type=int, default=100)

    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Use GPU (CUDA) for the larger AI models (Transformer, Mamba, "
            "and optionally VAE/CGAN). Small models (GenAI, Hybrid) stay "
            "on CPU because kernel-launch overhead exceeds any speed-up."
        ),
    )
    p.add_argument(
        "--log-timings",
        action="store_true",
        help="Print elapsed times and device selection for each training/evaluation stage.",
    )
    p.add_argument(
        "--save-weights",
        action="store_true",
        help=(
            "Save all trained model weights to <weights-dir>/seed_<seed>/ "
            "for later inference-only runs."
        ),
    )
    p.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Skip training entirely; load previously saved weights from "
            "<weights-dir>/seed_<seed>/ and run evaluation only. "
            "Much faster — useful for regenerating plots."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Smart train-or-load mode: load weights for relays that have "
            "a saved checkpoint and train only those that are missing. "
            "Saves all weights (including newly trained) at the end."
        ),
    )
    p.add_argument(
        "--weights-dir",
        type=str,
        default="trained_weights",
        help="Directory for saving/loading model weights (default: trained_weights).",
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


def create_relay_instances(args):
    """Create **untrained** relay instances (same architectures as train_models).

    Used by ``--inference-only`` mode to instantiate models before loading
    saved weights.
    """
    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": MinimalGenAIRelay(
            window_size=5, hidden_size=24, prefer_gpu=False,
        ),
        "Hybrid": HybridRelay(snr_threshold=5.0, prefer_gpu=False),
        "VAE": VAERelay(
            window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
        ),
        "CGAN (WGAN-GP)": CGANRelay(
            window_size=7, noise_size=8, lambda_gp=10, lambda_l1=20,
            n_critic=5, prefer_gpu=args.gpu,
        ),
    }
    if args.include_sequence_models:
        if _HAS_SEQUENCE_MODELS:
            relays["Transformer"] = TransformerRelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                num_heads=4, num_layers=2, prefer_gpu=args.gpu,
            )
            relays["Mamba S6"] = MambaRelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                d_state=16, num_layers=2, prefer_gpu=args.gpu,
            )
            relays["Mamba2 (SSD)"] = Mamba2RelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                d_state=16, num_layers=2, prefer_gpu=args.gpu,
            )
        else:
            print("  [WARNING] Transformer/Mamba checkpoints not available; skipping.")
    return relays


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
    cgan = CGANRelay(
        window_size=7,
        noise_size=8,
        lambda_gp=10,
        lambda_l1=20,
        n_critic=5,
        prefer_gpu=args.gpu,
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
                prefer_gpu=args.gpu,
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
                prefer_gpu=args.gpu,
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

            print("  Training Mamba-2 (SSD) relay …")
            mamba2 = Mamba2RelayWrapper(
                target_power=1.0,
                window_size=11,
                d_model=32,
                d_state=16,
                num_layers=2,
                prefer_gpu=args.gpu,
            )
            print(f"    device: {_relay_device(mamba2)}")
            _, elapsed = _timed(
                "train Mamba2 (SSD)",
                mamba2.train,
                training_snrs=[5, 10, 15],
                num_samples=args.mamba2_samples,
                epochs=args.mamba2_epochs,
                lr=0.001,
                log_timings=args.log_timings,
            )
            timing_summary["Mamba2 (SSD)"] = (_relay_device(mamba2), elapsed)

            relays["Transformer"] = transformer
            relays["Mamba S6"] = mamba
            relays["Mamba2 (SSD)"] = mamba2

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
        "Mamba2 (SSD)": "#e377c2",
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
    print("\n=== 2\u00d72 MIMO Topology – Rayleigh Fading – ZF Equalization ===")
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
    print("\n=== 2\u00d72 MIMO Topology – Rayleigh Fading – MMSE Equalization ===")
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


def run_mimo_sic_comparison(relays, snr_range, bits_per_trial, num_trials):
    print("\n=== 2\u00d72 MIMO Topology \u2013 Rayleigh Fading \u2013 SIC Equalization ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_sic_channel,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def train_normalized_models(args):
    """Train all six AI relay models at ~3K parameters."""
    print("\n" + "=" * 60)
    print("=== Training NORMALIZED (~3K params) relay models ===")
    print("=" * 60)

    timing = {}
    relays = build_all_3k(
        prefer_gpu=args.gpu,
        include_sequence_models=args.include_sequence_models,
        include_cgan=getattr(args, 'include_cgan', False),
    )

    # --- GenAI-3K ---
    r = relays["GenAI-3K"]
    print(f"  Training GenAI-3K ({r.num_params}p) \u2026")
    _, elapsed = _timed(
        f"train GenAI-3K ({r.num_params}p)", r.train,
        training_snrs=[5, 10, 15],
        num_samples=args.genai_samples, epochs=args.genai_epochs,
        seed=args.seed, log_timings=args.log_timings,
    )
    timing[f"GenAI-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)

    # --- Hybrid-3K ---
    r = relays["Hybrid-3K"]
    print(f"  Training Hybrid-3K ({r.num_params}p) \u2026")
    _, elapsed = _timed(
        f"train Hybrid-3K ({r.num_params}p)", r.train,
        training_snrs=[2, 4, 6],
        num_samples=args.hybrid_samples, epochs=args.hybrid_epochs,
        seed=args.seed, log_timings=args.log_timings,
    )
    timing[f"Hybrid-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)

    # --- VAE-3K ---
    r = relays["VAE-3K"]
    print(f"  Training VAE-3K ({r.num_params}p) \u2026")
    _, elapsed = _timed(
        f"train VAE-3K ({r.num_params}p)", r.train,
        training_snrs=[5, 10, 15],
        num_samples=args.vae_samples, epochs=args.vae_epochs,
        seed=args.seed, log_timings=args.log_timings,
    )
    timing[f"VAE-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)

    # --- CGAN-3K ---
    if "CGAN-3K" in relays:
        r = relays["CGAN-3K"]
        print(f"  Training CGAN-3K ({r.num_params}p) \u2026")
        _, elapsed = _timed(
            f"train CGAN-3K ({r.num_params}p)", r.train,
            training_snrs=[5, 10, 15],
            num_samples=args.cgan_samples, epochs=args.cgan_epochs,
            seed=args.seed, log_timings=args.log_timings,
        )
        timing[f"CGAN-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)

    # --- Transformer-3K ---
    if "Transformer-3K" in relays:
        r = relays["Transformer-3K"]
        print(f"  Training Transformer-3K ({r.num_params}p) \u2026")
        _, elapsed = _timed(
            f"train Transformer-3K ({r.num_params}p)", r.train,
            training_snrs=[5, 10, 15],
            num_samples=args.transformer_samples,
            epochs=args.transformer_epochs, lr=0.001,
            log_timings=args.log_timings,
        )
        timing[f"Transformer-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)

    # --- Mamba-3K ---
    if "Mamba-3K" in relays:
        r = relays["Mamba-3K"]
        print(f"  Training Mamba-3K ({r.num_params}p) \u2026")
        _, elapsed = _timed(
            f"train Mamba-3K ({r.num_params}p)", r.train,
            training_snrs=[5, 10, 15],
            num_samples=args.mamba_samples,
            epochs=args.mamba_epochs, lr=0.001,
            log_timings=args.log_timings,
        )
        timing[f"Mamba-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)
    # --- Mamba2-3K ---
    if "Mamba2-3K" in relays:
        r = relays["Mamba2-3K"]
        print(f"  Training Mamba2-3K ({r.num_params}p) …")
        _, elapsed = _timed(
            f"train Mamba2-3K ({r.num_params}p)", r.train,
            training_snrs=[5, 10, 15],
            num_samples=args.mamba2_samples,
            epochs=args.mamba2_epochs, lr=0.001,
            log_timings=args.log_timings,
        )
        timing[f"Mamba2-3K ({r.num_params}p)"] = (_relay_device(r), elapsed)
    if args.log_timings:
        print("\n=== Normalized Training Time Summary ===")
        for name, (device, elapsed) in timing.items():
            print(f"  {name:<28} {device:<6} {elapsed:>8.2f}s")

    return relays


# ======================================================================
# Smart train-or-load (--resume mode)
# ======================================================================

_NON_TRAINABLE = {"AF", "DF"}


def _training_configs(args):
    """Return a dict mapping relay display names to their training kwargs.

    This centralises every relay's hyper-parameters so that
    ``train_or_load_models`` can train only the missing ones without
    duplicating the per-model blocks from ``train_models()``.
    """
    configs = {
        "GenAI (169p)": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.genai_samples,
            epochs=args.genai_epochs,
            seed=args.seed,
        ),
        "Hybrid": dict(
            training_snrs=[2, 4, 6],
            num_samples=args.hybrid_samples,
            epochs=args.hybrid_epochs,
            seed=args.seed,
        ),
        "VAE": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.vae_samples,
            epochs=args.vae_epochs,
            seed=args.seed,
        ),
        "CGAN (WGAN-GP)": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.cgan_samples,
            epochs=args.cgan_epochs,
            seed=args.seed,
        ),
        "Transformer": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.transformer_samples,
            epochs=args.transformer_epochs,
            lr=0.001,
        ),
        "Mamba S6": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.mamba_samples,
            epochs=args.mamba_epochs,
            lr=0.001,
        ),
        "Mamba2 (SSD)": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.mamba2_samples,
            epochs=args.mamba2_epochs,
            lr=0.001,
        ),
    }
    return configs


def _norm_training_configs(args):
    """Training kwargs for the normalized 3K models."""
    configs = {
        "GenAI-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.genai_samples,
            epochs=args.genai_epochs,
            seed=args.seed,
        ),
        "Hybrid-3K": dict(
            training_snrs=[2, 4, 6],
            num_samples=args.hybrid_samples,
            epochs=args.hybrid_epochs,
            seed=args.seed,
        ),
        "VAE-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.vae_samples,
            epochs=args.vae_epochs,
            seed=args.seed,
        ),
        "CGAN-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.cgan_samples,
            epochs=args.cgan_epochs,
            seed=args.seed,
        ),
        "Transformer-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.transformer_samples,
            epochs=args.transformer_epochs,
            lr=0.001,
        ),
        "Mamba-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.mamba_samples,
            epochs=args.mamba_epochs,
            lr=0.001,
        ),
        "Mamba2-3K": dict(
            training_snrs=[5, 10, 15],
            num_samples=args.mamba2_samples,
            epochs=args.mamba2_epochs,
            lr=0.001,
        ),
    }
    return configs


def train_or_load_models(args, ckpt_mgr):
    """Resume mode: load weights where available, train only the missing ones.

    Returns the same ``relays`` dict as ``train_models``.
    """
    print("\n=== Resume mode: loading existing weights, training missing models ===")

    relays = create_relay_instances(args)
    configs = _training_configs(args)
    timing_summary = {}

    loaded, skipped = ckpt_mgr.load_all(relays, args.seed)
    loaded_set = set(loaded)

    if loaded:
        print(f"  Loaded from checkpoint: {', '.join(loaded)}")

    to_train = [
        name for name in relays
        if name not in loaded_set and name not in _NON_TRAINABLE
    ]
    if to_train:
        print(f"  Need to train: {', '.join(to_train)}")
    else:
        print("  All models loaded — no training needed.")

    for name in to_train:
        relay = relays[name]
        cfg = configs.get(name)
        if cfg is None:
            print(f"  [WARNING] No training config for '{name}'; skipping.")
            continue
        print(f"  Training {name} …")
        print(f"    device: {_relay_device(relay)}")
        _, elapsed = _timed(
            f"train {name}",
            relay.train,
            **cfg,
            log_timings=args.log_timings,
        )
        timing_summary[name] = (_relay_device(relay), elapsed)

    # Always save — this persists both previously existing and newly trained
    config_dict = {
        k: v for k, v in vars(args).items()
        if isinstance(v, (int, float, str, bool, list))
    }
    saved = ckpt_mgr.save_all(relays, args.seed, config_dict)
    ckpt_dir = ckpt_mgr.checkpoint_dir(args.seed)
    print(f"  Saved {len(saved)} relay(s) → {ckpt_dir}/")

    if timing_summary and args.log_timings:
        print("\n=== Training Time Summary (resume) ===")
        for name, (device, elapsed) in timing_summary.items():
            print(f"  {name:<18} {device:<6} {elapsed:>8.2f}s")

    return relays


def train_or_load_normalized(args, ckpt_mgr):
    """Resume mode for normalized 3K models."""
    print("\n" + "=" * 60)
    print("=== Resume: NORMALIZED (~3K params) relay models ===")
    print("=" * 60)

    relays = build_all_3k(
        prefer_gpu=args.gpu,
        include_sequence_models=args.include_sequence_models,
        include_cgan=getattr(args, 'include_cgan', False),
    )
    configs = _norm_training_configs(args)
    timing = {}

    loaded, _ = ckpt_mgr.load_all(relays, args.seed)
    loaded_set = set(loaded)

    if loaded:
        print(f"  Loaded from checkpoint: {', '.join(loaded)}")

    to_train = [name for name in relays if name not in loaded_set]
    if to_train:
        print(f"  Need to train: {', '.join(to_train)}")
    else:
        print("  All normalized models loaded — no training needed.")

    for name in to_train:
        relay = relays[name]
        cfg = configs.get(name)
        if cfg is None:
            print(f"  [WARNING] No training config for '{name}'; skipping.")
            continue
        print(f"  Training {name} ({relay.num_params}p) …")
        _, elapsed = _timed(
            f"train {name} ({relay.num_params}p)",
            relay.train,
            **cfg,
            log_timings=args.log_timings,
        )
        timing[f"{name} ({relay.num_params}p)"] = (_relay_device(relay), elapsed)

    ckpt_mgr.save_all(relays, args.seed)

    if timing and args.log_timings:
        print("\n=== Normalized Training Time Summary (resume) ===")
        for name, (device, elapsed) in timing.items():
            print(f"  {name:<28} {device:<6} {elapsed:>8.2f}s")

    return relays


def run_normalized_comparison(relays_3k, snr_range, bits_per_trial, num_trials,
                              no_plots=False):
    """Run all channels for the normalized ~3K-param models."""
    channels = [
        ("AWGN",        None,                    "normalized_awgn_3k_ci.png"),
        ("Rayleigh",    rayleigh_fading_channel,  "normalized_rayleigh_3k_ci.png"),
        ("Rician K=3",
         lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0),
         "normalized_rician_3k_ci.png"),
        ("2\u00d72 MIMO ZF",   mimo_2x2_channel,        "normalized_mimo_zf_3k_ci.png"),
        ("2\u00d72 MIMO MMSE", mimo_2x2_mmse_channel,   "normalized_mimo_mmse_3k_ci.png"),
        ("2\u00d72 MIMO SIC",  mimo_2x2_sic_channel,    "normalized_mimo_sic_3k_ci.png"),
    ]

    for ch_name, ch_fn, plot_file in channels:
        print(f"\n=== Normalized 3K: {ch_name} Channel ===")
        all_ber, all_trials = {}, {}
        for name, relay in relays_3k.items():
            print(f"  {name} \u2026", end=" ", flush=True)
            start = perf_counter()
            kw = dict(
                num_bits_per_trial=bits_per_trial,
                num_trials=num_trials,
            )
            if ch_fn is not None:
                kw["channel_fn"] = ch_fn
            _, ber, trials = run_monte_carlo(relay, snr_range, **kw)
            all_ber[name] = ber
            all_trials[name] = trials
            elapsed = perf_counter() - start
            print(f"done (time={elapsed:.2f}s, BER [{ber.min():.2e}, {ber.max():.2e}])")

        # Significance (use first model as baseline if no DF)
        baseline = list(all_ber.keys())[0]
        methods = [k for k in all_ber if k != baseline]
        if methods:
            print(f"\n=== Significance vs {baseline} ===")
            significance_table(
                snr_range, methods,
                {k: all_trials[k] for k in all_trials},
                baseline=baseline,
            )

        if not no_plots:
            ci_dict = {}
            for name, trials in all_trials.items():
                lo, hi = compute_confidence_interval(trials)
                ci_dict[name] = (lo, hi)
            plot_ber_with_ci(
                snr_range, all_ber, ci_dict,
                title=f"Normalized ~3K params \u2013 {ch_name} (95% CI)",
                save_path=f"results/{plot_file}",
            )


def print_significance(snr_range, all_ber, all_trials, baseline="DF"):
    methods = [k for k in all_ber if k != baseline]
    print(f"\n=== Statistical Significance vs {baseline} ===")
    trials_dict = {k: all_trials[k] for k in all_trials}
    significance_table(snr_range, methods, trials_dict, baseline=baseline)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Warm up CUDA to avoid lazy-init warnings from cuBLAS/cuDNN.
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                _tmp = torch.randn(2, 2, device="cuda", requires_grad=True)
                (_tmp @ _tmp).sum().backward()  # force cuBLAS + autograd init
                del _tmp
        except Exception:
            pass

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
        args.mamba2_samples = min(args.mamba2_samples, 3_000)
        args.transformer_epochs = min(args.transformer_epochs, 10)
        args.mamba_epochs = min(args.mamba_epochs, 10)
        args.mamba2_epochs = min(args.mamba2_epochs, 10)

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)
    os.makedirs("results", exist_ok=True)

    ckpt_mgr = CheckpointManager(args.weights_dir)

    if args.inference_only:
        if not ckpt_mgr.has_checkpoint(args.seed):
            print(f"\nERROR: No saved weights for seed={args.seed} "
                  f"in '{args.weights_dir}/'")
            print(f"  Available seeds: "
                  f"{ckpt_mgr.list_checkpoints() or 'none'}")
            print("  Run with --save-weights first to create a checkpoint.")
            sys.exit(1)
        print(f"\n=== Inference-only mode (seed={args.seed}) ===")
        meta = ckpt_mgr.get_metadata(args.seed)
        if meta:
            print(f"  Checkpoint from: {meta.get('date', 'unknown')}")
        relays = create_relay_instances(args)
        loaded, skipped = ckpt_mgr.load_all(relays, args.seed)
        print(f"  Loaded: {', '.join(loaded) if loaded else 'none'}")
        non_trainable = {"AF", "DF"}
        actually_skipped = [s for s in skipped if s not in non_trainable]
        if actually_skipped:
            print(f"  [WARNING] Missing weights for: "
                  f"{', '.join(actually_skipped)}")
    elif args.resume:
        # Smart train-or-load: load what exists, train the rest, save all
        relays = train_or_load_models(args, ckpt_mgr)
    else:
        relays = train_models(args)
        if args.save_weights:
            print(f"\n=== Saving weights (seed={args.seed}) ===")
            config = {
                k: v for k, v in vars(args).items()
                if isinstance(v, (int, float, str, bool, list))
            }
            saved = ckpt_mgr.save_all(relays, args.seed, config)
            ckpt_dir = ckpt_mgr.checkpoint_dir(args.seed)
            print(f"  Saved {len(saved)} relay(s) \u2192 {ckpt_dir}/")

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
            title="2\u00d72 MIMO (Rayleigh) \u2013 ZF Equalization \u2013 All Relay Methods (95% CI)",
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
            title="2\u00d72 MIMO (Rayleigh) \u2013 MMSE Equalization \u2013 All Relay Methods (95% CI)",
            save_path="results/mimo_2x2_mmse_comparison_ci.png",
        )

    # 2×2 MIMO SIC comparison
    mimo_sic_ber, mimo_sic_trials = run_mimo_sic_comparison(
        relays, snr_range, args.bits_per_trial, args.num_trials
    )
    print_significance(snr_range, mimo_sic_ber, mimo_sic_trials)

    if not args.no_plots:
        mimo_sic_ci = {}
        for name, trials in mimo_sic_trials.items():
            lo, hi = compute_confidence_interval(trials)
            mimo_sic_ci[name] = (lo, hi)
        plot_ber_with_ci(
            snr_range, mimo_sic_ber, mimo_sic_ci,
            title="2\u00d72 MIMO (Rayleigh) \u2013 SIC Equalization \u2013 All Relay Methods (95% CI)",
            save_path="results/mimo_2x2_sic_comparison_ci.png",
        )

    # ── Normalized (~3K params) apples-to-apples comparison ──────────
    if args.include_normalized:
        if not _HAS_NORMALIZED:
            print("\n  [WARNING] checkpoint_22_normalized_3k could not be imported; "
                  "skipping normalized comparison.")
        else:
            if args.inference_only:
                relays_3k = build_all_3k(
                    prefer_gpu=args.gpu,
                    include_sequence_models=args.include_sequence_models,
                    include_cgan=getattr(args, 'include_cgan', False),
                )
                loaded_3k, _ = ckpt_mgr.load_all(relays_3k, args.seed)
                print(f"\n  Loaded normalized models: "
                      f"{', '.join(loaded_3k) if loaded_3k else 'none'}")
            elif args.resume:
                relays_3k = train_or_load_normalized(args, ckpt_mgr)
            else:
                relays_3k = train_normalized_models(args)
                if args.save_weights:
                    ckpt_mgr.save_all(relays_3k, args.seed)
            run_normalized_comparison(
                relays_3k, snr_range,
                args.bits_per_trial, args.num_trials,
                no_plots=args.no_plots,
            )

    print("\nDone. Results saved to results/")
    if args.log_timings:
        print(f"Total elapsed time: {perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
