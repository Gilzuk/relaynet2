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
from relaynet.utils.activations import get_clip_range

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


# ── Activation comparison constants ──────────────────────────────────

_ACT_ACTIVATIONS = ("sigmoid", "hardtanh", "scaled_tanh")

_ACT_VARIANT_STYLE = {
    "sigmoid":           {"ls": "-",  "alpha": 0.9},
    "hardtanh":          {"ls": "--", "alpha": 0.9},
    "scaled_tanh":       {"ls": "-.", "alpha": 0.9},
    "sigmoid+LN":        {"ls": (0, (3, 1, 1, 1)),       "alpha": 0.9},
    "hardtanh+LN":       {"ls": (0, (5, 2)),              "alpha": 0.9},
    "scaled_tanh+LN":    {"ls": ":",                      "alpha": 0.9},
}

_ACT_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#e6beff", "#ffe119", "#fabebe", "#7f7f7f", "#a9a9a9",
    "#008080", "#e41a1c", "#377eb8", "#ff7f00", "#984ea3",
    "#00ced1", "#ff1493", "#228b22", "#8b4513", "#4b0082",
]

_ACT_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "<", ">",
    "h", "p", "*", "H", "d", "8", "+", "x", "1",
    "2", "3", "4", "|", "_",
]


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
    p.add_argument(
        "--constellations",
        type=str,
        nargs="+",
        default=["bpsk", "qpsk", "qam16"],
        help="List of constellations to test: bpsk, qpsk, qam16 (default: all)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip experiment stages whose output plot files already exist. "
            "Useful for resuming after a partial failure without re-running "
            "experiments that already completed successfully."
        ),
    )
    p.add_argument(
        "--layer-norm",
        action="store_true",
        help=(
            "Enable Input LayerNorm on sequence models (Transformer, Mamba S6, "
            "Mamba2). Weights are stored in a separate subdirectory and plots "
            "are saved under results/layernorm/."
        ),
    )
    p.add_argument(
        "--compare-activations",
        action="store_true",
        help=(
            "Run activation comparison study: compare sigmoid vs hardtanh vs "
            "scaled_tanh across all models on QPSK/QAM16.  CGAN is excluded "
            "by default (use --include-cgan to add it).  Sequence models are "
            "always included.  Runs INSTEAD of the normal comparison."
        ),
    )
    p.add_argument(
        "--compare-layernorm",
        action="store_true",
        help=(
            "When used with --compare-activations, also compare with/without "
            "Input LayerNorm on sequence models (adds +LN variants: 6 total)."
        ),
    )
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
    _act = getattr(args, 'output_activation', 'tanh')
    _cr = getattr(args, 'clip_range', None)
    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": MinimalGenAIRelay(
            window_size=5, hidden_size=24, prefer_gpu=False,
            output_activation=_act, clip_range=_cr,
        ),
        "Hybrid": HybridRelay(snr_threshold=5.0, prefer_gpu=False,
                               output_activation=_act, clip_range=_cr),
        "VAE": VAERelay(
            window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
            output_activation=_act, clip_range=_cr,
        ),
        "CGAN (WGAN-GP)": CGANRelay(
            window_size=7, noise_size=8, lambda_gp=10, lambda_l1=20,
            n_critic=5, prefer_gpu=args.gpu,
            output_activation=_act, clip_range=_cr,
        ),
    }
    if args.include_sequence_models:
        if _HAS_SEQUENCE_MODELS:
            _ln = getattr(args, 'layer_norm', False)
            relays["Transformer"] = TransformerRelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                num_heads=4, num_layers=2, prefer_gpu=args.gpu,
                use_input_norm=_ln, output_activation=_act, clip_range=_cr,
            )
            relays["Mamba S6"] = MambaRelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                d_state=16, num_layers=2, prefer_gpu=args.gpu,
                use_input_norm=_ln, output_activation=_act, clip_range=_cr,
            )
            relays["Mamba2 (SSD)"] = Mamba2RelayWrapper(
                target_power=1.0, window_size=11, d_model=32,
                d_state=16, num_layers=2, prefer_gpu=args.gpu,
                use_input_norm=_ln, output_activation=_act, clip_range=_cr,
            )
        else:
            print("  [WARNING] Transformer/Mamba checkpoints not available; skipping.")
    return relays


def train_models(args):
    print("\n=== Training AI relay models ===")

    _act = getattr(args, 'output_activation', 'tanh')
    _tmod = getattr(args, 'training_modulation', 'bpsk')
    timing_summary = {}

    print("  Training Minimal GenAI relay (169 params) …")
    # Tiny models are typically faster on CPU than GPU due to kernel-launch overhead.
    genai = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False,
                              output_activation=_act)
    print(f"    device: {_relay_device(genai)}")
    _, elapsed = _timed(
        "train GenAI (169p)",
        genai.train,
        training_snrs=[5, 10, 15],
        num_samples=args.genai_samples,
        epochs=args.genai_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
        training_modulation=_tmod,
    )
    timing_summary["GenAI (169p)"] = (_relay_device(genai), elapsed)

    print("  Training Hybrid relay …")
    hybrid = HybridRelay(snr_threshold=5.0, prefer_gpu=False,
                         output_activation=_act, clip_range=_cr)
    print(f"    device: {_relay_device(hybrid)}")
    _, elapsed = _timed(
        "train Hybrid",
        hybrid.train,
        training_snrs=[2, 4, 6],
        num_samples=args.hybrid_samples,
        epochs=args.hybrid_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
        training_modulation=_tmod,
    )
    timing_summary["Hybrid"] = (_relay_device(hybrid), elapsed)

    print("  Training VAE relay …")
    vae = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
                   output_activation=_act, clip_range=_cr)
    print(f"    device: {_relay_device(vae)}")
    _, elapsed = _timed(
        "train VAE",
        vae.train,
        training_snrs=[5, 10, 15],
        num_samples=args.vae_samples,
        epochs=args.vae_epochs,
        seed=args.seed,
        log_timings=args.log_timings,
        training_modulation=_tmod,
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
        output_activation=_act,
        clip_range=_cr,
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
        training_modulation=_tmod,
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
            _ln = getattr(args, 'layer_norm', False)
            print("  Training Transformer relay …")
            transformer = TransformerRelayWrapper(
                target_power=1.0,
                window_size=11,
                d_model=32,
                num_heads=4,
                num_layers=2,
                prefer_gpu=args.gpu,
                use_input_norm=_ln,
                output_activation=_act,
                clip_range=_cr,
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
                training_modulation=_tmod,
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
                use_input_norm=_ln,
                output_activation=_act,
                clip_range=_cr,
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
                training_modulation=_tmod,
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
                use_input_norm=_ln,
                output_activation=_act,
                clip_range=_cr,
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
                training_modulation=_tmod,
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


def run_awgn_comparison(relays, snr_range, bits_per_trial, num_trials,
                       modulation="bpsk"):
    print(f"\n=== AWGN Channel Comparison [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_fading_comparison(relays, snr_range, bits_per_trial, num_trials,
                         modulation="bpsk"):
    print(f"\n=== Rayleigh Fading Channel Comparison [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} …", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=rayleigh_fading_channel,
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_rician_comparison(relays, snr_range, bits_per_trial, num_trials,
                         modulation="bpsk"):
    print(f"\n=== Rician Fading Channel (K=3) Comparison [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0),
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_mimo_comparison(relays, snr_range, bits_per_trial, num_trials,
                       modulation="bpsk"):
    print(f"\n=== 2\u00d72 MIMO Topology – Rayleigh Fading – ZF Equalization [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_channel,
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_mimo_mmse_comparison(relays, snr_range, bits_per_trial, num_trials,
                            modulation="bpsk"):
    print(f"\n=== 2\u00d72 MIMO Topology – Rayleigh Fading – MMSE Equalization [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_mmse_channel,
            modulation=modulation,
        )
        all_ber[name] = ber
        all_trials[name] = trials
        elapsed = perf_counter() - start
        print(f"done (device={_relay_device(relay)}, time={elapsed:.2f}s, mean BER range [{ber.min():.2e}, {ber.max():.2e}])")
    return all_ber, all_trials


def run_mimo_sic_comparison(relays, snr_range, bits_per_trial, num_trials,
                           modulation="bpsk"):
    print(f"\n=== 2\u00d72 MIMO Topology – Rayleigh Fading – SIC Equalization [{modulation.upper()}] ===")
    all_ber, all_trials = {}, {}
    for name, relay in relays.items():
        print(f"  {name} \u2026", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=mimo_2x2_sic_channel,
            modulation=modulation,
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
    _ln = getattr(args, 'layer_norm', False)
    _act = getattr(args, 'output_activation', 'tanh')
    _tmod = getattr(args, 'training_modulation', 'bpsk')
    _cr = getattr(args, 'clip_range', None)
    relays = build_all_3k(
        prefer_gpu=args.gpu,
        include_sequence_models=args.include_sequence_models,
        include_cgan=getattr(args, 'include_cgan', False),
        use_input_norm=_ln,
        output_activation=_act,
        clip_range=_cr,
    )

    # --- GenAI-3K ---
    r = relays["GenAI-3K"]
    print(f"  Training GenAI-3K ({r.num_params}p) \u2026")
    _, elapsed = _timed(
        f"train GenAI-3K ({r.num_params}p)", r.train,
        training_snrs=[5, 10, 15],
        num_samples=args.genai_samples, epochs=args.genai_epochs,
        seed=args.seed, log_timings=args.log_timings,
        training_modulation=_tmod,
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
        training_modulation=_tmod,
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
        training_modulation=_tmod,
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
            training_modulation=_tmod,
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
            training_modulation=_tmod,
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
            training_modulation=_tmod,
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
            training_modulation=_tmod,
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
    _tmod = getattr(args, 'training_modulation', 'bpsk')
    for cfg in configs.values():
        cfg['training_modulation'] = _tmod
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
    _tmod = getattr(args, 'training_modulation', 'bpsk')
    for cfg in configs.values():
        cfg['training_modulation'] = _tmod
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

    _ln = getattr(args, 'layer_norm', False)
    _act = getattr(args, 'output_activation', 'tanh')
    relays = build_all_3k(
        prefer_gpu=args.gpu,
        include_sequence_models=args.include_sequence_models,
        include_cgan=getattr(args, 'include_cgan', False),
        use_input_norm=_ln,
        output_activation=_act,
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
                              no_plots=False, modulation="bpsk",
                              skip_existing=False, results_dir="results"):
    """Run all channels for the normalized ~3K-param models."""
    channels = [
        ("AWGN",        None,                    f"normalized_awgn_3k_ci_{modulation}.png"),
        ("Rayleigh",    rayleigh_fading_channel,  f"normalized_rayleigh_3k_ci_{modulation}.png"),
        ("Rician K=3",
         lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0),
         f"normalized_rician_3k_ci_{modulation}.png"),
        ("2\u00d72 MIMO ZF",   mimo_2x2_channel,        f"normalized_mimo_zf_3k_ci_{modulation}.png"),
        ("2\u00d72 MIMO MMSE", mimo_2x2_mmse_channel,   f"normalized_mimo_mmse_3k_ci_{modulation}.png"),
        ("2\u00d72 MIMO SIC",  mimo_2x2_sic_channel,    f"normalized_mimo_sic_3k_ci_{modulation}.png"),
    ]

    for ch_name, ch_fn, plot_file in channels:
        save_path = f"{results_dir}/{plot_file}"
        if skip_existing and os.path.isfile(save_path):
            print(f"\n  [SKIP] Normalized 3K: {ch_name} [{modulation.upper()}] — {save_path} exists")
            continue
        print(f"\n=== Normalized 3K: {ch_name} Channel [{modulation.upper()}] ===")
        all_ber, all_trials = {}, {}
        for name, relay in relays_3k.items():
            print(f"  {name} \u2026", end=" ", flush=True)
            start = perf_counter()
            kw = dict(
                num_bits_per_trial=bits_per_trial,
                num_trials=num_trials,
                modulation=modulation,
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
                title=f"Normalized ~3K params \u2013 {ch_name} (95% CI) [{modulation.upper()}]",
                save_path=save_path,
            )


# ======================================================================
# Activation comparison  (--compare-activations)
# ======================================================================

def _act_train_models(args, activation, layer_norm=False, constellation="qam16"):
    """Train all relay models with a specific activation for the comparison."""
    _tmod = constellation if constellation in ("qam16", "qpsk") else "bpsk"
    _cr = get_clip_range(constellation)
    relays = {}

    print(f"    GenAI …", end=" ", flush=True)
    g = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False,
                          output_activation=activation, clip_range=_cr)
    _, elapsed = _timed("train GenAI", g.train,
                        training_snrs=[5, 10, 15],
                        num_samples=args.genai_samples,
                        epochs=args.genai_epochs,
                        seed=args.seed, training_modulation=_tmod,
                        log_timings=args.log_timings)
    print(f"done ({elapsed:.1f}s)")
    relays["GenAI"] = g

    print(f"    Hybrid …", end=" ", flush=True)
    h = HybridRelay(snr_threshold=5.0, prefer_gpu=False,
                    output_activation=activation, clip_range=_cr)
    _, elapsed = _timed("train Hybrid", h.train,
                        training_snrs=[2, 4, 6],
                        num_samples=args.hybrid_samples,
                        epochs=args.hybrid_epochs,
                        seed=args.seed, training_modulation=_tmod,
                        log_timings=args.log_timings)
    print(f"done ({elapsed:.1f}s)")
    relays["Hybrid"] = h

    print(f"    VAE …", end=" ", flush=True)
    v = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
                 output_activation=activation, clip_range=_cr)
    _, elapsed = _timed("train VAE", v.train,
                        training_snrs=[5, 10, 15],
                        num_samples=args.vae_samples,
                        epochs=args.vae_epochs,
                        seed=args.seed, training_modulation=_tmod,
                        log_timings=args.log_timings)
    print(f"done ({elapsed:.1f}s)")
    relays["VAE"] = v

    if getattr(args, "include_cgan", False):
        print(f"    CGAN …", end=" ", flush=True)
        c = CGANRelay(window_size=7, noise_size=8, lambda_gp=10,
                      lambda_l1=20, n_critic=5, prefer_gpu=args.gpu,
                      output_activation=activation, clip_range=_cr)
        _, elapsed = _timed("train CGAN", c.train,
                            training_snrs=[5, 10, 15],
                            num_samples=args.cgan_samples,
                            epochs=args.cgan_epochs,
                            seed=args.seed, training_modulation=_tmod,
                            log_timings=args.log_timings)
        print(f"done ({elapsed:.1f}s)")
        relays["CGAN"] = c

    if _HAS_SEQUENCE_MODELS:
        for name, cls in [("Transformer", TransformerRelayWrapper),
                          ("Mamba S6", MambaRelayWrapper),
                          ("Mamba2", Mamba2RelayWrapper)]:
            print(f"    {name} …", end=" ", flush=True)
            kw = dict(target_power=1.0, window_size=11, d_model=32,
                      num_layers=2, prefer_gpu=args.gpu,
                      output_activation=activation,
                      use_input_norm=layer_norm,
                      clip_range=_cr)
            if "Mamba" in name:
                kw["d_state"] = 16
            else:
                kw["num_heads"] = 4
            r = cls(**kw)
            _, elapsed = _timed(
                f"train {name}", r.train,
                training_snrs=[5, 10, 15],
                num_samples=args.transformer_samples if name == "Transformer"
                else (args.mamba2_samples if name == "Mamba2"
                      else args.mamba_samples),
                epochs=args.transformer_epochs if name == "Transformer"
                else (args.mamba2_epochs if name == "Mamba2"
                      else args.mamba_epochs),
                lr=0.001, training_modulation=_tmod,
                log_timings=args.log_timings)
            print(f"done ({elapsed:.1f}s)")
            relays[name] = r

    relays["AF"] = AmplifyAndForwardRelay()
    relays["DF"] = DecodeAndForwardRelay()
    return relays


def _act_train_normalized(args, activation, layer_norm=False, constellation="qam16"):
    """Train normalized ~3K-param models with a specific activation."""
    if not _HAS_NORMALIZED:
        print("  [WARNING] Normalized 3K checkpoint not available; skipping.")
        return {}

    _tmod = constellation if constellation in ("qam16", "qpsk") else "bpsk"
    _cr = get_clip_range(constellation)
    relays = build_all_3k(
        prefer_gpu=args.gpu,
        include_sequence_models=_HAS_SEQUENCE_MODELS,
        include_cgan=getattr(args, "include_cgan", False),
        use_input_norm=layer_norm,
        output_activation=activation,
        clip_range=_cr,
    )

    feed_cfg = dict(training_snrs=[5, 10, 15],
                    num_samples=args.genai_samples, epochs=args.genai_epochs,
                    seed=args.seed, training_modulation=_tmod)
    hybrid_cfg = dict(training_snrs=[2, 4, 6],
                      num_samples=args.hybrid_samples, epochs=args.hybrid_epochs,
                      seed=args.seed, training_modulation=_tmod)
    seq_cfg = dict(training_snrs=[5, 10, 15],
                   num_samples=args.transformer_samples,
                   epochs=args.transformer_epochs,
                   lr=0.001, training_modulation=_tmod)

    for name, relay in relays.items():
        print(f"    {name} ({relay.num_params}p) …", end=" ", flush=True)
        if "Hybrid" in name:
            _, elapsed = _timed(f"train {name}", relay.train,
                                log_timings=args.log_timings, **hybrid_cfg)
        elif any(s in name for s in ("Transformer", "Mamba")):
            _, elapsed = _timed(f"train {name}", relay.train,
                                log_timings=args.log_timings, **seq_cfg)
        else:
            _, elapsed = _timed(f"train {name}", relay.train,
                                log_timings=args.log_timings, **feed_cfg)
        print(f"done ({elapsed:.1f}s)")

    return relays


def _act_evaluate(relays, snr_range, args, channel_name, modulation,
                  channel_fn=None):
    """Evaluate relays and return {name: (ber, trials)}."""
    results = {}
    for name, relay in relays.items():
        print(f"    {name} …", end=" ", flush=True)
        start = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            modulation=modulation,
        )
        elapsed = perf_counter() - start
        print(f"done ({elapsed:.1f}s, BER [{ber.min():.2e}, {ber.max():.2e}])")
        results[name] = (ber, trials)
    return results


def plot_activation_comparison(all_results, snr_range, constellation,
                               channel_name, out_dir, suffix=""):
    """Overlay plot with unique colour+marker per curve and zoom inset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(14, 9))

    curve_list = []
    baselines = []
    first_variant = list(all_results.keys())[0]
    for bl in ("AF", "DF"):
        if bl in all_results[first_variant]:
            ber, _ = all_results[first_variant][bl]
            baselines.append((bl, ber))
    for variant, results in all_results.items():
        for relay_name, (ber, _) in results.items():
            if relay_name in ("AF", "DF"):
                continue
            curve_list.append((variant, relay_name, ber))

    baseline_styles = {
        "AF":  {"color": "#999999", "marker": "o", "ls": "-"},
        "DF":  {"color": "#333333", "marker": "s", "ls": "-"},
    }
    for name, ber in baselines:
        st = baseline_styles[name]
        ax.semilogy(snr_range, np.clip(ber, 1e-5, 1),
                    ls=st["ls"], color=st["color"], lw=1.0,
                    marker=st["marker"], markevery=2, markersize=5,
                    label=name, alpha=0.8)

    ci = 0
    for variant, relay_name, ber in curve_list:
        style = _ACT_VARIANT_STYLE.get(variant, {"ls": "-", "alpha": 0.9})
        color = _ACT_PALETTE[ci % len(_ACT_PALETTE)]
        marker = _ACT_MARKERS[ci % len(_ACT_MARKERS)]
        ci += 1
        ax.semilogy(snr_range, np.clip(ber, 1e-5, 1),
                    ls=style["ls"], color=color, lw=1.0,
                    alpha=style["alpha"],
                    marker=marker, markevery=2, markersize=5,
                    label=f"{relay_name} [{variant}]")

    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title(
        f"Activation Comparison \u2014 {channel_name} [{constellation.upper()}]{suffix}",
        fontsize=13)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_ylim(1e-4, 1)

    snr_arr = np.asarray(snr_range)
    zoom_lo, zoom_hi = 10.0, 14.0
    mask = (snr_arr >= zoom_lo) & (snr_arr <= zoom_hi)
    if mask.sum() >= 2:
        axins = inset_axes(ax, width="35%", height="30%", loc="center right",
                           borderpad=2)
        for name, ber in baselines:
            st = baseline_styles[name]
            axins.semilogy(snr_arr[mask], np.clip(ber[mask], 1e-5, 1),
                           ls=st["ls"], color=st["color"], lw=1.0,
                           marker=st["marker"], markersize=4, alpha=0.8)
        ci2 = 0
        for variant, relay_name, ber in curve_list:
            style = _ACT_VARIANT_STYLE.get(variant, {"ls": "-", "alpha": 0.9})
            color = _ACT_PALETTE[ci2 % len(_ACT_PALETTE)]
            marker = _ACT_MARKERS[ci2 % len(_ACT_MARKERS)]
            ci2 += 1
            axins.semilogy(snr_arr[mask], np.clip(ber[mask], 1e-5, 1),
                           ls=style["ls"], color=color, lw=1.0,
                           alpha=style["alpha"], marker=marker, markersize=4)
        axins.set_xlim(zoom_lo, zoom_hi)
        axins.grid(True, which="both", alpha=0.2, linewidth=0.4)
        axins.set_title("Zoom 10\u201314 dB", fontsize=8)
        axins.tick_params(labelsize=7)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",
                   lw=0.6, ls="--")

    ax.legend(fontsize=7, ncol=2,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, framealpha=0.9)

    safe_ch = channel_name.lower().replace(" ", "_")
    tag = suffix.lower().replace(" ", "_").replace("[", "").replace("]", "")
    fname = os.path.join(out_dir,
                         f"{constellation}_activation_{safe_ch}{tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved \u2192 {fname}")


def print_activation_summary(all_results, snr_range, constellation,
                             channel_name, variants):
    """Print BER table at 4 dB / 16 dB with delta between activations."""
    snr_list = list(np.asarray(snr_range).tolist())
    idx_hi = snr_list.index(16.0) if 16.0 in snr_list else len(snr_list) - 1
    idx_lo = snr_list.index(4.0) if 4.0 in snr_list else min(2, len(snr_list) - 1)
    _dash = "\u2014"

    print(f"\n{'='*90}")
    print(f"  {constellation.upper()} BER \u2014 {channel_name}")
    print(f"{'='*90}")
    header = f"  {'Relay':<18}"
    for v in variants:
        header += f" {v+' @4dB':>16} {v+' @16dB':>16}"
    print(header)
    print(f"  {'-'*18}" + f" {'-'*16} {'-'*16}" * len(variants))

    all_names = []
    for var in all_results.values():
        for n in var:
            if n not in all_names:
                all_names.append(n)

    for name in all_names:
        row = f"  {name:<18}"
        for v in variants:
            if v in all_results and name in all_results[v]:
                ber, _ = all_results[v][name]
                row += f" {ber[idx_lo]:>16.6f} {ber[idx_hi]:>16.6f}"
            else:
                row += f" {_dash:>16} {_dash:>16}"
        print(row)

    if "sigmoid" in all_results and "hardtanh" in all_results:
        print(f"\n  \u0394 BER (sigmoid \u2212 hardtanh):")
        for name in all_names:
            if name in ("AF", "DF"):
                continue
            if name in all_results["sigmoid"] and name in all_results["hardtanh"]:
                sig_ber, _ = all_results["sigmoid"][name]
                ht_ber, _ = all_results["hardtanh"][name]
                d_lo = sig_ber[idx_lo] - ht_ber[idx_lo]
                d_hi = sig_ber[idx_hi] - ht_ber[idx_hi]
                tag_lo = "sigmoid \u2193" if d_lo < 0 else "hardtanh \u2193"
                tag_hi = "sigmoid \u2193" if d_hi < 0 else "hardtanh \u2193"
                print(f"    {name:<18} @4dB: {d_lo:+.6f} ({tag_lo})  "
                      f"@16dB: {d_hi:+.6f} ({tag_hi})")

    if "sigmoid+LN" in all_results and "hardtanh+LN" in all_results:
        print(f"\n  \u0394 BER (sigmoid+LN \u2212 hardtanh+LN):")
        for name in all_names:
            if name in ("AF", "DF"):
                continue
            if name in all_results["sigmoid+LN"] and name in all_results["hardtanh+LN"]:
                sig_ber, _ = all_results["sigmoid+LN"][name]
                ht_ber, _ = all_results["hardtanh+LN"][name]
                d_lo = sig_ber[idx_lo] - ht_ber[idx_lo]
                d_hi = sig_ber[idx_hi] - ht_ber[idx_hi]
                tag_lo = "sigmoid \u2193" if d_lo < 0 else "hardtanh \u2193"
                tag_hi = "sigmoid \u2193" if d_hi < 0 else "hardtanh \u2193"
                print(f"    {name:<18} @4dB: {d_lo:+.6f} ({tag_lo})  "
                      f"@16dB: {d_hi:+.6f} ({tag_hi})")
    print()


def run_activation_comparison(args, snr_range):
    """Run the full activation comparison study.

    Compares sigmoid vs hardtanh vs scaled_tanh across all relay models
    on QPSK / QAM16 (AWGN + Rayleigh), optionally with LayerNorm variants.
    """
    out_dir = os.path.join("results", "activation_comparison")
    os.makedirs(out_dir, exist_ok=True)

    variants = list(_ACT_ACTIVATIONS)
    if getattr(args, "compare_layernorm", False):
        variants += [f"{a}+LN" for a in _ACT_ACTIVATIONS]

    for constellation in args.constellations:
        print(f"\n{'#'*70}")
        print(f"#  Activation Comparison \u2014 {constellation.upper()}")
        print(f"{'#'*70}")

        results_awgn = {}
        results_rayleigh = {}

        for variant in variants:
            if variant.endswith("+LN"):
                act = variant[:-3]
                use_ln = True
            else:
                act = variant
                use_ln = False

            print(f"\n{'='*60}")
            print(f"  Variant: {variant} \u2014 {act} activation"
                  f"{' + Input LayerNorm' if use_ln else ''}")
            print(f"{'='*60}")

            print(f"\n  Training models [{variant}] \u2026")
            relays = _act_train_models(args, act, layer_norm=use_ln,
                                       constellation=constellation)

            print(f"\n  Evaluating {constellation.upper()} AWGN [{variant}] \u2026")
            results_awgn[variant] = _act_evaluate(
                relays, snr_range, args, "AWGN", constellation)

            print(f"\n  Evaluating {constellation.upper()} Rayleigh [{variant}] \u2026")
            results_rayleigh[variant] = _act_evaluate(
                relays, snr_range, args, "Rayleigh", constellation,
                channel_fn=rayleigh_fading_channel)

        print_activation_summary(results_awgn, snr_range, constellation,
                                 "AWGN", variants)
        print_activation_summary(results_rayleigh, snr_range, constellation,
                                 "Rayleigh", variants)

        plot_activation_comparison(results_awgn, snr_range, constellation,
                                   "AWGN", out_dir)
        plot_activation_comparison(results_rayleigh, snr_range, constellation,
                                   "Rayleigh", out_dir)

        # Optional: normalized 3K comparison
        if args.include_normalized:
            print(f"\n{'='*60}")
            print(f"  Normalized ~3K-param activation comparison "
                  f"[{constellation.upper()}]")
            print(f"{'='*60}")

            norm_awgn = {}
            norm_rayleigh = {}

            for variant in variants:
                if variant.endswith("+LN"):
                    act, use_ln = variant[:-3], True
                else:
                    act, use_ln = variant, False

                print(f"\n  Training normalized 3K [{variant}] \u2026")
                relays_3k = _act_train_normalized(args, act, layer_norm=use_ln,
                                                  constellation=constellation)
                if not relays_3k:
                    continue

                print(f"\n  Evaluating normalized 3K "
                      f"{constellation.upper()} AWGN [{variant}] \u2026")
                norm_awgn[variant] = _act_evaluate(
                    relays_3k, snr_range, args, "AWGN", constellation)

                print(f"\n  Evaluating normalized 3K "
                      f"{constellation.upper()} Rayleigh [{variant}] \u2026")
                norm_rayleigh[variant] = _act_evaluate(
                    relays_3k, snr_range, args, "Rayleigh", constellation,
                    channel_fn=rayleigh_fading_channel)

            if norm_awgn:
                print_activation_summary(norm_awgn, snr_range, constellation,
                                         "AWGN (3K)", variants)
                print_activation_summary(norm_rayleigh, snr_range,
                                         constellation, "Rayleigh (3K)",
                                         variants)
                plot_activation_comparison(norm_awgn, snr_range, constellation,
                                           "AWGN", out_dir, "_3k")
                plot_activation_comparison(norm_rayleigh, snr_range,
                                           constellation, "Rayleigh", out_dir,
                                           "_3k")


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

    # ── Activation comparison mode ──────────────────────────────────
    if getattr(args, "compare_activations", False):
        run_activation_comparison(args, snr_range)
        elapsed = perf_counter() - total_start
        print(f"\n  Total time: {elapsed:.1f}s")
        print("  Done!")
        return

    # ── Layer-norm mode: separate weights & results directories ──────
    if args.layer_norm:
        if args.weights_dir == "trained_weights":
            args.weights_dir = "trained_weights_layernorm"
        results_dir = "results/layernorm"
        ln_tag = " [+LayerNorm]"
    else:
        results_dir = "results"
        ln_tag = ""

    os.makedirs(results_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(args.weights_dir)

    for constellation in args.constellations:
        # ── QAM16 activation study (§7.11): hardtanh + QAM16 training ──
        if constellation == "qam16":
            args.output_activation = "hardtanh"
            args.training_modulation = "qam16"
            args.clip_range = get_clip_range("qam16")
            _ckpt = CheckpointManager(args.weights_dir + "_qam16")
        else:
            args.output_activation = "tanh"
            args.training_modulation = "bpsk"
            args.clip_range = None  # tanh ignores clip_range
            _ckpt = ckpt_mgr

        print(f"\n=== Running experiments for constellation: {constellation.upper()} ===")
        if constellation == "qam16":
            print(f"  \u2192 Using hardtanh activation + QAM16 training data (activation study \u00a77.11)")

        # Training/loading relays
        if args.inference_only:
            if not _ckpt.has_checkpoint(args.seed):
                print(f"\nERROR: No saved weights for seed={args.seed} "
                      f"in '{args.weights_dir}/'")
                print(f"  Available seeds: "
                      f"{_ckpt.list_checkpoints() or 'none'}")
                print("  Run with --save-weights first to create a checkpoint.")
                sys.exit(1)
            print(f"\n=== Inference-only mode (seed={args.seed}) ===")
            meta = _ckpt.get_metadata(args.seed)
            if meta:
                print(f"  Checkpoint from: {meta.get('date', 'unknown')}")
            relays = create_relay_instances(args)
            loaded, skipped = _ckpt.load_all(relays, args.seed)
            print(f"  Loaded: {', '.join(loaded) if loaded else 'none'}")
            non_trainable = {"AF", "DF"}
            actually_skipped = [s for s in skipped if s not in non_trainable]
            if actually_skipped:
                print(f"  [WARNING] Missing weights for: "
                      f"{', '.join(actually_skipped)}")
        elif args.resume:
            relays = train_or_load_models(args, _ckpt)
        else:
            relays = train_models(args)
            if args.save_weights:
                print(f"\n=== Saving weights (seed={args.seed}) ===")
                config = {
                    k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, list))
                }
                saved = _ckpt.save_all(relays, args.seed, config)
                ckpt_dir = _ckpt.checkpoint_dir(args.seed)
                print(f"  Saved {len(saved)} relay(s) \u2192 {ckpt_dir}/")

        # ── Helper: check whether to skip a stage ────────────────────
        def _should_skip(save_path):
            if args.skip_existing and os.path.isfile(save_path):
                print(f"\n  [SKIP] {save_path} already exists")
                return True
            return False

        # AWGN comparison
        awgn_plot = f"{results_dir}/awgn_comparison_ci_{constellation}.png"
        if not _should_skip(awgn_plot):
            awgn_ber, awgn_trials = run_awgn_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, awgn_ber, awgn_trials)

            if not args.no_plots:
                ci_dict = {}
                for name, trials in awgn_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    ci_dict[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, awgn_ber, ci_dict,
                    title=f"AWGN Channel \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=awgn_plot,
                )
                plot_complexity_comparison(
                    relays, awgn_ber, snr_range,
                    save_path=f"{results_dir}/complexity_comparison_all_relays_{constellation}.png",
                )

        # Rayleigh Fading comparison
        fading_plot = f"{results_dir}/fading_comparison_{constellation}.png"
        if not _should_skip(fading_plot):
            fading_ber, fading_trials = run_fading_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, fading_ber, fading_trials)

            if not args.no_plots:
                fading_ci = {}
                for name, trials in fading_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    fading_ci[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, fading_ber, fading_ci,
                    title=f"Rayleigh Fading Channel \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=fading_plot,
                )

        # Rician fading comparison
        rician_plot = f"{results_dir}/rician_comparison_ci_{constellation}.png"
        if not _should_skip(rician_plot):
            rician_ber, rician_trials = run_rician_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, rician_ber, rician_trials)

            if not args.no_plots:
                rician_ci = {}
                for name, trials in rician_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    rician_ci[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, rician_ber, rician_ci,
                    title=f"Rician Fading Channel (K=3) \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=rician_plot,
                )

        # 2\u00d72 MIMO ZF comparison
        mimo_plot = f"{results_dir}/mimo_2x2_comparison_ci_{constellation}.png"
        if not _should_skip(mimo_plot):
            mimo_ber, mimo_trials = run_mimo_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, mimo_ber, mimo_trials)

            if not args.no_plots:
                mimo_ci = {}
                for name, trials in mimo_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    mimo_ci[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, mimo_ber, mimo_ci,
                    title=f"2\u00d72 MIMO (Rayleigh) \u2013 ZF Equalization \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=mimo_plot,
                )

        # 2\u00d72 MIMO MMSE comparison
        mimo_mmse_plot = f"{results_dir}/mimo_2x2_mmse_comparison_ci_{constellation}.png"
        if not _should_skip(mimo_mmse_plot):
            mimo_mmse_ber, mimo_mmse_trials = run_mimo_mmse_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, mimo_mmse_ber, mimo_mmse_trials)

            if not args.no_plots:
                mimo_mmse_ci = {}
                for name, trials in mimo_mmse_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    mimo_mmse_ci[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, mimo_mmse_ber, mimo_mmse_ci,
                    title=f"2\u00d72 MIMO (Rayleigh) \u2013 MMSE Equalization \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=mimo_mmse_plot,
                )

        # 2\u00d72 MIMO SIC comparison
        mimo_sic_plot = f"{results_dir}/mimo_2x2_sic_comparison_ci_{constellation}.png"
        if not _should_skip(mimo_sic_plot):
            mimo_sic_ber, mimo_sic_trials = run_mimo_sic_comparison(
                relays, snr_range, args.bits_per_trial, args.num_trials,
                modulation=constellation,
            )
            print_significance(snr_range, mimo_sic_ber, mimo_sic_trials)

            if not args.no_plots:
                mimo_sic_ci = {}
                for name, trials in mimo_sic_trials.items():
                    lo, hi = compute_confidence_interval(trials)
                    mimo_sic_ci[name] = (lo, hi)
                plot_ber_with_ci(
                    snr_range, mimo_sic_ber, mimo_sic_ci,
                    title=f"2\u00d72 MIMO (Rayleigh) \u2013 SIC Equalization \u2013 All Relay Methods (95% CI) [{constellation.upper()}]{ln_tag}",
                    save_path=mimo_sic_plot,
                )

        # \u2500\u2500 Normalized (~3K params) apples-to-apples comparison \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if args.include_normalized:
            if not _HAS_NORMALIZED:
                print("\n  [WARNING] checkpoint_22_normalized_3k could not be imported; "
                      "skipping normalized comparison.")
            else:
                if args.inference_only:
                    _ln = getattr(args, 'layer_norm', False)
                    _act = getattr(args, 'output_activation', 'tanh')
                    relays_3k = build_all_3k(
                        prefer_gpu=args.gpu,
                        include_sequence_models=args.include_sequence_models,
                        include_cgan=getattr(args, 'include_cgan', False),
                        use_input_norm=_ln,
                        output_activation=_act,
                    )
                    loaded_3k, _ = _ckpt.load_all(relays_3k, args.seed)
                    print(f"\n  Loaded normalized models: "
                          f"{', '.join(loaded_3k) if loaded_3k else 'none'}")
                elif args.resume:
                    relays_3k = train_or_load_normalized(args, _ckpt)
                else:
                    relays_3k = train_normalized_models(args)
                    if args.save_weights:
                        _ckpt.save_all(relays_3k, args.seed)
                run_normalized_comparison(
                    relays_3k, snr_range,
                    args.bits_per_trial, args.num_trials,
                    no_plots=args.no_plots,
                    modulation=constellation,
                    skip_existing=args.skip_existing,
                    results_dir=results_dir,
                )

    print(f"\nDone. Results saved to {results_dir}/")
    if args.log_timings:
        print(f"Total elapsed time: {perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
