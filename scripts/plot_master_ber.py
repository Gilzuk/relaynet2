"""
Generate a Master BER Chart — all 9 relay strategies across all 6 channels.

Loads pre-trained weights from trained_weights/seed_42/ and runs a quick
Monte-Carlo simulation (5 000 bits × 5 trials per relay/channel) to produce
a publication-quality 2×3 grid figure.

Usage:
    python scripts/plot_master_ber.py
    python scripts/plot_master_ber.py --full      # 10 000 bits × 10 trials
    python scripts/plot_master_ber.py --seed 42
"""

import os, sys, argparse, time
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 for Windows console
for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Imports ─────────────────────────────────────────────────────────
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay
from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper

from relaynet.channels.awgn import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.channels.mimo import mimo_2x2_channel, mimo_2x2_mmse_channel, mimo_2x2_sic_channel
from relaynet.simulation.runner import run_monte_carlo
from relaynet.utils.checkpoint_manager import CheckpointManager

# ── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Master BER Chart — all relays, all channels")
    p.add_argument("--full", action="store_true",
                   help="Higher fidelity (10k bits × 10 trials)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights-dir", type=str, default="trained_weights")
    p.add_argument("--gpu", action="store_true", default=True)
    return p.parse_args()

# ── Constants ───────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CHANNELS = [
    ("AWGN",           None),
    ("Rayleigh",       rayleigh_fading_channel),
    ("Rician (K=3)",   lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0)),
    ("2×2 MIMO ZF",   mimo_2x2_channel),
    ("2×2 MIMO MMSE", mimo_2x2_mmse_channel),
    ("2×2 MIMO SIC",  mimo_2x2_sic_channel),
]

# 9 distinct styles for 9 relay strategies
STYLES = {
    "AF":              dict(color="gray",    marker="x", ls="--", lw=1.5, ms=6, alpha=0.6),
    "DF":              dict(color="black",   marker="o", ls="-",  lw=2.0, ms=7, alpha=0.9),
    "GenAI (169p)":    dict(color="magenta", marker="d", ls="-",  lw=2.0, ms=7, alpha=0.8),
    "Hybrid":          dict(color="red",     marker="h", ls="-",  lw=2.0, ms=7, alpha=0.8),
    "VAE":             dict(color="cyan",    marker="s", ls="-",  lw=2.0, ms=7, alpha=0.8),
    "CGAN (WGAN-GP)":  dict(color="orange",  marker="P", ls="-",  lw=2.0, ms=7, alpha=0.8),
    "Transformer":     dict(color="green",   marker="^", ls="-",  lw=2.0, ms=7, alpha=0.8),
    "Mamba S6":        dict(color="blue",    marker="s", ls="-",  lw=2.5, ms=7, alpha=0.9),
    "Mamba2 (SSD)":    dict(color="#e377c2", marker="X", ls="-",  lw=2.5, ms=7, alpha=0.9),
}

# ── Relay construction + weight loading ─────────────────────────────

def build_relays(args):
    """Create relay instances and load trained weights."""
    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False),
        "Hybrid": HybridRelay(snr_threshold=5.0, prefer_gpu=False),
        "VAE": VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False),
        "CGAN (WGAN-GP)": CGANRelay(window_size=7, noise_size=8, lambda_gp=10,
                                     lambda_l1=20, n_critic=5, prefer_gpu=args.gpu),
        "Transformer": TransformerRelayWrapper(target_power=1.0, window_size=11,
                                                d_model=32, num_heads=4,
                                                num_layers=2, prefer_gpu=args.gpu),
        "Mamba S6": MambaRelayWrapper(target_power=1.0, window_size=11,
                                       d_model=32, d_state=16,
                                       num_layers=2, prefer_gpu=args.gpu),
        "Mamba2 (SSD)": Mamba2RelayWrapper(target_power=1.0, window_size=11,
                                            d_model=32, d_state=16,
                                            num_layers=2, prefer_gpu=args.gpu),
    }

    ckpt = CheckpointManager(args.weights_dir)
    loaded = 0
    for name, relay in relays.items():
        if ckpt.load_relay(relay, name, args.seed):
            loaded += 1
            print(f"  ✓ Loaded weights: {name}")
        else:
            print(f"  • No weights (classical/untrained): {name}")

    print(f"  Loaded {loaded} relay weight(s) from seed_{args.seed}\n")
    return relays

# ── Simulation ──────────────────────────────────────────────────────

def simulate_all(relays, snr_range, bits, trials):
    """Return results[channel_name][relay_name] = mean_ber_array."""
    results = {}
    for ch_name, ch_fn in CHANNELS:
        print(f"  === {ch_name} ===")
        results[ch_name] = {}
        for r_name, relay in relays.items():
            print(f"    {r_name} …", end=" ", flush=True)
            t0 = time.perf_counter()
            kw = dict(num_bits_per_trial=bits, num_trials=trials)
            if ch_fn is not None:
                kw["channel_fn"] = ch_fn
            _, ber, _ = run_monte_carlo(relay, snr_range, **kw)
            dt = time.perf_counter() - t0
            results[ch_name][r_name] = ber
            print(f"done [{ber.min():.2e}, {ber.max():.2e}] ({dt:.1f}s)")
    return results

# ── Plotting ────────────────────────────────────────────────────────

def plot_master(snr_range, results):
    """Create 2×3 grid with all 6 channels."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ch_names = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.flatten()

    for idx, ch_name in enumerate(ch_names):
        ax = axes_flat[idx]
        relay_results = results[ch_name]
        for r_name, ber in relay_results.items():
            sty = STYLES.get(r_name, dict(color="purple", marker=".", ls="-",
                                           lw=1.5, ms=6, alpha=0.7))
            clipped = np.clip(ber, 1e-5, 1)
            ax.semilogy(snr_range, clipped,
                        marker=sty["marker"], color=sty["color"],
                        linestyle=sty["ls"], linewidth=sty["lw"],
                        markersize=sty["ms"], alpha=sty["alpha"],
                        label=r_name)
        ax.set_xlabel("SNR (dB)", fontsize=10)
        ax.set_ylabel("BER", fontsize=10)
        ax.set_title(ch_name, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=5e-5)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="lower left", ncol=2)

    fig.suptitle("Master BER Comparison — All Nine Relay Strategies Across All Channels",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(RESULTS_DIR, "master_ber_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved: {path}")
    return path

# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    t_total = time.perf_counter()

    snr_range = np.arange(0, 22, 2)
    bits   = 10_000 if args.full else 5_000
    trials = 10     if args.full else 5

    print("=" * 60)
    print("MASTER BER CHART — 9 relays × 6 channels")
    print(f"  bits/trial={bits}  trials={trials}  seed={args.seed}")
    print("=" * 60)

    print("\nLoading relay models …")
    relays = build_relays(args)

    print("Running simulations …")
    results = simulate_all(relays, snr_range, bits, trials)

    print("\nGenerating master chart …")
    plot_master(snr_range, results)

    dt = time.perf_counter() - t_total
    print(f"\nTotal time: {dt:.0f}s")


if __name__ == "__main__":
    main()
