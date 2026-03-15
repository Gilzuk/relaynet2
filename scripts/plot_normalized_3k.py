"""
Plot normalized ~3K-parameter relay comparison across all channels.
Runs a quick Monte-Carlo simulation and generates publication-quality plots.

Usage:
    python scripts/plot_normalized_3k.py
    python scripts/plot_normalized_3k.py --full   # more samples/trials
"""

import os, sys, time, argparse
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

from relaynet.channels.awgn import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.channels.mimo import mimo_2x2_channel, mimo_2x2_mmse_channel, mimo_2x2_sic_channel
from relaynet.simulation.runner import run_monte_carlo
from relaynet.utils.checkpoint_manager import CheckpointManager
from checkpoints.checkpoint_22_normalized_3k import build_all_3k

# ── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Normalized 3K relay plots")
    p.add_argument("--full", action="store_true",
                   help="Use higher-fidelity settings (more samples/epochs/trials)")
    p.add_argument("--include-cgan", action="store_true",
                   help="Include CGAN (WGAN-GP) relay (slow: ~12x training overhead)")
    p.add_argument("--gpu", action="store_true", default=True,
                   help="Use GPU for Transformer/Mamba models (default: True)")
    p.add_argument("--weights-dir", type=str, default="trained_weights",
                   help="Directory for saving/loading 3K model weights")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ── Constants ───────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CHANNELS = [
    ("AWGN",           None),
    ("Rayleigh",       rayleigh_fading_channel),
    ("Rician (K=3)",   lambda sig, snr: rician_fading_channel(sig, snr, k_factor=3.0)),
    ("2x2 MIMO ZF",   mimo_2x2_channel),
    ("2x2 MIMO MMSE", mimo_2x2_mmse_channel),
    ("2x2 MIMO SIC",  mimo_2x2_sic_channel),
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# ── Training ────────────────────────────────────────────────────────

def train_relays(full=False, seed=42, include_cgan=False, gpu=True,
                 weights_dir="trained_weights"):
    """Build, load (if available) or train all 3K relays."""
    print("\n=== Building 3K-normalized relays ===")
    relays = build_all_3k(prefer_gpu=False, include_sequence_models=True,
                           include_cgan=include_cgan, prefer_gpu_seq=gpu)

    ckpt = CheckpointManager(weights_dir)
    loaded, _ = ckpt.load_all(relays, seed)
    loaded_set = set(loaded)
    if loaded:
        print(f"  Loaded from checkpoint: {', '.join(loaded)}")

    to_train = [n for n in relays if n not in loaded_set]
    if not to_train:
        print("  All 3K models loaded — skipping training.")
        return relays

    print(f"  Need to train: {', '.join(to_train)}")
    samples = 25_000 if full else 10_000
    epochs  = 100    if full else 50
    cgan_epochs = 200 if full else 100

    for name in to_train:
        relay = relays[name]
        n = getattr(relay, "num_params", "?")
        print(f"  Training {name} ({n}p) ...", end=" ", flush=True)
        t0 = time.perf_counter()
        ep = cgan_epochs if "CGAN" in name else epochs
        if "Transformer" in name or "Mamba" in name:
            relay.train(training_snrs=[5, 10, 15], num_samples=samples,
                        epochs=ep, lr=0.001)
        else:
            relay.train(training_snrs=[5, 10, 15], num_samples=samples,
                        epochs=ep, seed=seed)
        dt = time.perf_counter() - t0
        print(f"done ({dt:.1f}s)")

    # Save all weights (including freshly trained)
    saved = ckpt.save_all(relays, seed)
    if saved:
        print(f"  Saved weights: {', '.join(saved.keys())}")

    return relays

# ── Simulation ──────────────────────────────────────────────────────

def run_all_channels(relays, snr_range, bits, trials):
    """Return results[channel_name][relay_name] = (ber_mean, ber_trials)."""
    results = {}
    for ch_name, ch_fn in CHANNELS:
        print(f"\n=== {ch_name} Channel ===")
        results[ch_name] = {}
        for r_name, relay in relays.items():
            print(f"  {r_name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            kw = dict(num_bits_per_trial=bits, num_trials=trials)
            if ch_fn is not None:
                kw["channel_fn"] = ch_fn
            _, ber, trials_arr = run_monte_carlo(relay, snr_range, **kw)
            dt = time.perf_counter() - t0
            results[ch_name][r_name] = (ber, trials_arr)
            print(f"done  BER=[{ber.min():.2e}, {ber.max():.2e}]  ({dt:.1f}s)")
    return results

# ── Plotting ────────────────────────────────────────────────────────

def _plot_channel(ax, snrs, relay_results, title):
    """Plot BER curves for one channel onto given axes."""
    for i, (r_name, (ber, trials_arr)) in enumerate(relay_results.items()):
        lo = np.percentile(trials_arr, 2.5, axis=1)
        hi = np.percentile(trials_arr, 97.5, axis=1)
        clipped = np.clip(ber, 1e-5, 1)
        ax.semilogy(snrs, clipped,
                     marker=MARKERS[i % len(MARKERS)],
                     color=COLORS[i % len(COLORS)],
                     linewidth=2, markersize=6,
                     label=r_name)
        ax.fill_between(snrs,
                         np.clip(lo, 1e-5, 1),
                         np.clip(hi, 1e-5, 1),
                         alpha=0.12, color=COLORS[i % len(COLORS)])
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=5e-5)


def plot_individual(snrs, results):
    """One PNG per channel."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for ch_name, relay_results in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_channel(ax, snrs, relay_results, f"Normalized ~3K Params - {ch_name} (95% CI)")
        ax.legend(fontsize=10, loc="upper right")
        safe = ch_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        path = os.path.join(RESULTS_DIR, f"normalized_3k_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_consolidated(snrs, results, relays):
    """3x3 grid with all channels + parameter table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ch_names = list(results.keys())

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes_flat = axes.flatten()

    for idx, ch_name in enumerate(ch_names):
        _plot_channel(axes_flat[idx], snrs, results[ch_name], ch_name)
        if idx == 0:
            axes_flat[idx].legend(fontsize=7, loc="upper right")

    # 7th cell: parameter table
    ax = axes_flat[6]
    ax.axis("off")
    table_data = []
    for r_name, relay in relays.items():
        n = getattr(relay, "num_params", "?")
        table_data.append([r_name, f"{n:,}" if isinstance(n, int) else str(n)])
    tbl = ax.table(cellText=table_data, colLabels=["Model", "Params"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(0.8, 1.8)
    ax.set_title("Parameter Counts", fontsize=12, fontweight="bold")

    # Hide remaining empty cells
    for i in range(7, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle("Normalized ~3K Parameter Relay Comparison Across All Channels",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(RESULTS_DIR, "normalized_3k_all_channels.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved consolidated: {path}")


# ── BER Summary Table ───────────────────────────────────────────────

def print_summary(snrs, results):
    """Print a BER summary table at key SNR points."""
    key_snrs = [0, 10, 20]
    snrs_list = list(snrs)
    idx_map = {s: snrs_list.index(s) for s in key_snrs if s in snrs_list}

    print("\n" + "=" * 80)
    print("BER Summary Table (Normalized 3K Params)")
    print("=" * 80)

    relay_names = list(list(results.values())[0].keys())
    header = f"{'Channel':<18}" + "".join(f"{'  ' + rn:>16}" for rn in relay_names)
    print(header)
    print("-" * len(header))

    for ch_name, relay_results in results.items():
        for snr_db in key_snrs:
            if snr_db not in idx_map:
                continue
            si = idx_map[snr_db]
            row = f"{ch_name + f' @{snr_db}dB':<18}"
            for r_name in relay_names:
                ber_val = relay_results[r_name][0][si]
                row += f"{ber_val:>16.4e}"
            print(row)
        print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    t_total = time.perf_counter()

    snr_range = np.arange(0, 22, 2)
    bits   = 10_000 if args.full else 5_000
    trials = 10     if args.full else 5

    relays  = train_relays(full=args.full, seed=args.seed,
                            include_cgan=args.include_cgan, gpu=args.gpu,
                            weights_dir=args.weights_dir)
    results = run_all_channels(relays, snr_range, bits, trials)

    print_summary(snr_range, results)

    print("\nGenerating plots...")
    plot_individual(snr_range, results)
    plot_consolidated(snr_range, results, relays)

    dt = time.perf_counter() - t_total
    print(f"\nTotal time: {dt:.0f}s")


if __name__ == "__main__":
    main()
