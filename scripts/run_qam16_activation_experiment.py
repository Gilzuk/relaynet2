"""
run_qam16_activation_experiment.py
===================================
Experiment: does replacing tanh with a linear / hardtanh output
activation *and* training on 16-QAM (PAM-4) data eliminate the
AI-relay BER floor on 16-QAM?

The script trains each AI relay three ways:

1. **Baseline** — tanh output, trained on BPSK  (existing regime)
2. **QAM16-linear** — identity output, trained on PAM-4 (QAM16 I/Q)
3. **QAM16-hardtanh** — Hardtanh[-0.949,+0.949], trained on PAM-4

All variants are evaluated on 16-QAM AWGN (10 trials × 10 000 bits).
The classical AF and DF relays are included as non-trainable baselines.

Usage::

    python scripts/run_qam16_activation_experiment.py
    python scripts/run_qam16_activation_experiment.py --quick
    python scripts/run_qam16_activation_experiment.py --include-sequence-models --gpu

Output:
    results/qam16_activation/qam16_activation_awgn.png
    results/qam16_activation/qam16_activation_rayleigh.png
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
from relaynet.simulation.statistics import compute_confidence_interval
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


# ── Helpers ──────────────────────────────────────────────────────────

ACTIVATIONS = {
    "tanh":     "tanh",      # baseline (BPSK-trained)
    "linear":   "linear",    # QAM16-trained, no clipping
    "hardtanh": "hardtanh",  # QAM16-trained, clipped to ±3/√10
}

# Line-style per activation variant
VARIANT_STYLE = {
    "tanh":     {"ls": "--", "alpha": 0.5,  "lw": 1.0},   # dashed (baseline)
    "linear":   {"ls": "-",  "alpha": 1.0,  "lw": 2.0},   # solid (main)
    "hardtanh": {"ls": ":",  "alpha": 0.85, "lw": 2.5},   # dotted
}

COLORS = {
    "GenAI":       "#e41a1c",
    "Hybrid":      "#377eb8",
    "VAE":         "#4daf4a",
    "CGAN":        "#984ea3",
    "Transformer": "#ff7f00",
    "Mamba S6":    "#a65628",
    "Mamba2":      "#f781bf",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="QAM16 activation experiment — tanh vs linear vs hardtanh"
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--include-sequence-models", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Relay factories ──────────────────────────────────────────────────

def _build_feedforward_relays(activation, gpu, quick):
    """Build feedforward AI relays with a given output activation."""
    samples = 25_000 if not quick else 5_000
    epochs = 100 if not quick else 20
    vae_samples = 25_000 if not quick else 5_000
    cgan_samples = 10_000 if not quick else 5_000
    cgan_epochs = 50 if not quick else 20

    # Determine training modulation: tanh = BPSK baseline, others = QAM16
    train_mod = "bpsk" if activation == "tanh" else "qam16"

    relays = {}

    print(f"    GenAI …", end=" ", flush=True)
    t0 = perf_counter()
    g = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False,
                          output_activation=activation)
    g.train(training_snrs=[5, 10, 15], num_samples=samples,
            epochs=epochs, seed=42, training_modulation=train_mod)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["GenAI"] = g

    print(f"    Hybrid …", end=" ", flush=True)
    t0 = perf_counter()
    h = HybridRelay(snr_threshold=5.0, prefer_gpu=False,
                    output_activation=activation)
    h.train(training_snrs=[2, 4, 6], num_samples=samples,
            epochs=epochs, seed=42, training_modulation=train_mod)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["Hybrid"] = h

    print(f"    VAE …", end=" ", flush=True)
    t0 = perf_counter()
    v = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
                 output_activation=activation)
    v.train(training_snrs=[5, 10, 15], num_samples=vae_samples,
            epochs=epochs, seed=42, training_modulation=train_mod)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["VAE"] = v

    print(f"    CGAN …", end=" ", flush=True)
    t0 = perf_counter()
    c = CGANRelay(window_size=7, noise_size=8, lambda_gp=10,
                  lambda_l1=20, n_critic=5, prefer_gpu=gpu,
                  output_activation=activation)
    c.train(training_snrs=[5, 10, 15], num_samples=cgan_samples,
            epochs=cgan_epochs, seed=42, training_modulation=train_mod)
    print(f"done ({perf_counter()-t0:.1f}s)")
    relays["CGAN"] = c

    return relays


def _build_sequence_relays(activation, gpu, quick):
    """Build sequence model relays with a given output activation."""
    if not _HAS_SEQ:
        return {}

    train_mod = "bpsk" if activation == "tanh" else "qam16"
    seq_samples = 25_000 if not quick else 3_000
    seq_epochs = 50 if not quick else 10

    relays = {}
    for name, cls in [("Transformer", TransformerRelayWrapper),
                      ("Mamba S6", MambaRelayWrapper),
                      ("Mamba2", Mamba2RelayWrapper)]:
        print(f"    {name} …", end=" ", flush=True)
        t0 = perf_counter()
        kw = dict(target_power=1.0, window_size=11, d_model=32,
                  num_layers=2, prefer_gpu=gpu,
                  output_activation=activation)
        if "Mamba" in name:
            kw["d_state"] = 16
        else:
            kw["num_heads"] = 4
        r = cls(**kw)
        r.train(training_snrs=[5, 10, 15], num_samples=seq_samples,
                epochs=seq_epochs, lr=0.001,
                training_modulation=train_mod)
        print(f"done ({perf_counter()-t0:.1f}s)")
        relays[name] = r

    return relays


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(relays, snr_range, args, channel_name, channel_fn=None):
    """Evaluate all relays on 16-QAM. Returns {name: (ber, trials)}."""
    results = {}
    for name, relay in relays.items():
        print(f"    {name} …", end=" ", flush=True)
        t0 = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            modulation="qam16",
        )
        elapsed = perf_counter() - t0
        print(f"done ({elapsed:.1f}s, BER@16dB={ber[-1]:.4f})")
        results[name] = (ber, trials)
    return results


# ── Plotting ─────────────────────────────────────────────────────────

def plot_activation_comparison(all_results, snr_range, channel_name, out_dir):
    """Plot BER curves for all variants on a single figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 9))

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    relay_names = sorted({n for var in all_results.values() for n in var
                          if n not in ("AF", "DF")})
    marker_map = {n: markers[i % len(markers)] for i, n in enumerate(relay_names)}

    # Plot classical baselines first (always from "tanh" variant)
    for baseline_name in ("AF", "DF"):
        if "tanh" in all_results and baseline_name in all_results["tanh"]:
            ber, _ = all_results["tanh"][baseline_name]
            color = "gray" if baseline_name == "AF" else "black"
            ax.semilogy(snr_range, np.clip(ber, 1e-5, 1), "-",
                        color=color, lw=2.5, label=baseline_name,
                        marker="o" if baseline_name == "AF" else "s",
                        markevery=2, markersize=6)

    # Plot each relay × activation variant
    for act_name, results in all_results.items():
        style = VARIANT_STYLE[act_name]
        for relay_name, (ber, trials) in results.items():
            if relay_name in ("AF", "DF"):
                continue
            color = COLORS.get(relay_name, "#333333")
            label = f"{relay_name} [{act_name}]"
            ax.semilogy(snr_range, np.clip(ber, 1e-5, 1),
                        ls=style["ls"], color=color,
                        lw=style["lw"], alpha=style["alpha"],
                        marker=marker_map.get(relay_name, "o"),
                        markevery=2, markersize=5,
                        label=label)

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title(f"16-QAM Activation Experiment — {channel_name}", fontsize=14)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-4, 1)

    safe_ch = channel_name.lower().replace(" ", "_")
    fname = os.path.join(out_dir, f"qam16_activation_{safe_ch}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Summary table ────────────────────────────────────────────────────

def print_summary(all_results, snr_range, channel_name):
    """Print a summary table of BER at 16 dB."""
    idx_16 = list(snr_range).index(16.0) if 16.0 in snr_range else -1

    print(f"\n{'='*70}")
    print(f"  16-QAM BER @ 16 dB — {channel_name}")
    print(f"{'='*70}")
    print(f"  {'Relay':<16} {'tanh (BPSK)':>12} {'linear (QAM16)':>15} {'hardtanh (QAM16)':>17}")
    print(f"  {'-'*16} {'-'*12} {'-'*15} {'-'*17}")

    # Gather all relay names
    all_names = []
    for var in all_results.values():
        for n in var:
            if n not in all_names:
                all_names.append(n)

    for name in all_names:
        row = f"  {name:<16}"
        for act in ("tanh", "linear", "hardtanh"):
            if act in all_results and name in all_results[act]:
                ber, _ = all_results[act][name]
                val = ber[idx_16]
                row += f" {val:>12.4f}    " if act == "tanh" else f" {val:>12.4f}     "
            else:
                row += f" {'—':>12}     "
        print(row)
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)

    out_dir = os.path.join("results", "qam16_activation")
    os.makedirs(out_dir, exist_ok=True)

    t_start = perf_counter()

    # ── Train + evaluate for each activation variant ──────────────

    all_results_awgn = {}
    all_results_rayleigh = {}

    for act_name in ("tanh", "linear", "hardtanh"):
        print(f"\n{'='*70}")
        print(f"  VARIANT: {act_name.upper()} "
              f"({'BPSK-trained' if act_name == 'tanh' else 'QAM16-trained'})")
        print(f"{'='*70}")

        print(f"\n  Training feedforward relays [{act_name}] …")
        relays = _build_feedforward_relays(act_name, args.gpu, args.quick)

        if args.include_sequence_models:
            print(f"\n  Training sequence models [{act_name}] …")
            relays.update(_build_sequence_relays(act_name, args.gpu, args.quick))

        # Add classical baselines (only once — they are activation-independent)
        if act_name == "tanh":
            relays["AF"] = AmplifyAndForwardRelay(prefer_gpu=False)
            relays["DF"] = DecodeAndForwardRelay(prefer_gpu=False)

        print(f"\n  Evaluating on 16-QAM AWGN [{act_name}] …")
        all_results_awgn[act_name] = evaluate(
            relays, snr_range, args, "AWGN", channel_fn=None,
        )

        print(f"\n  Evaluating on 16-QAM Rayleigh [{act_name}] …")
        all_results_rayleigh[act_name] = evaluate(
            relays, snr_range, args, "Rayleigh",
            channel_fn=rayleigh_fading_channel,
        )

    # ── Summary ───────────────────────────────────────────────────

    print_summary(all_results_awgn, snr_range, "AWGN")
    print_summary(all_results_rayleigh, snr_range, "Rayleigh")

    # ── Plots ─────────────────────────────────────────────────────

    if _HAS_PLOTS:
        plot_activation_comparison(all_results_awgn, snr_range, "AWGN", out_dir)
        plot_activation_comparison(all_results_rayleigh, snr_range, "Rayleigh", out_dir)

    elapsed = perf_counter() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
