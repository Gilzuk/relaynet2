"""
run_layernorm_experiment.py
============================
Experiment: does adding input LayerNorm after ``input_proj`` in the
sequence-model relays improve BER performance or at least preserve it?

The script trains each sequence relay (Transformer, Mamba S6, Mamba2)
twice:

1. **Baseline** — original architecture (no input LayerNorm)
2. **+InputLN** — ``nn.LayerNorm(d_model)`` inserted after ``input_proj``

Both variants are evaluated on BPSK AWGN and Rayleigh (10 trials ×
10 000 bits) across SNR = 0 … 20 dB in steps of 2 dB.  AF and DF
baselines are included for reference.

Usage::

    python scripts/run_layernorm_experiment.py
    python scripts/run_layernorm_experiment.py --quick

Output:
    results/layernorm/layernorm_awgn.png
    results/layernorm/layernorm_rayleigh.png
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
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.fading import rayleigh_fading_channel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Input LayerNorm experiment — baseline vs +InputLN for sequence relays"
    )
    p.add_argument("--quick", action="store_true",
                   help="Reduced samples/epochs for fast smoke-test")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Relay builders ───────────────────────────────────────────────────

MODEL_SPECS = [
    ("Transformer", TransformerRelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, num_heads=4, num_layers=2)),
    ("Mamba S6", MambaRelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2)),
    ("Mamba2", Mamba2RelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2)),
]


def _build_relay(name, cls, base_kw, use_input_norm, gpu, quick):
    """Build & train a sequence relay with or without input LayerNorm."""
    tag = "+InputLN" if use_input_norm else "Baseline"
    samples = 50_000 if not quick else 5_000
    epochs  = 100    if not quick else 15
    seq_epochs = 50  if not quick else 15   # Transformer uses 50 in main pipeline

    print(f"\n  Building {name} ({tag}) …")
    t0 = perf_counter()
    kw = {**base_kw, "prefer_gpu": gpu, "output_activation": "tanh",
          "use_input_norm": use_input_norm}
    r = cls(**kw)

    ep = seq_epochs if name == "Transformer" else epochs
    r.train(training_snrs=[5, 10, 15], num_samples=samples,
            epochs=ep, lr=0.001, training_modulation="bpsk")
    dt = perf_counter() - t0
    print(f"  {name} ({tag}) training done ({dt:.1f}s)")
    return r


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(relays, snr_range, args, channel_name, channel_fn=None):
    """Evaluate all relays on BPSK.  Returns {name: (ber_mean, ber_trials)}."""
    results = {}
    for name, relay in relays.items():
        print(f"    {name} …", end=" ", flush=True)
        t0 = perf_counter()
        _, ber_mean, ber_trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            seed_offset=args.seed,
            modulation="bpsk",
        )
        print(f"done ({perf_counter()-t0:.1f}s)")
        results[name] = (ber_mean, ber_trials)
    return results


# ── Plotting ─────────────────────────────────────────────────────────

COLORS = {
    "AF":  "#888888",
    "DF":  "#444444",
    "Transformer (Baseline)": "#ff7f00",
    "Transformer (+InputLN)": "#e41a1c",
    "Mamba S6 (Baseline)":    "#a65628",
    "Mamba S6 (+InputLN)":    "#d62728",
    "Mamba2 (Baseline)":      "#f781bf",
    "Mamba2 (+InputLN)":      "#9467bd",
}

LINE_STYLES = {
    "AF":  {"ls": "-.", "lw": 1.0, "alpha": 0.6},
    "DF":  {"ls": "-.", "lw": 1.0, "alpha": 0.6},
}
# baseline = dashed, +InputLN = solid
for _m in ["Transformer", "Mamba S6", "Mamba2"]:
    LINE_STYLES[f"{_m} (Baseline)"] = {"ls": "--", "lw": 1.5, "alpha": 0.7}
    LINE_STYLES[f"{_m} (+InputLN)"] = {"ls": "-",  "lw": 2.5, "alpha": 1.0}


def plot_comparison(results, snr_range, channel_name, out_path):
    """Plot BER vs SNR for baseline and +InputLN variants."""
    if not _HAS_PLT:
        print("  matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    order = ["AF", "DF"]
    for m in ["Transformer", "Mamba S6", "Mamba2"]:
        order += [f"{m} (Baseline)", f"{m} (+InputLN)"]

    for name in order:
        if name not in results:
            continue
        ber_mean, ber_trials = results[name]
        ci_lo = np.percentile(ber_trials, 5, axis=1)
        ci_hi = np.percentile(ber_trials, 95, axis=1)
        color = COLORS.get(name, "#333333")
        style = LINE_STYLES.get(name, {})
        ax.semilogy(snr_range, ber_mean, marker="o", markersize=4,
                     color=color, label=name, **style)
        ax.fill_between(snr_range, ci_lo, ci_hi, color=color, alpha=0.10)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"Input LayerNorm Effect on Sequence Relays — BPSK {channel_name}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ── Summary table ────────────────────────────────────────────────────

def print_table(results, snr_range, channel_name):
    """Print a compact BER table with Δ summaries per model."""
    print(f"\n  === {channel_name} BER Results ===")
    header = f"  {'Relay':<30}"
    for s in snr_range:
        header += f" {s:>5.0f}"
    print(header)
    print("  " + "-" * len(header))

    order = ["AF", "DF"]
    for m in ["Transformer", "Mamba S6", "Mamba2"]:
        order += [f"{m} (Baseline)", f"{m} (+InputLN)"]

    for name in order:
        if name not in results:
            continue
        ber_mean, _ = results[name]
        row = f"  {name:<30}"
        for b in ber_mean:
            row += f" {b:>5.4f}" if b >= 0.0001 else f" {b:>5.1e}"
        print(row)

    # Improvement summary per model
    for m in ["Transformer", "Mamba S6", "Mamba2"]:
        bkey = f"{m} (Baseline)"
        nkey = f"{m} (+InputLN)"
        if bkey in results and nkey in results:
            base = results[bkey][0]
            norm = results[nkey][0]
            print(f"\n  Δ BER {m} (Baseline − InputLN):")
            for i, s in enumerate(snr_range):
                delta = base[i] - norm[i]
                pct = 100.0 * delta / max(base[i], 1e-12)
                sign = "+" if delta > 0 else ""
                print(f"    SNR {s:>4.0f} dB : {sign}{delta:+.6f}  ({sign}{pct:+.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "layernorm")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("INPUT LAYERNORM EXPERIMENT — SEQUENCE RELAYS")
    print("=" * 70)

    # ── Build classical baselines ────────────────────────────────────
    print("\n[1/4] Building classical baselines …")
    af = AmplifyAndForwardRelay(target_power=1.0)
    df = DecodeAndForwardRelay()

    # ── Train all sequence models (baseline + InputLN) ───────────────
    print("\n[2/4] Training sequence relay variants …")
    relays = {"AF": af, "DF": df}
    for model_name, cls, base_kw in MODEL_SPECS:
        r_base = _build_relay(model_name, cls, base_kw,
                              use_input_norm=False, gpu=args.gpu, quick=args.quick)
        r_norm = _build_relay(model_name, cls, base_kw,
                              use_input_norm=True,  gpu=args.gpu, quick=args.quick)
        relays[f"{model_name} (Baseline)"] = r_base
        relays[f"{model_name} (+InputLN)"] = r_norm

    # ── AWGN evaluation ──────────────────────────────────────────────
    print("\n[3/4] Evaluating on BPSK AWGN …")
    res_awgn = evaluate(relays, snr_range, args, "AWGN")
    print_table(res_awgn, snr_range, "AWGN")
    plot_comparison(res_awgn, snr_range, "AWGN",
                    os.path.join(out_dir, "layernorm_awgn.png"))

    # ── Rayleigh evaluation ──────────────────────────────────────────
    print("\n[4/4] Evaluating on BPSK Rayleigh …")
    res_ray = evaluate(relays, snr_range, args, "Rayleigh",
                       channel_fn=rayleigh_fading_channel)
    print_table(res_ray, snr_range, "Rayleigh")
    plot_comparison(res_ray, snr_range, "Rayleigh",
                    os.path.join(out_dir, "layernorm_rayleigh.png"))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
