"""
run_layernorm_experiment.py
============================
Experiment: does adding input LayerNorm after ``input_proj`` in the
sequence-model relays improve BER performance or at least preserve it?

The script trains each sequence relay (Transformer, Mamba S6, Mamba2)
twice:

1. **Baseline** — original architecture (no input LayerNorm)
2. **+InputLN** — ``nn.LayerNorm(d_model)`` inserted after ``input_proj``

Both variants are evaluated on AWGN and Rayleigh (10 trials ×
10 000 bits) across SNR = 0 … 20 dB in steps of 2 dB.  AF and DF
baselines are included for reference.

Supports BPSK, QPSK, and 16-QAM constellations via ``--constellations``.

Usage::

    python scripts/run_layernorm_experiment.py
    python scripts/run_layernorm_experiment.py --quick
    python scripts/run_layernorm_experiment.py --constellations bpsk qpsk qam16

Output:
    results/layernorm/layernorm_awgn[_<mod>].png
    results/layernorm/layernorm_rayleigh[_<mod>].png
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
from relaynet.utils.activations import get_clip_range

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
    p.add_argument("--constellations", type=str, nargs="+",
                   default=["bpsk"],
                   choices=["bpsk", "qpsk", "qam16"],
                   help="Constellations to evaluate (default: bpsk)")
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


def _build_relay(name, cls, base_kw, use_input_norm, gpu, quick,
                 output_activation="tanh", clip_range=None,
                 training_modulation="bpsk", tag_override=None):
    """Build & train a sequence relay with or without input LayerNorm."""
    tag = tag_override if tag_override else ("+InputLN" if use_input_norm else "Baseline")
    samples = 50_000 if not quick else 5_000
    epochs  = 100    if not quick else 15
    seq_epochs = 50  if not quick else 15   # Transformer uses 50 in main pipeline

    print(f"\n  Building {name} ({tag}) …")
    t0 = perf_counter()
    kw = {**base_kw, "prefer_gpu": gpu, "output_activation": output_activation,
          "use_input_norm": use_input_norm, "clip_range": clip_range}
    r = cls(**kw)

    ep = seq_epochs if name == "Transformer" else epochs
    r.train(training_snrs=[5, 10, 15], num_samples=samples,
            epochs=ep, lr=0.001, training_modulation=training_modulation)
    dt = perf_counter() - t0
    print(f"  {name} ({tag}) training done ({dt:.1f}s)")
    return r


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(relays, snr_range, args, channel_name, channel_fn=None,
             modulation="bpsk"):
    """Evaluate all relays.  Returns {name: (ber_mean, ber_trials)}."""
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
            modulation=modulation,
        )
        print(f"done ({perf_counter()-t0:.1f}s)")
        results[name] = (ber_mean, ber_trials)
    return results


# ── Plotting ─────────────────────────────────────────────────────────
# See CHART_GUIDELINES.md for rationale behind each style decision.
# Colorblind-friendly palette (ColorBrewer-inspired, guideline 13)

# Unique color + marker + linestyle per curve (guidelines 1, 7, 8, 9, 14)
_STYLE_TABLE = {
    #  name                          color      marker  ls           alpha
    "AF":                          ("#999999", "x",    (0,(5,3)),   0.55),
    "DF":                          ("#555555", "+",    (0,(3,2,1,2)), 0.55),
    "Transformer (Baseline)":      ("#e69f00", "o",    "--",        0.85),
    "Transformer (+InputLN)":      ("#d55e00", "s",    "-",         0.95),
    "Transformer (+LN+Scaled)":    ("#cc5500", "*",    ":",         0.95),
    "Mamba S6 (Baseline)":         ("#0072b2", "^",    "--",        0.85),
    "Mamba S6 (+InputLN)":         ("#009e73", "D",    "-",         0.95),
    "Mamba S6 (+LN+Scaled)":       ("#005050", "*",    ":",         0.95),
    "Mamba2 (Baseline)":           ("#cc79a7", "v",    "--",        0.85),
    "Mamba2 (+InputLN)":           ("#56b4e9", "P",    "-",         0.95),
    "Mamba2 (+LN+Scaled)":         ("#0070a0", "*",    ":",         0.95),
}

_CURVE_ORDER = ["AF", "DF",
                "Transformer (Baseline)", "Transformer (+InputLN)", "Transformer (+LN+Scaled)",
                "Mamba S6 (Baseline)",    "Mamba S6 (+InputLN)", "Mamba S6 (+LN+Scaled)",
                "Mamba2 (Baseline)",      "Mamba2 (+InputLN)", "Mamba2 (+LN+Scaled)"]

# Small jitter so identical BER values remain separable (guideline 5)
_JITTER_MAP = {
    "Transformer (Baseline)": 1.00,
    "Transformer (+InputLN)": 1.00,
    "Transformer (+LN+Scaled)": 1.00,
    "Mamba S6 (Baseline)":    1.03,
    "Mamba S6 (+InputLN)":    0.97,
    "Mamba S6 (+LN+Scaled)":  1.00,
    "Mamba2 (Baseline)":      1.06,
    "Mamba2 (+InputLN)":      0.94,
    "Mamba2 (+LN+Scaled)":    1.00,
}

_NEURAL_NAMES = [n for n in _CURVE_ORDER if n not in ("AF", "DF")]
_MODEL_NAMES  = ["Transformer", "Mamba S6", "Mamba2"]


def _focused_ylim(results):
    """Return (ymin, ymax) focused one decade below the min nonzero BER (guideline 6)."""
    all_ber = []
    for ber_mean, _ in results.values():
        all_ber.extend(ber_mean[ber_mean > 0])
    if not all_ber:
        return 1e-5, 1.0
    min_ber = min(all_ber)
    decade = 10 ** np.floor(np.log10(min_ber))
    ymin = max(decade / 10, 1e-7)
    return ymin, 1.0


def _add_zoom_inset(ax, snr_range, results, ymin):
    """Add an inset zoom box for the congested low-BER region (guideline 2, 15)."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    means = {}
    for name in _CURVE_ORDER:
        if name not in results:
            continue
        m = results[name][0].copy()
        m[m <= 0] = np.nan
        means[name] = m
    if not means:
        return

    best_idx = None
    for idx in range(len(snr_range) - 1, -1, -1):
        valid = sum(1 for n in _NEURAL_NAMES
                    if n in means and not np.isnan(means[n][idx]))
        if valid >= 3:
            best_idx = idx
            break
    if best_idx is None or best_idx < 2:
        return

    lo = max(0, best_idx - 2)
    hi = min(len(snr_range) - 1, best_idx)
    x0, x1 = snr_range[lo] - 0.5, snr_range[hi] + 0.5

    vals = []
    for n in _NEURAL_NAMES:
        if n not in means:
            continue
        for idx in range(lo, hi + 1):
            v = means[n][idx]
            if not np.isnan(v):
                vals.append(v)
    if len(vals) < 2:
        return
    y0 = min(vals) * 0.4
    y1 = max(vals) * 2.5
    if y0 <= 0:
        y0 = ymin

    axins = inset_axes(ax, width="38%", height="35%", loc="lower left",
                       bbox_to_anchor=(0.07, 0.08, 1, 1),
                       bbox_transform=ax.transAxes)
    for name in _CURVE_ORDER:
        if name not in results:
            continue
        ber_mean = results[name][0].copy()
        jitter = _JITTER_MAP.get(name, 1.0)
        ber_plot = ber_mean * jitter
        ber_plot[ber_plot <= 0] = np.nan
        color, marker, ls, alpha = _STYLE_TABLE[name]
        axins.semilogy(snr_range, ber_plot, marker=marker, markersize=3,
                       color=color, ls=ls, lw=0.9, alpha=alpha)

    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.grid(True, which="both", ls=":", alpha=0.25)
    axins.tick_params(labelsize=6)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--", lw=0.5)


def _annotate_congested(ax, snr_range, results):
    """Add leader-line annotations in congested regions (guideline 4, 16)."""
    neural = {}
    for name in _NEURAL_NAMES:
        if name not in results:
            continue
        ber = results[name][0].copy()
        jitter = _JITTER_MAP.get(name, 1.0)
        ber = ber * jitter
        idxs = np.where(ber > 0)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[-1]
        neural[name] = (snr_range[idx], ber[idx])

    if len(neural) < 2:
        return

    items = sorted(neural.items(), key=lambda kv: kv[1][1], reverse=True)
    offsets_x = [30, 34, 38, 30, 34, 38]
    offsets_y  = [20, -20, 24, -24, 28, -28]
    for i, (name, (sx, sy)) in enumerate(items):
        short = name.replace(" (Baseline)", " Base").replace(" (+InputLN)", " +LN")
        color = _STYLE_TABLE[name][0]
        ax.annotate(
            short, xy=(sx, sy), fontsize=6.5, color=color, weight="bold",
            xytext=(offsets_x[i % 6], offsets_y[i % 6]),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6),
            ha="left", va="center",
        )


def plot_comparison(results, snr_range, channel_name, out_path,
                    modulation_tag="BPSK"):
    """Plot BER vs SNR — follows CHART_GUIDELINES.md (guidelines 1-16)."""
    if not _HAS_PLT:
        print("  matplotlib not available — skipping plot")
        return

    # Guideline 10: consistent aspect ratio ≈ 1.6 : 1
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in _CURVE_ORDER:
        if name not in results:
            continue
        ber_mean, ber_trials = results[name]
        ci_lo = np.percentile(ber_trials, 5, axis=1)
        ci_hi = np.percentile(ber_trials, 95, axis=1)

        color, marker, ls, alpha = _STYLE_TABLE[name]
        jitter = _JITTER_MAP.get(name, 1.0)
        ber_plot = ber_mean * jitter

        # Guideline 2: thin lines (lw ≤ 1.5)
        ax.semilogy(snr_range, ber_plot, marker=marker, markersize=4,
                     color=color, label=name, ls=ls, lw=1.2, alpha=alpha)
        ax.fill_between(snr_range,
                         ci_lo * jitter, ci_hi * jitter,
                         color=color, alpha=0.07)

    # Guideline 11: readable font sizes
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("BER", fontsize=14)
    ax.set_title(f"Input LayerNorm Effect — {modulation_tag} {channel_name}",
                 fontsize=16)
    ax.tick_params(labelsize=12)
    # Guideline 12: light gridlines
    ax.grid(True, which="both", ls=":", alpha=0.3, lw=0.5)
    # Guideline 6: focused y-axis
    ymin, ymax = _focused_ylim(results)
    ax.set_ylim(ymin, ymax)
    # Guideline 3: legend outside curve area
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              framealpha=0.92, edgecolor="0.8", borderpad=0.6)

    _add_zoom_inset(ax, snr_range, results, ymin)
    _annotate_congested(ax, snr_range, results)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ── Summary / achievement charts (guidelines 19, 20) ────────────────

def _delta_ber(results, model, snr_idx):
    """Return (baseline − +InputLN) BER at given SNR index."""
    bkey = f"{model} (Baseline)"
    nkey = f"{model} (+InputLN)"
    if bkey not in results or nkey not in results:
        return np.nan
    return results[bkey][0][snr_idx] - results[nkey][0][snr_idx]


def plot_summary_heatmap(all_results, snr_range, out_dir):
    """Guideline 19: heatmap of Δ BER (%) across constellations × models × channels."""
    if not _HAS_PLT:
        return
    import matplotlib.colors as mcolors

    snr_4  = int(np.argmin(np.abs(snr_range - 4)))
    snr_16 = int(np.argmin(np.abs(snr_range - 16)))

    for snr_idx, snr_label in [(snr_4, int(snr_range[snr_4])),
                                (snr_16, int(snr_range[snr_16]))]:
        labels_row = []
        labels_col = _MODEL_NAMES
        matrix = []
        for (constellation, channel), results in sorted(all_results.items()):
            labels_row.append(f"{constellation.upper()} {channel}")
            row = []
            for model in _MODEL_NAMES:
                delta = _delta_ber(results, model, snr_idx)
                base = results.get(f"{model} (Baseline)", (np.array([1]),))[0]
                bv = base[snr_idx] if snr_idx < len(base) else 1.0
                pct = 100.0 * delta / max(bv, 1e-12) if not np.isnan(delta) else 0.0
                row.append(pct)
            matrix.append(row)

        matrix = np.array(matrix)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(labels_row) + 1.5)))
        vmax = max(abs(matrix.min()), abs(matrix.max()), 5)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(labels_col)))
        ax.set_xticklabels(labels_col, fontsize=11)
        ax.set_yticks(range(len(labels_row)))
        ax.set_yticklabels(labels_row, fontsize=11)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center",
                        fontsize=10, color="black")
        ax.set_title(f"Δ BER % (Baseline − +InputLN) at {snr_label} dB",
                     fontsize=14)
        fig.colorbar(im, ax=ax, label="Δ BER %", shrink=0.8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"layernorm_summary_{snr_label}dB.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved summary heatmap → {path}")


def plot_achievement(all_results, snr_range, out_dir):
    """Guideline 20: winner chart — best variant highlighted, rest faded."""
    if not _HAS_PLT:
        return

    snr_16 = int(np.argmin(np.abs(snr_range - 16)))

    scenarios = []
    winners = []
    for (constellation, channel), results in sorted(all_results.items()):
        for model in _MODEL_NAMES:
            bkey = f"{model} (Baseline)"
            nkey = f"{model} (+InputLN)"
            if bkey not in results or nkey not in results:
                continue
            delta = _delta_ber(results, model, snr_16)
            win = "+InputLN" if delta > 0 else "Baseline"
            scenarios.append(f"{constellation.upper()}\n{channel}\n{model}")
            winners.append(win)

    if not scenarios:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 0.9), 5))
    colors_bar = ["#009e73" if w == "+InputLN" else "#d55e00" for w in winners]
    alphas = [1.0 if w == "+InputLN" else 0.35 for w in winners]
    x = np.arange(len(scenarios))

    for i, (xi, ci, ai, w) in enumerate(zip(x, colors_bar, alphas, winners)):
        ax.bar(xi, 1, color=ci, alpha=ai, edgecolor="white", lw=0.5)
        ax.text(xi, 0.5, w, ha="center", va="center", fontsize=9,
                weight="bold" if ai == 1.0 else "normal",
                color="white" if ai == 1.0 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=7, ha="center")
    ax.set_yticks([])
    ax.set_title("LayerNorm Winner per Scenario (SNR = 16 dB)", fontsize=14)
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "layernorm_achievement.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved achievement chart → {path}")


# ── Summary table ────────────────────────────────────────────────────

def print_table(results, snr_range, channel_name):
    """Print a compact BER table with Δ summaries per model (guideline 18)."""
    print(f"\n  === {channel_name} BER Results ===")
    header = f"  {'Relay':<30}"
    for s in snr_range:
        header += f" {s:>5.0f}"
    print(header)
    print("  " + "-" * len(header))

    order = ["AF", "DF"]
    for m in _MODEL_NAMES:
        order += [f"{m} (Baseline)", f"{m} (+InputLN)", f"{m} (+LN+Scaled)"]

    for name in order:
        if name not in results:
            continue
        ber_mean, _ = results[name]
        row = f"  {name:<30}"
        for b in ber_mean:
            row += f" {b:>5.4f}" if b >= 0.0001 else f" {b:>5.1e}"
        print(row)

    # Improvement summary per model
    for m in _MODEL_NAMES:
        bkey = f"{m} (Baseline)"
        nkey = f"{m} (+InputLN)"
        skey = f"{m} (+LN+Scaled)"
        
        if bkey in results and nkey in results:
            base = results[bkey][0]
            norm = results[nkey][0]
            print(f"\n  Δ BER {m} (Baseline − InputLN):")
            for i, s in enumerate(snr_range):
                delta = base[i] - norm[i]
                pct = 100.0 * delta / max(base[i], 1e-12)
                sign = "+" if delta > 0 else ""
                print(f"    SNR {s:>4.0f} dB : {sign}{delta:+.6f}  ({sign}{pct:+.1f}%)")
                
        if bkey in results and skey in results:
            base = results[bkey][0]
            scaled = results[skey][0]
            print(f"\n  Δ BER {m} (Baseline − LN+Scaled):")
            for i, s in enumerate(snr_range):
                delta = base[i] - scaled[i]
                pct = 100.0 * delta / max(base[i], 1e-12)
                sign = "+" if delta > 0 else ""
                print(f"    SNR {s:>4.0f} dB : {sign}{delta:+.6f}  ({sign}{pct:+.1f}%)")

    # Guideline 18: summary at key SNR points (4 dB, 16 dB)
    snr_4  = int(np.argmin(np.abs(snr_range - 4)))
    snr_16 = int(np.argmin(np.abs(snr_range - 16)))
    print(f"\n  Key SNR Summary (4 dB / 16 dB):")
    print(f"  {'Model':<18} {'Base@4dB':>10} {'+LN@4dB':>10} {'Δ@4dB':>10} "
          f"{'Base@16dB':>10} {'+LN@16dB':>10} {'Δ@16dB':>10}")
    print("  " + "-" * 82)
    for m in _MODEL_NAMES:
        bkey = f"{m} (Baseline)"
        nkey = f"{m} (+InputLN)"
        if bkey not in results or nkey not in results:
            continue
        b4  = results[bkey][0][snr_4]
        n4  = results[nkey][0][snr_4]
        b16 = results[bkey][0][snr_16]
        n16 = results[nkey][0][snr_16]
        d4  = b4 - n4
        d16 = b16 - n16
        print(f"  {m:<18} {b4:>10.6f} {n4:>10.6f} {d4:>+10.6f} "
              f"{b16:>10.6f} {n16:>10.6f} {d16:>+10.6f}")


# ── Main ─────────────────────────────────────────────────────────────

def _constellation_config(constellation):
    """Return (output_activation, clip_range, training_modulation) for a constellation."""
    clip = get_clip_range(constellation)
    if constellation == "qam16":
        return "hardtanh", clip, "qam16"
    if constellation == "qpsk":
        return "tanh", clip, "qpsk"
    return "tanh", clip, "bpsk"


def _file_suffix(constellation):
    """Return filename suffix: '' for bpsk, '_qpsk' / '_qam16' otherwise."""
    return "" if constellation == "bpsk" else f"_{constellation}"


def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "layernorm")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("INPUT LAYERNORM EXPERIMENT — SEQUENCE RELAYS")
    print(f"Constellations: {', '.join(c.upper() for c in args.constellations)}")
    print("=" * 70)

    all_results = {}   # {(constellation, channel_name): results}

    for constellation in args.constellations:
        act, clip, tmod = _constellation_config(constellation)
        mod_tag = constellation.upper()
        suffix = _file_suffix(constellation)

        print(f"\n{'─' * 70}")
        print(f"  Constellation: {mod_tag}  |  activation={act}  |  training_mod={tmod}")
        print(f"{'─' * 70}")

        # ── Build classical baselines ────────────────────────────────
        print(f"\n[1/4] Building classical baselines ({mod_tag}) …")
        af = AmplifyAndForwardRelay(target_power=1.0)
        df = DecodeAndForwardRelay()

        # ── Train all sequence models (baseline + InputLN) ───────────
        print(f"\n[2/4] Training sequence relay variants ({mod_tag}) …")
        relays = {"AF": af, "DF": df}
        for model_name, cls, base_kw in MODEL_SPECS:
            r_base = _build_relay(
                model_name, cls, base_kw,
                use_input_norm=False, gpu=args.gpu, quick=args.quick,
                output_activation=act, clip_range=clip,
                training_modulation=tmod,
            )
            r_norm = _build_relay(
                model_name, cls, base_kw,
                use_input_norm=True,  gpu=args.gpu, quick=args.quick,
                output_activation=act, clip_range=clip,
                training_modulation=tmod,
            )
            r_both = _build_relay(
                model_name, cls, base_kw,
                use_input_norm=True,  gpu=args.gpu, quick=args.quick,
                output_activation="scaled_tanh", clip_range=clip,
                training_modulation=tmod,
                tag_override="+LN+Scaled"
            )
            relays[f"{model_name} (Baseline)"] = r_base
            relays[f"{model_name} (+InputLN)"] = r_norm
            relays[f"{model_name} (+LN+Scaled)"] = r_both

        # ── AWGN evaluation ──────────────────────────────────────────
        print(f"\n[3/4] Evaluating on {mod_tag} AWGN …")
        res_awgn = evaluate(relays, snr_range, args, "AWGN",
                            modulation=constellation)
        print_table(res_awgn, snr_range, f"{mod_tag} AWGN")
        plot_comparison(res_awgn, snr_range, "AWGN",
                        os.path.join(out_dir, f"layernorm_awgn{suffix}.png"),
                        modulation_tag=mod_tag)
        all_results[(constellation, "AWGN")] = res_awgn

        # ── Rayleigh evaluation ──────────────────────────────────────
        print(f"\n[4/4] Evaluating on {mod_tag} Rayleigh …")
        res_ray = evaluate(relays, snr_range, args, "Rayleigh",
                           channel_fn=rayleigh_fading_channel,
                           modulation=constellation)
        print_table(res_ray, snr_range, f"{mod_tag} Rayleigh")
        plot_comparison(res_ray, snr_range, "Rayleigh",
                        os.path.join(out_dir, f"layernorm_rayleigh{suffix}.png"),
                        modulation_tag=mod_tag)
        all_results[(constellation, "Rayleigh")] = res_ray

    # ── Cross-constellation summary charts (guidelines 19, 20) ───────
    if len(all_results) > 1:
        print("\n[Summary] Generating cross-constellation charts …")
        plot_summary_heatmap(all_results, snr_range, out_dir)
        plot_achievement(all_results, snr_range, out_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
