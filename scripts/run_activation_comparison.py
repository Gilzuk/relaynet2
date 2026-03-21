"""                                                                 
run_activation_comparison.py
=============================
Compare **sigmoid** vs **hardtanh** output activations across all relay
models on QPSK and 16-QAM constellations, optionally also comparing
the effect of **Input LayerNorm** on sequence models.

Both activations map to the same bounded range (±3/√10 ≈ ±0.9487):
  - hardtanh: hard-clipping (flat beyond bounds)
  - sigmoid:  smooth S-curve (never saturates)

Comparison variants (controlled by ``--compare-layernorm``):
  - Default:  sigmoid, hardtanh  (2 variants)
  - With LN:  sigmoid, hardtanh, sigmoid+LN, hardtanh+LN  (4 variants)

CGAN is excluded by default (slow adversarial training); use
``--include-cgan`` to add it.

Usage::

    python scripts/run_activation_comparison.py --gpu
    python scripts/run_activation_comparison.py --gpu --quick
    python scripts/run_activation_comparison.py --gpu --compare-layernorm
    python scripts/run_activation_comparison.py --gpu --constellations qpsk
    python scripts/run_activation_comparison.py --gpu --include-cgan
    python scripts/run_activation_comparison.py --gpu --include-normalized

Output:
    results/activation_comparison/<constellation>_activation_awgn.png
    results/activation_comparison/<constellation>_activation_rayleigh.png
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

try:
    from checkpoints.checkpoint_22_normalized_3k import build_all_3k
    _HAS_NORMALIZED = True
except Exception:
    build_all_3k = None
    _HAS_NORMALIZED = False


# ── Constants ────────────────────────────────────────────────────────

ACTIVATIONS = ("sigmoid", "hardtanh", "scaled_tanh")

VARIANT_STYLE = {
    "sigmoid":           {"ls": "-",  "alpha": 0.9},
    "hardtanh":          {"ls": "--", "alpha": 0.9},
    "scaled_tanh":       {"ls": "-.", "alpha": 0.9},
    "sigmoid+LN":        {"ls": (0, (3, 1, 1, 1)),       "alpha": 0.9},
    "hardtanh+LN":       {"ls": (0, (5, 2)),              "alpha": 0.9},
    "scaled_tanh+LN":    {"ls": ":",                      "alpha": 0.9},
}

COLORS = {
    "AF":          "gray",
    "DF":          "black",
    "GenAI":       "#e41a1c",
    "Hybrid":      "#377eb8",
    "VAE":         "#4daf4a",
    "CGAN":        "#984ea3",
    "Transformer": "#ff7f00",
    "Mamba S6":    "#a65628",
    "Mamba2":      "#f781bf",
}

COLORS_3K = {
    "GenAI-3K":       "#e41a1c",
    "Hybrid-3K":      "#377eb8",
    "VAE-3K":         "#4daf4a",
    "CGAN-3K":        "#984ea3",
    "Transformer-3K": "#ff7f00",
    "Mamba-3K":       "#a65628",
    "Mamba2-3K":      "#f781bf",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Sigmoid vs Hardtanh activation comparison for QPSK / QAM16"
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--compare-layernorm", action="store_true",
        help=(
            "Also compare with/without Input LayerNorm on sequence models. "
            "Doubles the number of variants (sigmoid, hardtanh, sigmoid+LN, hardtanh+LN)."
        ),
    )
    p.add_argument(
        "--include-cgan", action="store_true",
        help="Include CGAN (WGAN-GP). Excluded by default due to ~12× training overhead.",
    )
    p.add_argument("--include-normalized", action="store_true",
                   help="Also compare normalized ~3K-param models.")
    p.add_argument(
        "--constellations", type=str, nargs="+",
        default=["qpsk", "qam16"],
        help="Constellations to test (default: qpsk qam16)",
    )
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Relay factories ─────────────────────────────────────────────────

def _build_feedforward(activation, gpu, quick, include_cgan=False):
    samples = 25_000 if not quick else 5_000
    epochs = 100 if not quick else 20
    vae_samples = 25_000 if not quick else 5_000
    cgan_samples = 10_000 if not quick else 5_000
    cgan_epochs = 50 if not quick else 20

    # Both sigmoid and hardtanh train on QAM16 I/Q data
    train_mod = "qam16"

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

    if include_cgan:
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


def _build_sequence(activation, gpu, quick, layer_norm=False):
    if not _HAS_SEQ:
        return {}

    train_mod = "qam16"
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
                  output_activation=activation,
                  use_input_norm=layer_norm)
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


def _build_normalized(activation, gpu, quick, layer_norm=False,
                      include_cgan=False):
    if not _HAS_NORMALIZED:
        print("  [WARNING] Normalized 3K checkpoint not available; skipping.")
        return {}

    train_mod = "qam16"
    samples = 25_000 if not quick else 5_000
    epochs = 100 if not quick else 20
    seq_samples = 25_000 if not quick else 3_000
    seq_epochs = 50 if not quick else 10

    relays = build_all_3k(
        prefer_gpu=gpu,
        include_sequence_models=_HAS_SEQ,
        include_cgan=include_cgan,
        use_input_norm=layer_norm,
        output_activation=activation,
    )

    # Train each normalized model
    feed_cfg = dict(training_snrs=[5, 10, 15], num_samples=samples,
                    epochs=epochs, seed=42, training_modulation=train_mod)
    hybrid_cfg = dict(training_snrs=[2, 4, 6], num_samples=samples,
                      epochs=epochs, seed=42, training_modulation=train_mod)
    seq_cfg = dict(training_snrs=[5, 10, 15], num_samples=seq_samples,
                   epochs=seq_epochs, lr=0.001, training_modulation=train_mod)

    for name, relay in relays.items():
        print(f"    {name} ({relay.num_params}p) …", end=" ", flush=True)
        t0 = perf_counter()
        if "Hybrid" in name:
            relay.train(**hybrid_cfg)
        elif any(s in name for s in ("Transformer", "Mamba")):
            relay.train(**seq_cfg)
        else:
            relay.train(**feed_cfg)
        print(f"done ({perf_counter()-t0:.1f}s)")

    return relays


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(relays, snr_range, args, channel_name, modulation,
             channel_fn=None):
    results = {}
    for name, relay in relays.items():
        print(f"    {name} …", end=" ", flush=True)
        t0 = perf_counter()
        _, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            modulation=modulation,
        )
        elapsed = perf_counter() - t0
        print(f"done ({elapsed:.1f}s, BER [{ber.min():.2e}, {ber.max():.2e}])")
        results[name] = (ber, trials)
    return results


# ── Plotting ─────────────────────────────────────────────────────────

# Large distinguishable palette — enough for every (relay × variant) curve.
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#e6beff", "#ffe119", "#fabebe", "#7f7f7f", "#a9a9a9",
    "#008080", "#e41a1c", "#377eb8", "#ff7f00", "#984ea3",
    "#00ced1", "#ff1493", "#228b22", "#8b4513", "#4b0082",
]

_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "<", ">",
    "h", "p", "*", "H", "d", "8", "+", "x", "1",
    "2", "3", "4", "|", "_",
]


def plot_comparison(all_results, snr_range, constellation, channel_name,
                    out_dir, color_map=None, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(14, 9))

    # Build a flat list of all curves so we can assign unique visuals.
    curve_list = []  # [(variant, relay_name, ber)] — order matters
    baselines = []   # [(relay_name, ber)]
    first_variant = list(all_results.keys())[0]
    for baseline in ("AF", "DF"):
        if baseline in all_results[first_variant]:
            ber, _ = all_results[first_variant][baseline]
            baselines.append((baseline, ber))
    for variant, results in all_results.items():
        for relay_name, (ber, _) in results.items():
            if relay_name in ("AF", "DF"):
                continue
            curve_list.append((variant, relay_name, ber))

    # Assign unique (color, marker) per curve — no repeats.
    ci = 0  # palette index
    # Baselines get grey/black with thin lines and simple markers.
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

    # AI relay curves — each gets a unique colour + marker.
    for variant, relay_name, ber in curve_list:
        style = VARIANT_STYLE.get(variant, {"ls": "-", "alpha": 0.9})
        color = _PALETTE[ci % len(_PALETTE)]
        marker = _MARKERS[ci % len(_MARKERS)]
        ci += 1
        ax.semilogy(snr_range, np.clip(ber, 1e-5, 1),
                    ls=style["ls"], color=color, lw=1.0,
                    alpha=style["alpha"],
                    marker=marker, markevery=2, markersize=5,
                    label=f"{relay_name} [{variant}]")

    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title(
        f"Sigmoid vs Hardtanh — {channel_name} [{constellation.upper()}]{suffix}",
        fontsize=13,
    )
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_ylim(1e-4, 1)

    # ── Zoom inset around 12 dB ──────────────────────────────────
    snr_arr = np.asarray(snr_range)
    zoom_lo, zoom_hi = 10.0, 14.0
    mask = (snr_arr >= zoom_lo) & (snr_arr <= zoom_hi)
    if mask.sum() >= 2:
        axins = inset_axes(ax, width="35%", height="30%", loc="center right",
                           borderpad=2)
        # Re-draw all curves in the inset
        for name, ber in baselines:
            st = baseline_styles[name]
            axins.semilogy(snr_arr[mask], np.clip(ber[mask], 1e-5, 1),
                           ls=st["ls"], color=st["color"], lw=1.0,
                           marker=st["marker"], markersize=4, alpha=0.8)
        ci2 = 0
        for variant, relay_name, ber in curve_list:
            style = VARIANT_STYLE.get(variant, {"ls": "-", "alpha": 0.9})
            color = _PALETTE[ci2 % len(_PALETTE)]
            marker = _MARKERS[ci2 % len(_MARKERS)]
            ci2 += 1
            axins.semilogy(snr_arr[mask], np.clip(ber[mask], 1e-5, 1),
                           ls=style["ls"], color=color, lw=1.0,
                           alpha=style["alpha"],
                           marker=marker, markersize=4)
        axins.set_xlim(zoom_lo, zoom_hi)
        axins.grid(True, which="both", alpha=0.2, linewidth=0.4)
        axins.set_title("Zoom 10–14 dB", fontsize=8)
        axins.tick_params(labelsize=7)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",
                   lw=0.6, ls="--")

    # ── Legend outside the chart ─────────────────────────────────
    ax.legend(fontsize=7, ncol=2,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, framealpha=0.9)

    safe_ch = channel_name.lower().replace(" ", "_")
    tag = suffix.lower().replace(" ", "_").replace("[", "").replace("]", "")
    fname = os.path.join(
        out_dir, f"{constellation}_activation_{safe_ch}{tag}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {fname}")


# ── Summary table ────────────────────────────────────────────────────

def print_summary(all_results, snr_range, constellation, channel_name,
                  variants=None):
    if variants is None:
        variants = ACTIVATIONS
    snr_list = list(np.asarray(snr_range).tolist())
    idx_hi = snr_list.index(16.0) if 16.0 in snr_list else len(snr_list) - 1
    idx_lo = snr_list.index(4.0) if 4.0 in snr_list else min(2, len(snr_list) - 1)

    print(f"\n{'='*90}")
    print(f"  {constellation.upper()} BER — {channel_name}")
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
                row += f" {'—':>16} {'—':>16}"
        print(row)

    # Δ summary (sigmoid − hardtanh) — positive = sigmoid worse
    if "sigmoid" in all_results and "hardtanh" in all_results:
        print(f"\n  Δ BER (sigmoid − hardtanh):")
        for name in all_names:
            if name in ("AF", "DF"):
                continue
            if name in all_results["sigmoid"] and name in all_results["hardtanh"]:
                sig_ber, _ = all_results["sigmoid"][name]
                ht_ber, _ = all_results["hardtanh"][name]
                d_lo = sig_ber[idx_lo] - ht_ber[idx_lo]
                d_hi = sig_ber[idx_hi] - ht_ber[idx_hi]
                tag_lo = "sigmoid ↓" if d_lo < 0 else "hardtanh ↓"
                tag_hi = "sigmoid ↓" if d_hi < 0 else "hardtanh ↓"
                print(f"    {name:<18} @4dB: {d_lo:+.6f} ({tag_lo})  "
                      f"@16dB: {d_hi:+.6f} ({tag_hi})")

    if "sigmoid+LN" in all_results and "hardtanh+LN" in all_results:
        print(f"\n  Δ BER (sigmoid+LN − hardtanh+LN):")
        for name in all_names:
            if name in ("AF", "DF"):
                continue
            if name in all_results["sigmoid+LN"] and name in all_results["hardtanh+LN"]:
                sig_ber, _ = all_results["sigmoid+LN"][name]
                ht_ber, _ = all_results["hardtanh+LN"][name]
                d_lo = sig_ber[idx_lo] - ht_ber[idx_lo]
                d_hi = sig_ber[idx_hi] - ht_ber[idx_hi]
                tag_lo = "sigmoid ↓" if d_lo < 0 else "hardtanh ↓"
                tag_hi = "sigmoid ↓" if d_hi < 0 else "hardtanh ↓"
                print(f"    {name:<18} @4dB: {d_lo:+.6f} ({tag_lo})  "
                      f"@16dB: {d_hi:+.6f} ({tag_hi})")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)

    out_dir = os.path.join("results", "activation_comparison")
    os.makedirs(out_dir, exist_ok=True)

    t_global = perf_counter()

    # Build the list of variants to compare
    variants = list(ACTIVATIONS)  # ["sigmoid", "hardtanh"]
    if args.compare_layernorm:
        variants += [f"{a}+LN" for a in ACTIVATIONS]  # + ["sigmoid+LN", "hardtanh+LN"]

    for constellation in args.constellations:
        print(f"\n{'#'*70}")
        print(f"#  Constellation: {constellation.upper()}")
        print(f"{'#'*70}")

        results_awgn = {}
        results_rayleigh = {}

        for variant in variants:
            # Parse variant → activation name + layer-norm flag
            if variant.endswith("+LN"):
                act = variant[:-3]   # "sigmoid" or "hardtanh"
                use_ln = True
                label = f"{act.upper()}+LN"
            else:
                act = variant
                use_ln = False
                label = act.upper()

            print(f"\n{'='*60}")
            print(f"  Variant: {label} — {act} activation"
                  f"{' + Input LayerNorm' if use_ln else ''}")
            print(f"{'='*60}")

            # --- Build + train relays ---
            print(f"\n  Training feedforward relays [{variant}] …")
            relays = _build_feedforward(act, args.gpu, args.quick,
                                        include_cgan=args.include_cgan)

            print(f"\n  Training sequence models [{variant}] …")
            relays.update(_build_sequence(act, args.gpu, args.quick,
                                          layer_norm=use_ln))

            # Classical baselines
            relays["AF"] = AmplifyAndForwardRelay(prefer_gpu=False)
            relays["DF"] = DecodeAndForwardRelay(prefer_gpu=False)

            # --- Evaluate ---
            print(f"\n  Evaluating on {constellation.upper()} AWGN [{variant}] …")
            results_awgn[variant] = evaluate(
                relays, snr_range, args, "AWGN", constellation,
            )

            print(f"\n  Evaluating on {constellation.upper()} Rayleigh [{variant}] …")
            results_rayleigh[variant] = evaluate(
                relays, snr_range, args, "Rayleigh", constellation,
                channel_fn=rayleigh_fading_channel,
            )

        # --- Summary + Plots ---
        print_summary(results_awgn, snr_range, constellation, "AWGN",
                      variants=variants)
        print_summary(results_rayleigh, snr_range, constellation, "Rayleigh",
                      variants=variants)

        if _HAS_PLOTS:
            plot_comparison(results_awgn, snr_range, constellation,
                            "AWGN", out_dir)
            plot_comparison(results_rayleigh, snr_range, constellation,
                            "Rayleigh", out_dir)

        # --- Optional: normalized 3K comparison ---
        if args.include_normalized:
            print(f"\n{'='*60}")
            print(f"  Normalized ~3K-param comparison [{constellation.upper()}]")
            print(f"{'='*60}")

            norm_awgn = {}
            norm_rayleigh = {}

            for variant in variants:
                if variant.endswith("+LN"):
                    act = variant[:-3]
                    use_ln = True
                else:
                    act = variant
                    use_ln = False

                print(f"\n  Training normalized 3K [{variant}] …")
                relays_3k = _build_normalized(
                    act, args.gpu, args.quick, layer_norm=use_ln,
                    include_cgan=args.include_cgan,
                )
                if not relays_3k:
                    continue

                print(f"\n  Evaluating normalized 3K on {constellation.upper()} AWGN [{variant}] …")
                norm_awgn[variant] = evaluate(
                    relays_3k, snr_range, args, "AWGN", constellation,
                )

                print(f"\n  Evaluating normalized 3K on {constellation.upper()} Rayleigh [{variant}] …")
                norm_rayleigh[variant] = evaluate(
                    relays_3k, snr_range, args, "Rayleigh", constellation,
                    channel_fn=rayleigh_fading_channel,
                )

            if norm_awgn:
                print_summary(norm_awgn, snr_range, constellation,
                              "AWGN (3K)", variants=variants)
                print_summary(norm_rayleigh, snr_range, constellation,
                              "Rayleigh (3K)", variants=variants)
                if _HAS_PLOTS:
                    plot_comparison(norm_awgn, snr_range, constellation,
                                    "AWGN", out_dir, COLORS_3K, "_3k")
                    plot_comparison(norm_rayleigh, snr_range, constellation,
                                    "Rayleigh", out_dir, COLORS_3K, "_3k")

    elapsed = perf_counter() - t_global
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
