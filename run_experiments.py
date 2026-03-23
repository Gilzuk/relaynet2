#!/usr/bin/env python3
"""
run_experiments.py
==================
Unified experiment runner for the thesis:
    "Neural Network-Based Relay Processing for Wireless Communications"

Consolidates all 16 thesis experiments (§7.1–§7.16) into a single
reproducible script.  Every experiment:
  - Saves full BER results (mean, per-trial, 95 % CI) to JSON.
  - Generates publication-quality charts following CHART_GUIDELINES.md.
  - Loads / saves trained weights for later retrieval or re-running.

Usage examples
--------------
    python run_experiments.py --list                # show available experiments
    python run_experiments.py --all                 # run everything
    python run_experiments.py --exp 7.2 7.3         # run specific sections
    python run_experiments.py --exp 7.2 --quick     # fewer trials / epochs
    python run_experiments.py --all --inference-only # plots from saved weights
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from time import perf_counter

# Force UTF-8 on Windows
for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import compute_confidence_interval
from relaynet.channels.awgn import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel, rician_fading_channel
from relaynet.channels.mimo import (
    mimo_2x2_channel,
    mimo_2x2_mmse_channel,
    mimo_2x2_sic_channel,
)
from relaynet.utils.activations import get_clip_range

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper
    _HAS_SEQ = True
except Exception:
    _HAS_SEQ = False

try:
    from checkpoints.checkpoint_22_normalized_3k import build_all_3k
    _HAS_3K = True
except Exception:
    _HAS_3K = False

try:
    from checkpoints.checkpoint_24_e2e_transmitter import (
        E2ETransmitter,
        DifferentiableRayleighChannel,
        E2EReceiver,
        train_e2e,
        evaluate_ber,
        constellation_metrics,
    )
    _HAS_E2E = True
except Exception:
    _HAS_E2E = False

from scipy import special  # for erfc / Q-function in channel analysis

# ════════════════════════════════════════════════════════════════════
# Chart guidelines palette (30 colours, colorblind-safe)
# ════════════════════════════════════════════════════════════════════
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#e6beff", "#ffe119", "#fabebe", "#7f7f7f", "#a9a9a9",
    "#008080", "#e41a1c", "#377eb8", "#ff7f00", "#984ea3",
    "#00ced1", "#ff1493", "#228b22", "#8b4513", "#4b0082",
]

MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "<", ">",
    "h", "p", "*", "H", "d", "8", "+", "x",
]

# Fixed colour / marker for baselines
BASELINE_STYLE = {
    "AF":  {"color": "grey",  "marker": "o", "ls": "-"},
    "DF":  {"color": "black", "marker": "s", "ls": "-"},
}

RELAY_STYLE = {
    "GenAI (169p)":   {"color": PALETTE[0],  "marker": "^"},
    "Hybrid":         {"color": PALETTE[1],  "marker": "D"},
    "VAE":            {"color": PALETTE[2],  "marker": "v"},
    "CGAN (WGAN-GP)": {"color": PALETTE[3],  "marker": "P"},
    "Transformer":    {"color": PALETTE[4],  "marker": "X"},
    "Mamba S6":       {"color": PALETTE[5],  "marker": "<"},
    "Mamba2 (SSD)":   {"color": PALETTE[6],  "marker": ">"},
}


# ════════════════════════════════════════════════════════════════════
# JSON persistence helpers
# ════════════════════════════════════════════════════════════════════

def _to_serialisable(obj):
    """Make numpy types JSON-safe."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_results_json(path, snr_range, results_dict, meta=None):
    """Persist full experiment results to JSON.

    Parameters
    ----------
    path : str
    snr_range : array-like
    results_dict : dict
        ``{name: {"ber_mean": [...], "ber_trials": [[...]], "ci_lower": [...], "ci_upper": [...]}}``
    meta : dict, optional
        Extra metadata to include.
    """
    data = {
        "created": datetime.now().isoformat(),
        "snr_range": _to_serialisable(snr_range),
        "results": {},
    }
    if meta:
        data["meta"] = meta

    for name, rd in results_dict.items():
        entry = {}
        for k, v in rd.items():
            entry[k] = _to_serialisable(v)
        data["results"][name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_to_serialisable)
    print(f"  JSON → {path}")


def load_results_json(path):
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════
# Chart helpers (CHART_GUIDELINES.md compliant)
# ════════════════════════════════════════════════════════════════════

def _style_for(name, idx=0):
    """Return colour / marker / linestyle for a curve name."""
    if name in BASELINE_STYLE:
        return BASELINE_STYLE[name]
    if name in RELAY_STYLE:
        return {**RELAY_STYLE[name], "ls": "-"}
    # Fallback
    return {"color": PALETTE[idx % len(PALETTE)],
            "marker": MARKERS[idx % len(MARKERS)], "ls": "-"}


def plot_ber_chart(snr, ber_dict, ci_dict=None, title="", save_path=None,
                   ylim_bottom=None, extra_styles=None):
    """Publication-quality BER vs SNR chart.

    Follows CHART_GUIDELINES.md rules 1–16, 22.
    """
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    snr = np.asarray(snr)

    style_map = extra_styles or {}
    for idx, (name, ber) in enumerate(ber_dict.items()):
        ber = np.asarray(ber, dtype=float)
        st = style_map.get(name, _style_for(name, idx))
        color = st.get("color", PALETTE[idx % len(PALETTE)])
        marker = st.get("marker", MARKERS[idx % len(MARKERS)])
        ls = st.get("ls", "-")
        lw = st.get("lw", 1.3)
        alpha = st.get("alpha", 0.9)
        # Curve jitter for overlapping values (rule 5)
        ber_plot = np.where(ber > 0, ber, 1e-10)
        ax.semilogy(snr, ber_plot, marker=marker, color=color,
                     linewidth=lw, markersize=6, label=name,
                     linestyle=ls, alpha=alpha)
        if ci_dict and name in ci_dict:
            lo, hi = ci_dict[name]
            lo = np.maximum(np.asarray(lo, dtype=float), 1e-10)
            hi = np.asarray(hi, dtype=float)
            ax.fill_between(snr, lo, hi, alpha=0.15, color=color)

    # Y-axis: one decade below min nonzero BER (rule 6)
    all_ber = np.concatenate([np.asarray(b, dtype=float) for b in ber_dict.values()])
    min_ber = all_ber[all_ber > 0].min() if np.any(all_ber > 0) else 1e-6
    bottom = ylim_bottom or 10 ** (np.floor(np.log10(min_ber)) - 1)
    ax.set_ylim(bottom=max(bottom, 1e-8), top=1)

    ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot → {save_path}")
    plt.close(fig)


def plot_top3_chart(snr, results_dict, title="", save_path=None):
    """Top-3 neural relays + AF/DF baselines (rule 22)."""
    if not _HAS_MPL:
        return
    half = len(snr) // 2
    ranked = []
    for name, rd in results_dict.items():
        if name in ("AF", "DF"):
            continue
        avg_upper = float(np.mean(rd["ber_mean"][half:]))
        ranked.append((name, avg_upper))
    ranked.sort(key=lambda x: x[1])
    top3 = [r[0] for r in ranked[:3]]

    ber_dict = {}
    ci_dict = {}
    for name in ["AF", "DF"] + top3:
        if name in results_dict:
            ber_dict[name] = results_dict[name]["ber_mean"]
            if "ci_lower" in results_dict[name]:
                ci_dict[name] = (results_dict[name]["ci_lower"],
                                 results_dict[name]["ci_upper"])
    plot_ber_chart(snr, ber_dict, ci_dict, title=title, save_path=save_path)


# ════════════════════════════════════════════════════════════════════
# Weight management
# ════════════════════════════════════════════════════════════════════

class WeightManager:
    """Save / load relay weights for reproducibility."""

    def __init__(self, base_dir="weights", seed=42):
        self.base_dir = os.path.join(base_dir, f"seed_{seed}")
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, name, subdir=None):
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        d = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{safe}.pt")

    def save(self, name, relay, subdir=None):
        if hasattr(relay, "save_weights"):
            p = self._path(name, subdir)
            relay.save_weights(p)
            print(f"    Weights → {p}")

    def load(self, name, relay, subdir=None):
        p = self._path(name, subdir)
        if os.path.exists(p) and hasattr(relay, "load_weights"):
            relay.load_weights(p)
            return True
        return False

    def save_metadata(self, meta, subdir=None):
        d = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "metadata.json")
        with open(p, "w") as f:
            json.dump(meta, f, indent=2, default=_to_serialisable)


# ════════════════════════════════════════════════════════════════════
# Relay factory
# ════════════════════════════════════════════════════════════════════

def build_base_relays(gpu=False, activation="tanh", clip_range=None,
                      layer_norm=False):
    """Build the 9-relay dictionary (untrained)."""
    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": MinimalGenAIRelay(
            window_size=5, hidden_size=24, prefer_gpu=False,
            output_activation=activation, clip_range=clip_range),
        "Hybrid": HybridRelay(
            snr_threshold=5.0, prefer_gpu=False,
            output_activation=activation, clip_range=clip_range),
        "VAE": VAERelay(
            window_size=7, latent_size=8, beta=0.1, prefer_gpu=False,
            output_activation=activation, clip_range=clip_range),
        "CGAN (WGAN-GP)": CGANRelay(
            window_size=7, noise_size=8, lambda_gp=10, lambda_l1=20,
            n_critic=5, prefer_gpu=gpu,
            output_activation=activation, clip_range=clip_range),
    }
    if _HAS_SEQ:
        relays["Transformer"] = TransformerRelayWrapper(
            target_power=1.0, window_size=11, d_model=32,
            num_heads=4, num_layers=2, prefer_gpu=gpu,
            use_input_norm=layer_norm,
            output_activation=activation, clip_range=clip_range)
        relays["Mamba S6"] = MambaRelayWrapper(
            target_power=1.0, window_size=11, d_model=32,
            d_state=16, num_layers=2, prefer_gpu=gpu,
            use_input_norm=layer_norm,
            output_activation=activation, clip_range=clip_range)
        relays["Mamba2 (SSD)"] = Mamba2RelayWrapper(
            target_power=1.0, window_size=11, d_model=32,
            d_state=16, num_layers=2, prefer_gpu=gpu,
            use_input_norm=layer_norm,
            output_activation=activation, clip_range=clip_range)
    return relays


def train_base_relays(relays, args, modulation="bpsk"):
    """Train all AI relays with standard parameters."""
    samples = 5_000 if args.quick else 25_000
    epochs = 20 if args.quick else 100
    vae_samples = 5_000 if args.quick else 50_000
    cgan_samples = 5_000 if args.quick else 50_000
    cgan_epochs = 20 if args.quick else 200
    seq_samples = 3_000 if args.quick else 50_000
    seq_epochs = 10 if args.quick else 100

    for name, relay in relays.items():
        if name in ("AF", "DF"):
            continue
        print(f"  Training {name} …", end=" ", flush=True)
        t0 = perf_counter()
        kw = {"seed": args.seed}

        if name == "GenAI (169p)":
            relay.train(training_snrs=[5, 10, 15], num_samples=samples,
                        epochs=epochs, **kw)
        elif name == "Hybrid":
            relay.train(training_snrs=[2, 4, 6], num_samples=samples,
                        epochs=epochs, **kw)
        elif name == "VAE":
            relay.train(training_snrs=[5, 10, 15], num_samples=vae_samples,
                        epochs=epochs, **kw)
        elif name == "CGAN (WGAN-GP)":
            relay.train(training_snrs=[5, 10, 15], num_samples=cgan_samples,
                        epochs=cgan_epochs, **kw)
        elif name in ("Transformer", "Mamba S6", "Mamba2 (SSD)"):
            relay.train(training_snrs=[5, 10, 15], num_samples=seq_samples,
                        epochs=seq_epochs, lr=0.001,
                        training_modulation=modulation)
        print(f"done ({perf_counter() - t0:.1f}s)")


# ════════════════════════════════════════════════════════════════════
# Monte Carlo evaluation helper
# ════════════════════════════════════════════════════════════════════

def evaluate_relays(relays, snr_range, channel_fn=None,
                    modulation="bpsk",
                    bits_per_trial=10_000, num_trials=10):
    """Run Monte Carlo BER evaluation for all relays.

    Returns dict ``{name: {"ber_mean": [...], "ber_trials": [[...]], ...}}``.
    """
    results = {}
    for name, relay in relays.items():
        print(f"    {name} …", end=" ", flush=True)
        t0 = perf_counter()
        snr, ber, trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=bits_per_trial,
            num_trials=num_trials,
            channel_fn=channel_fn,
            modulation=modulation,
        )
        ci_lo, ci_hi = compute_confidence_interval(trials)
        results[name] = {
            "ber_mean": ber,
            "ber_trials": trials,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }
        print(f"done ({perf_counter() - t0:.1f}s)")
    return results


# ════════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS — one per thesis section
# ════════════════════════════════════════════════════════════════════

# ── §7.1  Channel Model Analysis ───────────────────────────────────

def exp_7_1_channel_analysis(args):
    """§7.1 Theoretical vs simulative channel BER curves."""
    print("\n══ §7.1 Channel Model Analysis ══")
    out = os.path.join(args.results_dir, "channel_analysis")
    os.makedirs(out, exist_ok=True)
    snr_db = np.arange(0, 22, 2, dtype=float)
    snr_lin = 10 ** (snr_db / 10)
    bits = 5_000 if args.quick else 50_000
    trials = 5 if args.quick else 20

    def Q(x):
        return 0.5 * special.erfc(x / np.sqrt(2))

    # Theoretical curves
    ber_awgn_th = Q(np.sqrt(2 * snr_lin))
    ber_ray_th = 0.5 * (1 - np.sqrt(snr_lin / (1 + snr_lin)))
    ber_ric_th = []
    for g in snr_lin:
        K = 3.0
        ber_ric_th.append(0.5 * (1 - np.sqrt(g / (1 + g)) *
                                  (1 + K / (g + K))))
    ber_ric_th = np.clip(ber_ric_th, 1e-10, 1)

    # Two-hop theoretical
    ber_df_th = ber_awgn_th + ber_awgn_th - 2 * ber_awgn_th * ber_awgn_th
    snr_af_eff = snr_lin ** 2 / (2 * snr_lin + 1)
    ber_af_th = Q(np.sqrt(2 * snr_af_eff))

    # Simulative
    from relaynet.nodes import Source, Destination
    from relaynet.modulation.bpsk import bpsk_modulate, bpsk_demodulate, calculate_ber

    sim_results = {}
    for ch_name, ch_fn in [("AWGN", None),
                            ("Rayleigh", rayleigh_fading_channel),
                            ("Rician K=3",
                             lambda s, snr: rician_fading_channel(s, snr, k_factor=3.0))]:
        print(f"  Simulating {ch_name} …")
        bers = np.zeros(len(snr_db))
        for i, snr in enumerate(snr_db):
            trial_bers = []
            for t in range(trials):
                src = Source(seed=t, modulation="bpsk")
                dst = Destination(modulation="bpsk")
                tx_bits, tx_sym = src.transmit(bits)
                if ch_fn:
                    rx = ch_fn(tx_sym, snr)
                else:
                    rx = awgn_channel(tx_sym, snr)
                if isinstance(rx, tuple):
                    rx = rx[0]
                rx_bits = dst.receive(rx)
                b, _ = calculate_ber(tx_bits, rx_bits)
                trial_bers.append(b)
            bers[i] = np.mean(trial_bers)
        sim_results[ch_name] = bers

    # Save JSON
    save_results_json(
        os.path.join(out, "channel_analysis.json"),
        snr_db,
        {
            "AWGN_theoretical": {"ber_mean": ber_awgn_th},
            "AWGN_simulated": {"ber_mean": sim_results.get("AWGN", ber_awgn_th)},
            "Rayleigh_theoretical": {"ber_mean": ber_ray_th},
            "Rayleigh_simulated": {"ber_mean": sim_results.get("Rayleigh", ber_ray_th)},
            "Rician_theoretical": {"ber_mean": ber_ric_th},
            "Rician_simulated": {"ber_mean": sim_results.get("Rician K=3", ber_ric_th)},
            "TwoHop_DF_theoretical": {"ber_mean": ber_df_th},
            "TwoHop_AF_theoretical": {"ber_mean": ber_af_th},
        },
        meta={"experiment": "7.1", "bits": bits, "trials": trials},
    )

    if not _HAS_MPL:
        return

    # Individual channel plots
    for tag, th, sim_key in [
        ("awgn", ber_awgn_th, "AWGN"),
        ("rayleigh", ber_ray_th, "Rayleigh"),
        ("rician", ber_ric_th, "Rician K=3"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(snr_db, th, "k-", linewidth=1.3, label=f"{sim_key} (theory)")
        ax.semilogy(snr_db, sim_results[sim_key], "ro", markersize=5,
                     label=f"{sim_key} (sim)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_xlabel("SNR (dB)", fontsize=14)
        ax.set_ylabel("BER", fontsize=14)
        ax.set_title(f"Theoretical vs Simulated — {sim_key}", fontsize=16)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"channel_theoretical_{tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary plot: all SISO channels
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr_db, ber_awgn_th, "k-", lw=1.3, label="AWGN")
    ax.semilogy(snr_db, ber_ray_th, "b--", lw=1.3, label="Rayleigh")
    ax.semilogy(snr_db, np.array(ber_ric_th), "g-.", lw=1.3, label="Rician K=3")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_xlabel("SNR (dB)", fontsize=14); ax.set_ylabel("BER", fontsize=14)
    ax.set_title("SISO Channel Comparison", fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "channel_comparison_all.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  §7.1 complete → {out}/")


# ── §7.2–§7.7  BPSK relay comparisons on 6 channels ───────────────

_CHANNELS = {
    "awgn":         ("AWGN",             None),
    "rayleigh":     ("Rayleigh Fading",  rayleigh_fading_channel),
    "rician_k3":    ("Rician K=3",       lambda s, snr: rician_fading_channel(s, snr, k_factor=3.0)),
    "mimo_zf":      ("2×2 MIMO ZF",      mimo_2x2_channel),
    "mimo_mmse":    ("2×2 MIMO MMSE",    mimo_2x2_mmse_channel),
    "mimo_sic":     ("2×2 MIMO SIC",     mimo_2x2_sic_channel),
}

_SECTION_MAP = {
    "awgn":      "7.2",
    "rayleigh":  "7.3",
    "rician_k3": "7.4",
    "mimo_zf":   "7.5",
    "mimo_mmse": "7.6",
    "mimo_sic":  "7.7",
}


def exp_7_2_to_7_7_relay_comparison(args):
    """§7.2–§7.7 BPSK relay comparison on 6 channels."""
    print("\n══ §7.2–§7.7 BPSK Relay Comparison ══")
    wm = WeightManager(args.weights_dir, args.seed)
    relays = build_base_relays(gpu=args.gpu)
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)

    if args.inference_only:
        for name, relay in relays.items():
            wm.load(name, relay)
    else:
        train_base_relays(relays, args, modulation="bpsk")
        for name, relay in relays.items():
            wm.save(name, relay)
        wm.save_metadata({
            "seed": args.seed, "date": datetime.now().isoformat(),
            "modulation": "bpsk", "quick": args.quick,
        })

    out = os.path.join(args.results_dir, "bpsk_comparison")
    os.makedirs(out, exist_ok=True)

    for ch_key, (ch_name, ch_fn) in _CHANNELS.items():
        sec = _SECTION_MAP[ch_key]
        print(f"\n  §{sec} — {ch_name}")
        results = evaluate_relays(
            relays, snr_range, channel_fn=ch_fn,
            modulation="bpsk",
            bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
        )
        save_results_json(
            os.path.join(out, f"{ch_key}.json"), snr_range, results,
            meta={"experiment": sec, "channel": ch_name, "modulation": "bpsk"},
        )
        ber_dict = {n: r["ber_mean"] for n, r in results.items()}
        ci_dict = {n: (r["ci_lower"], r["ci_upper"]) for n, r in results.items()}
        plot_ber_chart(
            snr_range, ber_dict, ci_dict,
            title=f"{ch_name} — BPSK Relay Comparison (§{sec})",
            save_path=os.path.join(out, f"{ch_key}_ci.png"),
        )
        plot_top3_chart(
            snr_range, results,
            title=f"Top-3 Neural Relays — {ch_name} BPSK",
            save_path=os.path.join(out, f"{ch_key}_top3.png"),
        )

    print(f"\n  §7.2–§7.7 complete → {out}/")


# ── §7.8  Normalized 3K comparison ─────────────────────────────────

def exp_7_8_normalized_3k(args):
    """§7.8 Normalized ~3K-parameter relay comparison."""
    print("\n══ §7.8 Normalized 3K-Parameter Comparison ══")
    if not _HAS_3K:
        print("  [SKIP] checkpoint_22_normalized_3k not available")
        return

    wm = WeightManager(args.weights_dir, args.seed)
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "normalized_3k")
    os.makedirs(out, exist_ok=True)

    relays_3k = build_all_3k(prefer_gpu=args.gpu)
    # Add baselines
    relays_3k["AF"] = AmplifyAndForwardRelay()
    relays_3k["DF"] = DecodeAndForwardRelay()

    if not args.inference_only:
        epochs = 10 if args.quick else 100
        samples = 3_000 if args.quick else 25_000
        for name, relay in relays_3k.items():
            if name in ("AF", "DF"):
                continue
            if wm.load(name, relay, subdir="3k"):
                print(f"  Loaded {name}")
                continue
            print(f"  Training {name} …", end=" ", flush=True)
            t0 = perf_counter()
            # Sequence model wrappers don't accept seed=
            is_seq = any(tag in name for tag in ("Transformer", "Mamba"))
            kw = {} if is_seq else {"seed": args.seed}
            relay.train(training_snrs=[5, 10, 15], num_samples=samples,
                        epochs=epochs, **kw)
            wm.save(name, relay, subdir="3k")
            print(f"done ({perf_counter() - t0:.1f}s)")
    else:
        for name, relay in relays_3k.items():
            wm.load(name, relay, subdir="3k")

    for ch_key, (ch_name, ch_fn) in _CHANNELS.items():
        print(f"  {ch_name} …")
        results = evaluate_relays(
            relays_3k, snr_range, channel_fn=ch_fn,
            bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
        )
        save_results_json(
            os.path.join(out, f"3k_{ch_key}.json"), snr_range, results,
            meta={"experiment": "7.8", "channel": ch_name},
        )
        ber_dict = {n: r["ber_mean"] for n, r in results.items()}
        ci_dict = {n: (r["ci_lower"], r["ci_upper"]) for n, r in results.items()}
        plot_ber_chart(
            snr_range, ber_dict, ci_dict,
            title=f"Normalized 3K — {ch_name} (§7.8)",
            save_path=os.path.join(out, f"3k_{ch_key}.png"),
        )

    print(f"  §7.8 complete → {out}/")


# ── §7.9  Master 2×3 comparison chart ──────────────────────────────

def exp_7_9_master_chart(args):
    """§7.9 Master 2×3 summary chart from §7.2–§7.7 results."""
    print("\n══ §7.9 Master 2×3 Comparison Chart ══")
    if not _HAS_MPL:
        print("  [SKIP] matplotlib not available")
        return
    bpsk_dir = os.path.join(args.results_dir, "bpsk_comparison")
    panels = [
        ("awgn", "AWGN"), ("rayleigh", "Rayleigh"), ("rician_k3", "Rician K=3"),
        ("mimo_zf", "MIMO ZF"), ("mimo_mmse", "MIMO MMSE"), ("mimo_sic", "MIMO SIC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (ch_key, ch_name) in zip(axes.flat, panels):
        jpath = os.path.join(bpsk_dir, f"{ch_key}.json")
        if not os.path.exists(jpath):
            ax.set_title(f"{ch_name} (no data)")
            continue
        data = load_results_json(jpath)
        snr = np.array(data["snr_range"])
        for idx, (name, rd) in enumerate(data["results"].items()):
            ber = np.array(rd["ber_mean"])
            ber = np.where(ber > 0, ber, 1e-10)
            st = _style_for(name, idx)
            ax.semilogy(snr, ber, marker=st["marker"], color=st["color"],
                         linewidth=1.0, markersize=4, label=name,
                         linestyle=st.get("ls", "-"), alpha=0.85)
        ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_title(ch_name, fontsize=13)
        ax.set_xlabel("SNR (dB)", fontsize=11)
        ax.set_ylabel("BER", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Master BER Comparison — 9 Relays × 6 Channels (§7.9)", fontsize=16)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_path = os.path.join(args.results_dir, "master_ber_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  §7.9 complete → {save_path}")


# ── §7.10  Modulation comparison ───────────────────────────────────

def exp_7_10_modulation_comparison(args):
    """§7.10 BPSK / QPSK / 16-QAM relay comparison."""
    print("\n══ §7.10 Modulation Comparison ══")
    wm = WeightManager(args.weights_dir, args.seed)
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "modulation")
    os.makedirs(out, exist_ok=True)

    # Train once on BPSK
    relays = build_base_relays(gpu=args.gpu)
    if args.inference_only:
        for name, relay in relays.items():
            wm.load(name, relay)
    else:
        train_base_relays(relays, args, modulation="bpsk")

    modulations = ["bpsk", "qpsk", "qam16"]
    channels = [("awgn", None), ("rayleigh", rayleigh_fading_channel)]

    all_results = {}
    for mod in modulations:
        for ch_key, ch_fn in channels:
            tag = f"{mod}_{ch_key}"
            print(f"\n  {mod.upper()} × {ch_key.upper()}")
            results = evaluate_relays(
                relays, snr_range, channel_fn=ch_fn,
                modulation=mod,
                bits_per_trial=args.bits_per_trial,
                num_trials=args.num_trials,
            )
            all_results[tag] = results
            save_results_json(
                os.path.join(out, f"{tag}.json"), snr_range, results,
                meta={"experiment": "7.10", "modulation": mod, "channel": ch_key},
            )
            ber_dict = {n: r["ber_mean"] for n, r in results.items()}
            ci_dict = {n: (r["ci_lower"], r["ci_upper"]) for n, r in results.items()}
            plot_ber_chart(
                snr_range, ber_dict, ci_dict,
                title=f"{mod.upper()} — {ch_key.upper()} (§7.10)",
                save_path=os.path.join(out, f"{tag}_ci.png"),
            )

    print(f"\n  §7.10 complete → {out}/")


# ── §7.11  QAM16 activation study ─────────────────────────────────

def exp_7_11_qam16_activation(args):
    """§7.11 QAM16 activation: tanh (BPSK-trained) vs linear vs hardtanh."""
    print("\n══ §7.11 QAM16 Activation Study ══")
    wm = WeightManager(args.weights_dir, args.seed)
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "qam16_activation")
    os.makedirs(out, exist_ok=True)
    clip_range = get_clip_range("qam16")

    activations = {
        "tanh": {"act": "tanh", "cr": None, "mod": "bpsk"},
        "linear": {"act": "linear", "cr": None, "mod": "qam16"},
        "hardtanh": {"act": "hardtanh", "cr": clip_range, "mod": "qam16"},
    }

    for ch_key, ch_fn in [("awgn", None), ("rayleigh", rayleigh_fading_channel)]:
        print(f"\n  Channel: {ch_key.upper()}")
        all_ber = {}
        all_ci = {}
        for act_name, cfg in activations.items():
            print(f"    Activation: {act_name}")
            relays = build_base_relays(
                gpu=args.gpu, activation=cfg["act"], clip_range=cfg["cr"])
            if not args.inference_only:
                train_base_relays(relays, args, modulation=cfg["mod"])
            results = evaluate_relays(
                relays, snr_range, channel_fn=ch_fn,
                modulation="qam16",
                bits_per_trial=args.bits_per_trial,
                num_trials=args.num_trials,
            )
            for rname, rd in results.items():
                label = f"{rname} ({act_name})"
                all_ber[label] = rd["ber_mean"]
                all_ci[label] = (rd["ci_lower"], rd["ci_upper"])

        save_results_json(
            os.path.join(out, f"qam16_activation_{ch_key}.json"),
            snr_range,
            {k: {"ber_mean": v} for k, v in all_ber.items()},
            meta={"experiment": "7.11", "channel": ch_key},
        )
        plot_ber_chart(
            snr_range, all_ber, all_ci,
            title=f"QAM16 Activation Comparison — {ch_key.upper()} (§7.11)",
            save_path=os.path.join(out, f"qam16_activation_{ch_key}.png"),
        )

    print(f"  §7.11 complete → {out}/")


# ── §7.12  LayerNorm study ─────────────────────────────────────────

def exp_7_12_layernorm(args):
    """§7.12 Input LayerNorm impact on sequence models."""
    print("\n══ §7.12 LayerNorm Study ══")
    if not _HAS_SEQ:
        print("  [SKIP] Sequence model checkpoints unavailable")
        return

    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "layernorm")
    os.makedirs(out, exist_ok=True)

    variants = {
        "Baseline": {"ln": False, "act": "tanh"},
        "+InputLN": {"ln": True, "act": "tanh"},
        "+LN+Scaled": {"ln": True, "act": "scaled_tanh"},
    }

    seq_models = ["Transformer", "Mamba S6", "Mamba2 (SSD)"]
    modulations = ["qpsk", "qam16"]
    channels = [("awgn", None), ("rayleigh", rayleigh_fading_channel)]

    for mod in modulations:
        act_default = "tanh" if mod != "qam16" else "hardtanh"
        cr = get_clip_range(mod) if mod == "qam16" else None

        for ch_key, ch_fn in channels:
            print(f"\n  {mod.upper()} × {ch_key.upper()}")
            all_ber = {}
            all_ci = {}

            for var_name, cfg in variants.items():
                act = cfg["act"] if mod != "qam16" else "hardtanh"
                relays = build_base_relays(
                    gpu=args.gpu, activation=act, clip_range=cr,
                    layer_norm=cfg["ln"])

                if not args.inference_only:
                    subdir = f"layernorm/{var_name.strip('+').lower()}"
                    samples = 5_000 if args.quick else 50_000
                    epochs = 10 if args.quick else 100
                    for name in seq_models:
                        if name not in relays:
                            continue
                        print(f"    {name} {var_name} …", end=" ", flush=True)
                        t0 = perf_counter()
                        relays[name].train(
                            training_snrs=[5, 10, 15], num_samples=samples,
                            epochs=epochs, lr=0.001,
                            training_modulation=mod)
                        print(f"done ({perf_counter() - t0:.1f}s)")

                # Evaluate only sequence models + baselines
                eval_relays = {n: relays[n] for n in ["AF", "DF"] + seq_models
                               if n in relays}
                results = evaluate_relays(
                    eval_relays, snr_range, channel_fn=ch_fn,
                    modulation=mod,
                    bits_per_trial=args.bits_per_trial,
                    num_trials=args.num_trials,
                )
                for rname, rd in results.items():
                    if rname in ("AF", "DF") and var_name != "Baseline":
                        continue  # Only add baselines once
                    label = f"{rname} {var_name}" if rname not in ("AF", "DF") else rname
                    all_ber[label] = rd["ber_mean"]
                    all_ci[label] = (rd["ci_lower"], rd["ci_upper"])

            tag = f"{mod}_{ch_key}"
            save_results_json(
                os.path.join(out, f"layernorm_{tag}.json"),
                snr_range,
                {k: {"ber_mean": v} for k, v in all_ber.items()},
                meta={"experiment": "7.12", "modulation": mod, "channel": ch_key},
            )
            plot_ber_chart(
                snr_range, all_ber, all_ci,
                title=f"LayerNorm Study — {mod.upper()} {ch_key.upper()} (§7.12)",
                save_path=os.path.join(out, f"layernorm_{tag}.png"),
            )

    print(f"  §7.12 complete → {out}/")


# ── §7.13  Activation comparison ──────────────────────────────────

def exp_7_13_activation_comparison(args):
    """§7.13 Sigmoid vs hardtanh vs scaled_tanh across all relays."""
    print("\n══ §7.13 Activation Comparison ══")
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "activation_comparison")
    os.makedirs(out, exist_ok=True)

    activations = ["sigmoid", "hardtanh", "scaled_tanh"]
    act_styles = {
        "sigmoid": "-", "hardtanh": "--", "scaled_tanh": "-.",
    }
    modulations = ["qpsk", "qam16"]
    channels = [("awgn", None), ("rayleigh", rayleigh_fading_channel)]

    for mod in modulations:
        cr = get_clip_range(mod) if mod == "qam16" else None
        for ch_key, ch_fn in channels:
            print(f"\n  {mod.upper()} × {ch_key.upper()}")
            all_ber = {}
            all_ci = {}
            extra_styles = {}
            cidx = 0

            for act in activations:
                relays = build_base_relays(gpu=args.gpu, activation=act,
                                           clip_range=cr)
                if not args.inference_only:
                    train_base_relays(relays, args, modulation="qam16")

                results = evaluate_relays(
                    relays, snr_range, channel_fn=ch_fn,
                    modulation=mod,
                    bits_per_trial=args.bits_per_trial,
                    num_trials=args.num_trials,
                )
                for rname, rd in results.items():
                    label = f"{rname} ({act})"
                    all_ber[label] = rd["ber_mean"]
                    all_ci[label] = (rd["ci_lower"], rd["ci_upper"])
                    extra_styles[label] = {
                        "color": PALETTE[cidx % len(PALETTE)],
                        "marker": MARKERS[cidx % len(MARKERS)],
                        "ls": act_styles[act],
                    }
                    cidx += 1

            tag = f"{mod}_{ch_key}"
            save_results_json(
                os.path.join(out, f"activation_{tag}.json"),
                snr_range,
                {k: {"ber_mean": v} for k, v in all_ber.items()},
                meta={"experiment": "7.13", "modulation": mod, "channel": ch_key},
            )
            plot_ber_chart(
                snr_range, all_ber, all_ci,
                title=f"Activation Comparison — {mod.upper()} {ch_key.upper()} (§7.13)",
                save_path=os.path.join(out, f"{mod}_activation_{ch_key}.png"),
                extra_styles=extra_styles,
            )

    print(f"  §7.13 complete → {out}/")


# ── §7.14  CSI injection (single architecture) ────────────────────

def _build_csi_variant(model_name, cfg, gpu, constellation):
    """Build a single CSI experiment variant."""
    cr = get_clip_range(constellation)
    in_ch = 2 if cfg["csi"] else 1
    base_kw = dict(target_power=1.0, window_size=11, d_model=32,
                   num_layers=2, clip_range=cr, in_channels=in_ch,
                   use_input_norm=cfg["ln"],
                   output_activation=cfg["act"])
    if model_name == "Mamba S6":
        return MambaRelayWrapper(d_state=16, prefer_gpu=gpu, **base_kw)
    elif model_name == "Transformer":
        return TransformerRelayWrapper(num_heads=4, prefer_gpu=gpu, **base_kw)
    elif model_name == "Mamba2 (SSD)":
        return Mamba2RelayWrapper(d_state=16, prefer_gpu=gpu, **base_kw)
    return None


def exp_7_14_csi_injection(args):
    """§7.14 CSI injection experiment."""
    print("\n══ §7.14 CSI Injection Experiment ══")
    if not _HAS_SEQ:
        print("  [SKIP] Sequence model checkpoints unavailable")
        return

    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    out = os.path.join(args.results_dir, "csi")
    os.makedirs(out, exist_ok=True)

    constellations = ["qam16", "psk16"]
    model_names = ["Mamba S6", "Transformer", "Mamba2 (SSD)"]
    activations = ["hardtanh", "scaled_tanh", "tanh", "sigmoid"]
    configs = {
        "Baseline": {"csi": False, "ln": False},
        "LN":       {"csi": False, "ln": True},
        "CSI":      {"csi": True,  "ln": False},
        "CSI+LN":   {"csi": True,  "ln": True},
    }

    train_samples = 1_000 if args.quick else 20_000
    train_epochs = 5 if args.quick else 25

    ch_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)

    for constellation in constellations:
        print(f"\n  Constellation: {constellation.upper()}")
        all_results = {"AF": None, "DF": None}
        cr = get_clip_range(constellation)

        # Evaluate AF/DF baselines
        af = AmplifyAndForwardRelay()
        df = DecodeAndForwardRelay()
        for bname, relay in [("AF", af), ("DF", df)]:
            print(f"    {bname} …", end=" ", flush=True)
            _, ber, trials = run_monte_carlo(
                relay, snr_range, channel_fn=ch_fn,
                num_bits_per_trial=args.bits_per_trial,
                num_trials=args.num_trials,
                modulation=constellation)
            ci_lo, ci_hi = compute_confidence_interval(trials)
            all_results[bname] = {
                "ber_mean": ber, "ber_trials": trials,
                "ci_lower": ci_lo, "ci_upper": ci_hi,
            }
            print("done")

        # Train and evaluate all variants
        for model_name in model_names:
            for act in activations:
                for cfg_name, cfg_vals in configs.items():
                    label = f"{model_name} {cfg_name} {act}"
                    cfg = {**cfg_vals, "act": act}
                    print(f"    {label} …", end=" ", flush=True)
                    t0 = perf_counter()

                    relay = _build_csi_variant(model_name, cfg, args.gpu,
                                               constellation)
                    if relay is None:
                        print("SKIP")
                        continue

                    if not args.inference_only:
                        relay.train(
                            training_snrs=[5, 10, 15],
                            num_samples=train_samples,
                            epochs=train_epochs,
                            training_modulation=constellation,
                            use_rayleigh=True)

                    _, ber, trials = run_monte_carlo(
                        relay, snr_range, channel_fn=ch_fn,
                        num_bits_per_trial=args.bits_per_trial,
                        num_trials=args.num_trials,
                        modulation=constellation)
                    ci_lo, ci_hi = compute_confidence_interval(trials)
                    all_results[label] = {
                        "ber_mean": ber, "ber_trials": trials,
                        "ci_lower": ci_lo, "ci_upper": ci_hi,
                    }
                    print(f"done ({perf_counter() - t0:.1f}s)")

        # Save results
        save_results_json(
            os.path.join(out, f"csi_experiment_{constellation}_rayleigh.json"),
            snr_range, all_results,
            meta={"experiment": "7.14-7.15", "constellation": constellation},
        )

        # Plot all variants
        ber_dict = {n: r["ber_mean"] for n, r in all_results.items()}
        ci_dict = {n: (r["ci_lower"], r["ci_upper"]) for n, r in all_results.items()}
        plot_ber_chart(
            snr_range, ber_dict, ci_dict,
            title=f"CSI Experiment — {constellation.upper()} Rayleigh (§7.14–7.15)",
            save_path=os.path.join(out, f"csi_experiment_{constellation}_rayleigh.png"),
        )
        plot_top3_chart(
            snr_range, all_results,
            title=f"Top-3 CSI Variants — {constellation.upper()} Rayleigh",
            save_path=os.path.join(out, f"top3_{constellation}_rayleigh.png"),
        )

    print(f"  §7.14–7.15 complete → {out}/")


# ── §7.15  Multi-architecture CSI (uses same function as §7.14) ───

def exp_7_15_multi_csi(args):
    """§7.15 Multi-architecture CSI — results generated by §7.14."""
    print("\n══ §7.15 Multi-Architecture CSI ══")
    print("  Results combined with §7.14 (run --exp 7.14 to generate)")


# ── §7.16  E2E autoencoder ─────────────────────────────────────────

def exp_7_16_e2e(args):
    """§7.16 End-to-End autoencoder experiment."""
    print("\n══ §7.16 End-to-End Autoencoder ══")
    if not _HAS_E2E:
        print("  [SKIP] checkpoint_24_e2e_transmitter not available")
        return

    import torch
    out = os.path.join(args.results_dir, "e2e")
    os.makedirs(out, exist_ok=True)
    wm = WeightManager(args.weights_dir, args.seed)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available()
                          else "cpu")
    M = 16
    hidden = 64
    epochs = 500 if args.quick else 10_000

    # ── Train E2E autoencoder ──
    from checkpoints.checkpoint_24_e2e_transmitter import (
        E2EReceiver, plot_constellation, plot_ber_with_ci as e2e_plot_ber,
    )

    tx = E2ETransmitter(M=M, hidden_dim=hidden).to(device)
    channel = DifferentiableRayleighChannel(perfect_csi=True).to(device)
    rx = E2EReceiver(M=M, hidden_dim=hidden).to(device)

    weight_path = wm._path("e2e_transmitter", subdir="e2e")
    if args.inference_only and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device, weights_only=False)
        tx.load_state_dict(state["transmitter"])
        rx.load_state_dict(state["receiver"])
        print("  Loaded E2E weights")
    else:
        print(f"  Training E2E ({epochs} epochs) …")
        t0 = perf_counter()
        history = train_e2e(tx, channel, rx, epochs=epochs, seed=args.seed,
                            device=device, verbose=True)
        print(f"  Training done ({perf_counter() - t0:.1f}s)")

        # Save weights
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save({
            "transmitter": tx.state_dict(),
            "receiver": rx.state_dict(),
            "M": M, "hidden_dim": hidden, "epochs": epochs,
        }, weight_path)
        print(f"  Weights → {weight_path}")

        # Plot training loss
        if _HAS_MPL and history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history, linewidth=1.0)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Loss", fontsize=14)
            ax.set_title("E2E Training Loss", fontsize=16)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out, "e2e_training_loss.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ── Evaluate BER ──
    snr_eval = np.arange(0, 21, 1, dtype=float)
    bits_per_sym = int(np.log2(M))
    num_trials_e2e = 3 if args.quick else 10
    num_symbols = 1_000 if args.quick else 5_000
    ber_results = evaluate_ber(
        tx, channel, rx, M=M, snr_range_db=snr_eval,
        num_symbols=num_symbols, num_trials=num_trials_e2e,
        device=device)

    # ── Constellation metrics ──
    metrics = constellation_metrics(tx, M=M, device=device)
    print(f"  d_min = {metrics['d_min']:.4f}, PAPR = {metrics['papr']:.4f}")

    # ── Relay comparison (AF vs DF vs E2E) ──
    print("  Running relay comparison (AF vs DF vs E2E) …")
    from relaynet.relays.e2e import E2ERelay
    e2e_relay = E2ERelay(M=M, hidden_dim=hidden, prefer_gpu=args.gpu)
    e2e_relay._transmitter = tx
    e2e_relay._refresh_codebook()
    e2e_relay.is_trained = True

    snr_relay = np.arange(0, 22, 2, dtype=float)
    relay_ch = rayleigh_fading_channel
    relay_results = {}
    for rname, relay in [("AF", AmplifyAndForwardRelay()),
                          ("DF", DecodeAndForwardRelay()),
                          ("E2E", e2e_relay)]:
        print(f"    {rname} …", end=" ", flush=True)
        _, ber, trials = run_monte_carlo(
            relay, snr_relay, channel_fn=relay_ch,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            modulation="qam16")
        ci_lo, ci_hi = compute_confidence_interval(trials)
        relay_results[rname] = {
            "ber_mean": ber, "ber_trials": trials,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
        }
        print("done")

    # Save all results
    save_results_json(
        os.path.join(out, "e2e_ber.json"), snr_eval,
        {"E2E": {"ber_mean": [r[1] for r in ber_results],
                 "ber_trials": [r[2] for r in ber_results]}},
        meta={"experiment": "7.16", "M": M, "epochs": epochs,
              "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                          for k, v in metrics.items() if k != "points"}},
    )
    save_results_json(
        os.path.join(out, "e2e_relay_comparison.json"), snr_relay, relay_results,
        meta={"experiment": "7.16", "comparison": "AF vs DF vs E2E"},
    )

    # Plots
    if _HAS_MPL:
        # Constellation
        if hasattr(plot_constellation, '__call__'):
            try:
                plot_constellation(tx, M=M, device=device,
                                   save_path=os.path.join(out, "e2e_constellation.png"))
            except Exception:
                pass

        # BER vs SNR
        e2e_bers = np.array([r[1] for r in ber_results])
        plot_ber_chart(
            snr_eval, {"E2E Autoencoder": e2e_bers},
            title="E2E BER vs SNR — Rayleigh Fading (§7.16)",
            save_path=os.path.join(out, "e2e_ber_comparison.png"),
        )

        # Relay comparison
        ber_dict = {n: r["ber_mean"] for n, r in relay_results.items()}
        ci_dict = {n: (r["ci_lower"], r["ci_upper"]) for n, r in relay_results.items()}
        plot_ber_chart(
            snr_relay, ber_dict, ci_dict,
            title="E2E vs AF/DF — 16-QAM Rayleigh (§7.16)",
            save_path=os.path.join(out, "e2e_relay_comparison.png"),
        )

    print(f"  §7.16 complete → {out}/")


# ── Constellation diagrams ─────────────────────────────────────────

def exp_constellations(args):
    """Generate constellation diagrams for all 4 modulation schemes."""
    print("\n══ Constellation Diagrams ══")
    if not _HAS_MPL:
        return
    from relaynet.modulation import get_modulation_functions

    out = os.path.join(args.results_dir, "modulation")
    os.makedirs(out, exist_ok=True)

    schemes = {
        "BPSK": "bpsk", "QPSK": "qpsk",
        "16-QAM": "qam16", "16-PSK": "psk16",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (label, mod) in zip(axes.flat, schemes.items()):
        modulate, _, bps = get_modulation_functions(mod)
        bits = np.arange(2 ** bps)
        # Generate all possible symbols
        all_bits = []
        for b in bits:
            bit_arr = [(b >> (bps - 1 - i)) & 1 for i in range(bps)]
            all_bits.extend(bit_arr)
        symbols = modulate(np.array(all_bits))
        if np.iscomplexobj(symbols):
            ax.scatter(symbols.real, symbols.imag, s=80, c=PALETTE[0], zorder=5)
            for i, sym in enumerate(symbols):
                ax.annotate(f"{i}", (sym.real, sym.imag),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
        else:
            ax.scatter(symbols, np.zeros_like(symbols), s=80, c=PALETTE[0], zorder=5)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("In-phase (I)", fontsize=12)
        ax.set_ylabel("Quadrature (Q)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)

    plt.suptitle("Constellation Diagrams", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "constellation_diagrams.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Constellation diagrams → {out}/constellation_diagrams.png")


# ════════════════════════════════════════════════════════════════════
# Experiment registry
# ════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "7.1":  ("Channel Model Analysis",            exp_7_1_channel_analysis),
    "7.2":  ("BPSK AWGN Relay Comparison",         exp_7_2_to_7_7_relay_comparison),
    "7.3":  ("BPSK Rayleigh Relay Comparison",     exp_7_2_to_7_7_relay_comparison),
    "7.4":  ("BPSK Rician K=3",                    exp_7_2_to_7_7_relay_comparison),
    "7.5":  ("2×2 MIMO ZF",                        exp_7_2_to_7_7_relay_comparison),
    "7.6":  ("2×2 MIMO MMSE",                      exp_7_2_to_7_7_relay_comparison),
    "7.7":  ("2×2 MIMO SIC",                       exp_7_2_to_7_7_relay_comparison),
    "7.8":  ("Normalized 3K Comparison",            exp_7_8_normalized_3k),
    "7.9":  ("Master 2×3 Chart",                    exp_7_9_master_chart),
    "7.10": ("Modulation Comparison",               exp_7_10_modulation_comparison),
    "7.11": ("QAM16 Activation Study",              exp_7_11_qam16_activation),
    "7.12": ("LayerNorm Study",                     exp_7_12_layernorm),
    "7.13": ("Activation Comparison",               exp_7_13_activation_comparison),
    "7.14": ("CSI Injection",                       exp_7_14_csi_injection),
    "7.15": ("Multi-Architecture CSI",              exp_7_15_multi_csi),
    "7.16": ("E2E Autoencoder",                     exp_7_16_e2e),
    "constellations": ("Constellation Diagrams",    exp_constellations),
}

# §7.2–§7.7 share one function; avoid running it 6× when --all
_RELAY_COMPARISON_SECTIONS = {"7.2", "7.3", "7.4", "7.5", "7.6", "7.7"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified thesis experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --list
  python run_experiments.py --all --quick
  python run_experiments.py --exp 7.2 7.10 7.16
  python run_experiments.py --all --inference-only
""")
    p.add_argument("--list", action="store_true",
                   help="List all available experiments and exit.")
    p.add_argument("--all", action="store_true",
                   help="Run all experiments sequentially.")
    p.add_argument("--exp", nargs="+", default=[],
                   help="Run specific experiments by section number.")
    p.add_argument("--quick", action="store_true",
                   help="Reduced training/MC effort for fast testing.")
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA for sequence models and E2E.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference-only", action="store_true",
                   help="Skip training; load saved weights.")
    p.add_argument("--weights-dir", type=str, default="weights",
                   help="Directory for weight storage.")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Output directory for JSON + charts.")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable experiments:")
        print(f"  {'Section':<15} Description")
        print(f"  {'-------':<15} -----------")
        for key, (desc, _) in EXPERIMENTS.items():
            print(f"  {key:<15} {desc}")
        return

    # Determine which experiments to run
    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.exp:
        to_run = args.exp
    else:
        print("No experiments selected. Use --all or --exp <section>.")
        print("Run with --list to see available experiments.")
        return

    # Apply quick-mode defaults
    if args.quick:
        if args.bits_per_trial == 10_000:
            args.bits_per_trial = 2_000
        if args.num_trials == 10:
            args.num_trials = 3

    print(f"\n{'='*60}")
    print(f"  Thesis Experiment Runner")
    print(f"  Experiments: {', '.join(to_run)}")
    print(f"  Quick: {args.quick}  |  GPU: {args.gpu}  |  Seed: {args.seed}")
    print(f"  Results → {args.results_dir}/  |  Weights → {args.weights_dir}/")
    print(f"{'='*60}")

    t_start = perf_counter()
    ran_relay_comparison = False

    for exp_key in to_run:
        if exp_key not in EXPERIMENTS:
            print(f"\n  [WARNING] Unknown experiment: {exp_key}")
            continue

        # §7.2–7.7 share one function — run only once
        if exp_key in _RELAY_COMPARISON_SECTIONS:
            if ran_relay_comparison:
                continue
            ran_relay_comparison = True

        _, func = EXPERIMENTS[exp_key]
        func(args)

    elapsed = perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  All experiments completed in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
