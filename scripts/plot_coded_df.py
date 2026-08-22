#!/usr/bin/env python3
"""Generate the coded block-DF figures for Chapter 5 from
results/coded_df_experiment.json, following CHART_GUIDELINES.md
(distinct color+marker+linestyle per curve, thin lines, 95% CI bands,
legend outside/clear of curves, y-axis focused one decade below the
minimum nonzero BER, trial/bit-budget annotation).

CI bands use the binomial approximation 1.96*sqrt(p(1-p)/n) on the
total measured information bits per SNR point (10 trials x the frame
budget) -- the same "1.96*sigma/sqrt(n)" convention already used
elsewhere in this thesis (Chapter 7's composite-channel study) -- since
per-trial arrays were not persisted, only the trial-averaged BER.

Writes to both results/ and thesis/results/, matching
scripts/plot_e6_studies.py's convention so the repo copy and the copy
main.tex compiles against never drift apart.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]

N_TRIALS = 10
FRAME_INFO_BITS = 200
N_FRAMES = 500
N_BITS_TOTAL = N_TRIALS * N_FRAMES * FRAME_INFO_BITS  # 1,000,000 info bits/SNR point


def _save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _ci95(p, n=N_BITS_TOTAL):
    p = np.asarray(p, dtype=float)
    return 1.96 * np.sqrt(np.maximum(p * (1 - p), 0) / n)


def _plot_panel(snrs, series, title, name, note):
    """series: list of (label, ber_array, color, marker, ls)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ymin = min(min(b) for _, b, *_ in series)
    for label, ber, color, marker, ls in series:
        ber = np.asarray(ber, dtype=float)
        ci = _ci95(ber)
        ax.semilogy(snrs, np.maximum(ber, 1e-5), label=label, color=color,
                    marker=marker, ls=ls, lw=1.5, markersize=6)
        ax.fill_between(snrs, np.maximum(ber - ci, 1e-6), ber + ci,
                        color=color, alpha=0.15, lw=0)
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("BER", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(bottom=10 ** (np.floor(np.log10(ymin)) - 1))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9, color="0.35")
    fig.tight_layout()
    _save(fig, name)


def main():
    with open(os.path.join(ROOT, "results/coded_df_experiment.json")) as f:
        d = json.load(f)
    snrs = np.asarray(d["snr_db"], dtype=float)
    note = f"{N_TRIALS} trials $\\times$ {N_FRAMES * FRAME_INFO_BITS:,} info bits/SNR point"

    # Figure 1: coded block-DF vs uncoded DF and coded-aware learned relays (tbl:table34).
    _plot_panel(
        snrs,
        [
            ("uncoded DF (symbol-wise)", d["uncoded_df"], "0.35", "o", "-"),
            ("coded AF (no relay decode)", d["coded_af"], "tab:orange", "s", "--"),
            ("coded DF, K=3 (Viterbi)", d["coded_df"], "tab:red", "v", "-"),
            ("MLP-coded (756p)", d["mlp_coded"], "tab:blue", "^", "-."),
            ("Mamba-coded (24,084p)", d["mamba_coded"], "tab:green", "D", ":"),
        ],
        "Coded block-DF vs. uncoded DF and coded-aware learned relays (QPSK, K=3)",
        "coded_df_comparison.png",
        note,
    )

    # Figure 2: constraint-length sweep, QPSK (tbl:table35).
    ks = d["k_sweep"]
    _plot_panel(
        snrs,
        [
            ("K=3 (4 states)", ks["K3"]["ber"], "tab:blue", "^", "-"),
            ("K=5 (16 states)", ks["K5"]["ber"], "tab:purple", "v", "--"),
            ("K=7 (64 states)", ks["K7"]["ber"], "tab:red", "D", "-."),
        ],
        "Coded-DF BER vs. constraint length, QPSK/Rayleigh, rate 1/2",
        "coded_df_k_sweep_qpsk.png",
        note,
    )

    # Figure 3: constraint-length sweep, 16-QAM, with uncoded reference (tbl:table36).
    qk = d["qam16_k_sweep"]
    _plot_panel(
        snrs,
        [
            ("uncoded DF", d["qam16_uncoded_df"], "0.35", "o", "-"),
            ("K=3 (4 states)", qk["K3"]["ber"], "tab:blue", "^", "-"),
            ("K=5 (16 states)", qk["K5"]["ber"], "tab:purple", "v", "--"),
            ("K=7 (64 states)", qk["K7"]["ber"], "tab:red", "D", "-."),
        ],
        "Coded-DF BER vs. constraint length, 16-QAM/Rayleigh, rate 1/2",
        "coded_df_k_sweep_qam16.png",
        note,
    )




def plot_soft_decision():
    """Soft- vs hard-decision relaying (results/coded_soft_decision.json)."""
    import os as _os
    path = _os.path.join(ROOT, "results/coded_soft_decision.json")
    if not _os.path.exists(path):
        print("  (skipped soft-decision figure: results file not present)")
        return
    with open(path) as f:
        d = json.load(f)
    snrs = np.asarray(d["snr_db"], dtype=float)
    n = d["n_trials"] * d["n_frames"] * 200

    def series(key):
        return [d["summary"][str(int(s))][key]["mean"] for s in snrs]

    fig, ax = plt.subplots(figsize=(9, 6))
    spec = [
        ("hard block-DF (Viterbi)", "coded_df", "tab:red", "v", "-"),
        ("soft block-DF (BCJR)", "soft_df", "tab:orange", "s", "--"),
        ("MLP hard (argmax)", "mlp_hard", "tab:blue", "^", "-."),
        ("MLP soft (posterior mean)", "mlp_soft", "tab:green", "D", "-"),
        ("oracle relay (hop-2 floor)", "oracle", "0.35", "o", ":"),
    ]
    ymin = min(min(series(k)) for _, k, *_ in spec)
    for label, key, color, marker, ls in spec:
        ber = np.asarray(series(key))
        ci = 1.96 * np.sqrt(np.maximum(ber * (1 - ber), 0) / n)
        ax.semilogy(snrs, ber, label=label, color=color, marker=marker,
                    ls=ls, lw=1.5, markersize=6)
        ax.fill_between(snrs, np.maximum(ber - ci, 1e-7), ber + ci,
                        color=color, alpha=0.15, lw=0)
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("BER", fontsize=13)
    ax.set_title("Soft- vs hard-decision relaying on the coded link (QPSK, K=3)", fontsize=14)
    ax.set_ylim(bottom=10 ** (np.floor(np.log10(ymin)) - 1))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.annotate(f"{d['n_trials']} paired trials $\\times$ {d['n_frames']*200:,} info bits/SNR point",
                xy=(0.98, 0.97), xycoords="axes fraction", fontsize=9, color="0.35",
                ha="right", va="top")
    fig.tight_layout()
    _save(fig, "coded_soft_decision.png")


def plot_rate_adaptation():
    """Link-adaptation envelope (results/coded_rate_adaptation.json)."""
    import os as _os
    path = _os.path.join(ROOT, "results/coded_rate_adaptation.json")
    if not _os.path.exists(path):
        print("  (skipped rate-adaptation figure: results file not present)")
        return
    with open(path) as f:
        d = json.load(f)
    snrs = np.asarray(d["snr_db"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # every individual MCS, faint, so the envelope is visibly an upper hull
    for key, v in sorted(d["mcs"].items()):
        mod, rate, relay = key.split("|")
        if relay == "blockdf" or rate == "uncoded":
            ax.plot(snrs, v["goodput"], color="0.75", lw=0.9, zorder=1)

    env_b = [r["goodput"] for r in d["envelope"]["blockdf"]]
    env_n = [r["goodput"] for r in d["envelope"]["denoise"]]
    ax.plot(snrs, env_b, color="tab:red", marker="v", ls="-", lw=2.0,
            markersize=7, label="envelope, block-DF relay", zorder=3)
    ax.plot(snrs, env_n, color="tab:green", marker="D", ls="--", lw=2.0,
            markersize=6, label="envelope, denoise-only relay", zorder=3)

    # annotate which MCS wins each point on the block-DF envelope
    for r, g in zip(d["envelope"]["blockdf"], env_b):
        if g > 0.3:
            ax.annotate(r["mcs"].replace("qam16", "16QAM").replace("qpsk", "QPSK"),
                        xy=(r["snr_db"], g), xytext=(0, 9),
                        textcoords="offset points", fontsize=8,
                        ha="center", color="tab:red")

    ax.axhline(4.0, color="0.4", ls=":", lw=1.0)
    ax.text(snrs[0] + 0.3, 4.05, "16-QAM raw rate (4 bits/symbol)",
            fontsize=8.5, color="0.4")
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Goodput  $R\\,(1-\\mathrm{FER})$  [info bits/symbol]", fontsize=13)
    ax.set_title("Link-adaptation envelope: best MCS chosen per SNR point", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.35)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.annotate(f"grey: individual MCS  |  {d['n_trials']} trials $\\times$ "
                f"{d['n_frames']} frames/point",
                xy=(0.98, 0.03), xycoords="axes fraction", fontsize=9,
                color="0.35", ha="right")
    fig.tight_layout()
    _save(fig, "coded_rate_adaptation.png")


def plot_latency_capacity():
    """Goodput vs. round-trip latency budget (results/coded_latency_capacity.json)."""
    import os as _os
    path = _os.path.join(ROOT, "results/coded_latency_capacity.json")
    if not _os.path.exists(path):
        print("  (skipped latency-capacity figure: results file not present)")
        return
    with open(path) as f:
        d = json.load(f)

    snr_colors = {"16": "tab:blue", "20": "tab:orange",
                  "24": "tab:green", "28": "tab:red"}

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for snr, color in snr_colors.items():
        for relay, ls, marker in (("blockdf", "-", "v"), ("denoise", "--", "D")):
            pts = d["curves"][relay][snr]
            x = [p["l_max"] for p in pts]
            y = [p["goodput"] for p in pts]
            label = f"{snr} dB, " + ("block-DF" if relay == "blockdf" else "denoise-only")
            ax.plot(x, y, color=color, ls=ls, marker=marker, markersize=5,
                    lw=1.5, drawstyle="steps-post", label=label)

    ax.axvline(d["table_budget"], color="0.4", ls=":", lw=1.0)
    ax.text(d["table_budget"] + 4, ax.get_ylim()[1] * 0.03 if ax.get_ylim()[1] else 0.05,
            f"Table~budget = {d['table_budget']} sym", fontsize=8.5, color="0.4")
    ax.set_xlabel("Round-trip latency budget $L_{\\max}$ (symbols)", fontsize=13)
    ax.set_ylabel("Best achievable goodput within $L_{\\max}$ [info bits/symbol]", fontsize=13)
    ax.set_title("Capacity under a latency constraint: goodput vs. deadline", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 420)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.annotate("solid = block-DF (2x frame round-trip)  |  dashed = denoise-only (~1x frame)",
                xy=(0.98, 0.03), xycoords="axes fraction", fontsize=9,
                color="0.35", ha="right")
    fig.tight_layout()
    _save(fig, "coded_latency_capacity.png")


if __name__ == "__main__":
    main()
    plot_soft_decision()
    plot_rate_adaptation()
    plot_latency_capacity()
