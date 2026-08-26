"""gen_missing_figures.py — generate figures for Ch5/Ch6 tables that lack a chart.

Figures produced:
  results/normalized_3k_rayleigh_ber.png   (Ch5 tbl:table8 — 3K-param equal-budget)
  results/coded_df_ber.png                 (Ch5 tbl:table34 — coded block-DF study)
  results/coded_reliable_regime.png        (Ch5 tbl:table44 — reliable-decoding regime)
  results/coded_equal_throughput.png       (Ch5 tbl:table40 — coded vs uncoded equal SE)
  results/all_relays_16class/ber_16class_vs_snr.png  (Ch6 tbl:table24 — 16-QAM 4-cls vs 16-cls)

All data is read from committed JSON files in results/; no re-simulation.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

FIG_W, FIG_H = 9, 5


def _add_zoom_inset(ax, snr, series, snr_lo, snr_hi,
                    loc="lower left", width="38%", height="35%",
                    pad=1.2, ber_margin=0.35):
    """Add a zoom inset covering snr_lo..snr_hi.

    Parameters
    ----------
    ax       : parent Axes
    snr      : array-like of SNR values
    series   : dict  name -> {"ber": array, "color": str, "marker": str,
                               "ls": str, "lw": float}
    snr_lo/hi: SNR range to zoom into
    loc      : inset anchor location string
    ber_margin: multiplicative padding on ber y-axis (fraction)
    """
    snr = np.asarray(snr)
    mask = (snr >= snr_lo) & (snr <= snr_hi)
    if not np.any(mask):
        return
    snr_z = snr[mask]

    # Collect all BER values in the zoom window
    vals = []
    for info in series.values():
        ber = np.asarray(info["ber"])
        v = ber[mask]
        vals.extend(v[v > 0].tolist())
    if not vals:
        return

    ber_lo = min(vals) * (1 - ber_margin)
    ber_hi = max(vals) * (1 + ber_margin)

    axins = inset_axes(ax, width=width, height=height, loc=loc,
                       borderpad=pad)
    for name, info in series.items():
        ber = np.asarray(info["ber"])
        ber_z = np.where(ber[mask] > 0, ber[mask], 1e-12)
        axins.semilogy(snr_z, ber_z,
                       color=info["color"],
                       marker=info.get("marker", "o"),
                       ls=info.get("ls", "-"),
                       lw=info.get("lw", 1.2),
                       markersize=4,
                       alpha=0.9)

    axins.set_xlim(snr_z[0] - 0.3, snr_z[-1] + 0.3)
    axins.set_ylim(ber_lo, ber_hi)
    axins.tick_params(labelsize=7)
    axins.grid(True, which="both", ls="--", alpha=0.25, lw=0.4)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.55)
    except Exception:
        pass


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1 — tbl:table8 — equal-parameter 3K budget, BER vs SNR on Rayleigh
# ---------------------------------------------------------------------------
def fig_3k_budget():
    with open(os.path.join(RESULTS, "normalized_3k", "3k_rayleigh.json")) as f:
        d = json.load(f)

    snr = d["snr_range"]
    relays = d["results"]

    AF_style  = {"color": "grey",  "marker": "o", "ls": "--", "lw": 1.4}
    DF_style  = {"color": "black", "marker": "s", "ls": "--", "lw": 1.4}
    ai_styles = {
        "MLP-3K":         {"color": PALETTE[0],  "marker": MARKERS[0]},
        "Hybrid-3K":      {"color": PALETTE[1],  "marker": MARKERS[1]},
        "VAE-3K":         {"color": PALETTE[2],  "marker": MARKERS[2]},
        "Transformer-3K": {"color": PALETTE[3],  "marker": MARKERS[3]},
        "Mamba-3K":       {"color": PALETTE[4],  "marker": MARKERS[4]},
        "Mamba2-3K":      {"color": PALETTE[5],  "marker": MARKERS[5]},
    }

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for name, st in [("AF", AF_style), ("DF", DF_style)]:
        r = relays[name]
        ax.semilogy(snr, r["ber_mean"], label=name,
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=st["lw"], markersize=5)

    for i, (name, st) in enumerate(ai_styles.items()):
        if name not in relays:
            continue
        r = relays[name]
        ber = np.asarray(r["ber_mean"])
        lo  = np.asarray(r["ci_lower"])
        hi  = np.asarray(r["ci_upper"])
        ax.semilogy(snr, ber, label=name,
                    color=st["color"], marker=st["marker"],
                    ls="-", lw=1.3, markersize=5)
        ax.fill_between(snr, lo, hi, color=st["color"], alpha=0.12)

    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Equal-parameter-budget comparison (≈3 000 parameters), QPSK / Rayleigh")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xticks(snr[::2])

    # Zoom inset: high-SNR region where 6 AI curves + DF converge tightly
    zoom_series = {}
    for name, st in [("AF", AF_style), ("DF", DF_style)]:
        zoom_series[name] = {"ber": relays[name]["ber_mean"],
                             "color": st["color"], "marker": st["marker"],
                             "ls": st["ls"], "lw": st["lw"]}
    for name, st in ai_styles.items():
        if name in relays:
            zoom_series[name] = {"ber": relays[name]["ber_mean"],
                                 "color": st["color"], "marker": st["marker"],
                                 "ls": "-", "lw": 1.3}
    _add_zoom_inset(ax, snr, zoom_series, snr_lo=14, snr_hi=20,
                    loc="lower left", width="38%", height="38%", pad=1.0)

    _save(fig, os.path.join(RESULTS, "normalized_3k_rayleigh_ber.png"))


# ---------------------------------------------------------------------------
# Figure 2 — tbl:table34 — coded block-DF vs uncoded DF vs learned relays
# ---------------------------------------------------------------------------
def fig_coded_df():
    with open(os.path.join(RESULTS, "coded_df_experiment.json")) as f:
        d = json.load(f)

    snr = d["snr_db"]
    series = {
        "Uncoded DF":   (d["uncoded_df"],  {"color": "black",     "marker": "s", "ls": "--"}),
        "Coded DF":     (d["coded_df"],    {"color": "steelblue", "marker": "^", "ls": "-"}),
        "MLP (coded)":  (d["mlp_coded"],   {"color": PALETTE[0],  "marker": "o", "ls": "-"}),
        "Mamba (coded)":(d["mamba_coded"], {"color": PALETTE[4],  "marker": "D", "ls": "-"}),
    }

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for label, (vals, st) in series.items():
        ax.semilogy(snr, vals, label=label,
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.4, markersize=5)

    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Coded block-DF vs uncoded DF vs coded-aware learned relays (QPSK / Rayleigh, rate-1/2 K=3)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Zoom inset: SNR 12–20 dB where coded DF, MLP, Mamba converge
    zoom_series = {label: {"ber": vals, "color": st["color"],
                            "marker": st["marker"], "ls": st["ls"], "lw": 1.4}
                   for label, (vals, st) in series.items()}
    _add_zoom_inset(ax, snr, zoom_series, snr_lo=12, snr_hi=20,
                    loc="lower left", width="38%", height="38%", pad=1.0)

    _save(fig, os.path.join(RESULTS, "coded_df_ber.png"))


# ---------------------------------------------------------------------------
# Figure 3 — tbl:table44 — reliable-decoding regime (extended SNR)
# ---------------------------------------------------------------------------
def fig_reliable_regime():
    with open(os.path.join(RESULTS, "coded_reliable_regime.json")) as f:
        d = json.load(f)

    snr = d["snr_db"]
    series = {
        "Coded DF":       (d["coded_df"],   {"color": "steelblue", "marker": "^", "ls": "-"}),
        "MLP (thesis)":   (d["mlp_thesis"], {"color": PALETTE[0],  "marker": "o", "ls": "-"}),
        "MLP (extended)": (d["mlp_ext"],    {"color": PALETTE[1],  "marker": "s", "ls": "-."}),
        "Oracle":         (d["oracle"],     {"color": "grey",       "marker": "D", "ls": "--"}),
    }

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for label, (vals, st) in series.items():
        ax.semilogy(snr, vals, label=label,
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.4, markersize=6)

    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Reliable-decoding regime: BER at extended SNR (QPSK / Rayleigh, rate-1/2 K=3)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xticks(snr)

    # Zoom inset: 28–32 dB where all 4 curves are extremely close
    zoom_series = {label: {"ber": vals, "color": st["color"],
                            "marker": st["marker"], "ls": st["ls"], "lw": 1.4}
                   for label, (vals, st) in series.items()}
    _add_zoom_inset(ax, snr, zoom_series, snr_lo=28, snr_hi=32,
                    loc="upper right", width="38%", height="38%", pad=1.0,
                    ber_margin=0.5)

    _save(fig, os.path.join(RESULTS, "coded_reliable_regime.png"))


# ---------------------------------------------------------------------------
# Figure 4 — tbl:table40 — coded vs uncoded at equal spectral efficiency
# ---------------------------------------------------------------------------
def fig_equal_throughput():
    with open(os.path.join(RESULTS, "coded_latency_throughput.json")) as f:
        d = json.load(f)

    rows = d["equal_throughput"]
    snr  = [r["snr_db"] for r in rows]
    unc  = [r["uncoded_qpsk"] for r in rows]
    cod  = [r["coded_qam16"] for r in rows]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.semilogy(snr, unc, label="Uncoded QPSK (2.0 info-bits/sym)",
                color="black", marker="s", ls="--", lw=1.4, markersize=5)
    ax.semilogy(snr, cod, label="Rate-1/2 16-QAM (≈2.0 info-bits/sym)",
                color="steelblue", marker="^", ls="-", lw=1.4, markersize=5)

    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Coded vs uncoded at equal spectral efficiency (≈2 information bits per channel symbol)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Zoom inset: SNR 14–20 dB — crossover zone where lines nearly converge then swap
    zoom_series = {
        "Uncoded QPSK": {"ber": unc, "color": "black",     "marker": "s", "ls": "--", "lw": 1.4},
        "Rate-1/2 16-QAM": {"ber": cod, "color": "steelblue", "marker": "^", "ls": "-",  "lw": 1.4},
    }
    _add_zoom_inset(ax, snr, zoom_series, snr_lo=14, snr_hi=20,
                    loc="lower left", width="38%", height="38%", pad=1.0,
                    ber_margin=0.4)

    _save(fig, os.path.join(RESULTS, "coded_equal_throughput.png"))


# ---------------------------------------------------------------------------
# Figure 5 — tbl:table24 (Ch6) — 16-QAM: 4-class vs 16-class relays
# ---------------------------------------------------------------------------
def fig_16qam_4vs16():
    with open(os.path.join(RESULTS, "all_relays_16class", "all_relays_16class.json")) as f:
        d = json.load(f)

    snr    = d["snr_range"]
    relays = d["results"]

    # Separate 4-cls and 16-cls variants
    base_names = ["MLP", "VAE", "Hybrid", "Transformer", "Mamba-S6", "Mamba2"]
    colors = {n: PALETTE[i] for i, n in enumerate(base_names)}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Baselines first
    for name in ("AF", "DF"):
        if name in relays:
            r = relays[name]
            c = "grey" if name == "AF" else "black"
            ls = "--"
            ax.semilogy(snr, r["ber_mean"], label=name,
                        color=c, marker="o", ls=ls, lw=1.4, markersize=4)

    for base in base_names:
        for suffix, ls in [("4-cls", ":"), ("16-cls", "-")]:
            key = f"{base} {suffix}"
            if key not in relays:
                continue
            r = relays[key]
            ax.semilogy(snr, r["ber_mean"],
                        label=key,
                        color=colors[base],
                        marker=MARKERS[base_names.index(base)],
                        ls=ls, lw=1.2, markersize=4,
                        alpha=0.85)

    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("16-QAM BER vs SNR: per-axis 4-class (dotted) vs joint 16-class (solid), canonical Rayleigh")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xticks(snr[::2])

    _save(fig, os.path.join(RESULTS, "all_relays_16class", "ber_16class_vs_snr.png"))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating missing figures...")
    fig_3k_budget()
    fig_coded_df()
    fig_reliable_regime()
    fig_equal_throughput()
    fig_16qam_4vs16()
    print("Done.")
