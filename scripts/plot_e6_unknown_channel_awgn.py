#!/usr/bin/env python3
"""Regenerate the original fixed-ISI/AWGN figure from the published sources.

AF/DF/MLP use the historical sweep below 12 dB and the separately preserved
fixed-budget validation at 12--16 dB. The unvalidated MLP tail is omitted.
Viterbi uses e6_matched_protocol.json and the 16/20-dB high-budget counts,
exactly as the central table. Zero errors are drawn as open nominal 3/N
bounds at the actual exposures, not as measured positive BERs. Such bounds
assume independent Bernoulli errors; correlated-error coverage is not proved.
Writes the original PNG and vector PDF in results/ and thesis/results/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_DIR = os.path.join(ROOT, "e6_unknown_channel_results")
SIM_PATH = os.path.join(NPY_DIR, "e6_sim_ported_results.npy")
MATCHED_PATH = os.path.join(ROOT, "results", "e6_matched_protocol.json")
HIGH_SNR_PATH = os.path.join(ROOT, "results", "e6_matched_highsnr.json")
FIXED_BUDGET_PATH = os.path.join(NPY_DIR, "codex_isi_fixed_budget_validation.json")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]
SETUP = "S1: unknown ISI -> AWGN"

STYLE = {
    "AF":  dict(color="#E69F00", marker="s", ls="--", label="AF"),
    "DF":  dict(color="#D55E00", marker="o", ls="-", label="DF"),
    "MLP": dict(color="#0072B2", marker="^", ls="-", label="MLP (170 parameters)"),
}
VITERBI_STYLE = {
    "VIT-genie": dict(color="#009E73", marker="v", ls="-.", label="Viterbi (genie CSI)"),
    "VIT-est": dict(color="#CC79A7", marker="D", ls=":", label="Viterbi (200-pilot LS)"),
}

# Axis/interval clipping only; zero-error markers instead use actual 3/N bounds.
PLOT_FLOOR = 1e-8

def matched_points(name):
    """Return empirical BER and actual bit budgets, using the table sources."""
    with open(MATCHED_PATH, encoding="utf-8") as handle:
        matched = json.load(handle)
    with open(HIGH_SNR_PATH, encoding="utf-8") as handle:
        high = json.load(handle)
    record = matched["results"][name]
    errors = np.array(record["errors"], dtype=float)
    bits = np.array(record["bits"], dtype=float)
    for snr in (16, 20):
        i = matched["snrs"].index(snr)
        errors[i] = high["results"][name][str(snr)]["errors"]
        bits[i] = high["results"][name][str(snr)]["bits"]
    return np.asarray(matched["snrs"]), errors / bits, bits


def main():
    d = np.load(SIM_PATH, allow_pickle=True).item()
    snrs = np.asarray(d["snrs"], dtype=float)
    r = d["results"][SETUP]
    vg = {key: matched_points(key) for key in VITERBI_STYLE}

    # The original sweep used an adaptive, post-error extension at high SNR.
    # Replace 12--16 dB by predeclared fixed-budget results.  The 16-dB
    # confirmation supersedes the short initial run; the unvalidated MLP
    # values at 18 and 20 dB are not silently shown as zero/floor points.
    with open(FIXED_BUDGET_PATH, encoding="utf-8") as f:
        fixed = json.load(f)["points"]
    display = {}
    for key, value in r.items():
        display[key] = [np.asarray(part, dtype=float).copy() for part in value]
    for snr_text, point in fixed.items():
        idx = int(np.flatnonzero(snrs == float(snr_text))[0])
        for relay in ("AF", "DF", "MLP"):
            display[relay][0][idx] = point[relay]["mean"]
            display[relay][1][idx] = point[relay]["ci95_halfwidth"]
    display["MLP"][0][snrs > 16] = np.nan
    display["MLP"][1][snrs > 16] = np.nan

    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, st in STYLE.items():
        mu, ci = display[key]
        ax.semilogy(snrs, np.maximum(mu, PLOT_FLOOR), markersize=6, **st)
        ax.fill_between(snrs, np.maximum(mu - ci, PLOT_FLOOR), np.maximum(mu + ci, PLOT_FLOOR),
                        color=st["color"], alpha=0.18, lw=0)
    for key, st in VITERBI_STYLE.items():
        v_snrs, mu, bits = vg[key]
        np.testing.assert_array_equal(v_snrs, snrs)
        observed = mu > 0
        if np.any(observed):
            ax.semilogy(snrs[observed], mu[observed], markersize=6, **st)
        censored = ~observed
        if np.any(censored):
            bounds = 3.0 / bits[censored]
            ax.scatter(snrs[censored], bounds,
                       marker=st["marker"], s=42, facecolors="white", edgecolors=st["color"],
                       linewidths=1.2, zorder=4)
            for x, bound in zip(snrs[censored], bounds):
                ax.annotate("", xy=(x, max(PLOT_FLOOR, bound / 3)), xytext=(x, bound),
                            arrowprops={"arrowstyle": "-|>", "color": st["color"],
                                        "lw": 0.9, "alpha": 0.8})

    # Add one compact legend entry for the open-marker convention without
    # duplicating it for the two Viterbi variants.
    if any(np.any(vg[key][1] == 0) for key in VITERBI_STYLE):
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white",
                              linestyle="None", markersize=6,
                              label="Zero errors: nominal 3/N bound"))
        labels.append("Zero errors: nominal 3/N bound")
        ax.legend(handles, labels, loc="lower left", fontsize=9)

    ax.axhline(0.25, color="0.4", ls=":", lw=1.2)
    ax.text(0.3, 0.25 * 1.15, "AF / zero-threshold DF limit = 0.25", color="0.4", fontsize=9)

    ax.set_xlabel(r"Per-hop $E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Fixed three-tap ISI followed by AWGN")
    ax.set_ylim(PLOT_FLOOR, 1e0)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.08)
    if not any(np.any(vg[key][1] == 0) for key in VITERBI_STYLE):
        ax.legend(loc="lower left", fontsize=9)
    ax.annotate("MLP 18--20 dB\nnot re-estimated", xy=(16, 1.2e-7),
                xytext=(0.70, 0.48), textcoords="axes fraction", fontsize=8, color=STYLE["MLP"]["color"],
                arrowprops={"arrowstyle": "->", "color": STYLE["MLP"]["color"]})

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "e6_unknown_channel.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
