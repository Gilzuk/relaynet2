#!/usr/bin/env python3
"""Regenerate results/e6_unknown_channel.png as a single AWGN-only panel.

The original figure (never checked in with a regeneration script -- only its
data survived) had three panels: (a) unknown ISI -> AWGN, (b) unknown ISI ->
Rayleigh, (c) control: canonical Rayleigh. Per author instruction, panels
(b) and (c) -- the unknown-ISI-to-Rayleigh and canonical-Rayleigh-control
variants specific to this figure and Table tbl:tableE6 -- are being removed,
which leaves only panel (a). Other Rayleigh hop-2 studies in the same
chapter (flat unknown channels, composite cascade) are unaffected. This
script rebuilds panel (a) alone from the same committed data the
three-panel figure used: AF/DF/MLP from
e6_unknown_channel_results/e6_sim_ported_results.npy (setup "S1: unknown
ISI -> AWGN"), plus the two Viterbi MLSE baselines (genie CSI and 200-pilot
LS) from e6_unknown_channel_results/e6_viterbi_awgn.npy -- the same source
verify_thesis_tables.py's check_tableE6 checks the table's Viterbi rows
against. The five curve classes are verified against their declared sources.
The historical
0--10 dB values remain unchanged, while the AF/DF/MLP values at 12--16 dB
are replaced with the predeclared fixed-budget results in
e6_unknown_channel_results/codex_isi_fixed_budget_validation.json.  The MLP
18--20 dB adaptive rare-event tail is deliberately omitted.  The Viterbi
arrays carry no stored confidence interval, so only AF/DF/MLP get a shaded
CI band.

Figures are written to BOTH results/ and thesis/results/ so the repository
copy and the copy main.tex compiles against never drift apart, matching the
convention of scripts/plot_e6_studies.py.

Run after this script exists (once, to replace the checked-in three-panel
PNG):

    python3 scripts/plot_e6_unknown_channel_awgn.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_DIR = os.path.join(ROOT, "e6_unknown_channel_results")
SIM_PATH = os.path.join(NPY_DIR, "e6_sim_ported_results.npy")
VITERBI_PATH = os.path.join(NPY_DIR, "e6_viterbi_awgn.npy")
FIXED_BUDGET_PATH = os.path.join(NPY_DIR, "codex_isi_fixed_budget_validation.json")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]
SETUP = "S1: unknown ISI -> AWGN"

STYLE = {
    "AF":  dict(color="tab:orange", marker="s", ls="--", label="AF"),
    "DF":  dict(color="firebrick",  marker="o", ls="-",  label="DF"),
    "MLP": dict(color="tab:blue",   marker="^", ls="-",  label="MLP (170 params)"),
}
VITERBI_STYLE = {
    "VIT-genie": dict(color="tab:green", marker="v", ls="-.", label="Viterbi (genie CSI)"),
    "VIT-est":   dict(color="tab:purple", marker="D", ls=":", label="Viterbi (200-pilot LS)"),
}


def main():
    d = np.load(SIM_PATH, allow_pickle=True).item()
    snrs = np.asarray(d["snrs"], dtype=float)
    r = d["results"][SETUP]
    vg = np.load(VITERBI_PATH, allow_pickle=True).item()

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

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, st in STYLE.items():
        mu, ci = display[key]
        ax.semilogy(snrs, np.maximum(mu, 1e-8), markersize=6, **st)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-8), np.maximum(mu + ci, 1e-8),
                        color=st["color"], alpha=0.18, lw=0)
    for key, st in VITERBI_STYLE.items():
        mu = np.asarray(vg[key], dtype=float)
        ax.semilogy(snrs, np.maximum(mu, 1e-8), markersize=6, **st)

    ax.axhline(0.25, color="0.4", ls=":", lw=1.2)
    ax.text(0.3, 0.25 * 1.06, "memoryless floor = 0.25", color="0.4", fontsize=9)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Unknown ISI channel: fixed-budget validation through 16 dB")
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.annotate("MLP 18--20 dB\nnot re-estimated", xy=(16, 1.2e-7),
                xytext=(16.8, 2e-6), fontsize=8, color=STYLE["MLP"]["color"],
                arrowprops={"arrowstyle": "->", "color": STYLE["MLP"]["color"]})

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "e6_unknown_channel.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
