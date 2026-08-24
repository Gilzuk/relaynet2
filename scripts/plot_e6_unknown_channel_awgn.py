#!/usr/bin/env python3
"""Regenerate results/e6_unknown_channel.png as a single AWGN-only panel.

The original figure (never checked in with a regeneration script -- only its
data survived) had three panels: (a) unknown ISI -> AWGN, (b) unknown ISI ->
Rayleigh, (c) control: canonical Rayleigh. Per author instruction, every
BPSK-with-Rayleigh configuration in the unknown/mismatched-channels chapter
is being removed, which leaves
only panel (a). This script rebuilds that single panel from the same
committed data the three-panel figure used
(e6_unknown_channel_results/e6_sim_ported_results.npy, setup "S1: unknown
ISI -> AWGN"), so the curves are pixel-for-pixel the same data already
verified against Table tbl:tableE6's surviving AWGN row -- nothing is
re-simulated or touched numerically.

Figures are written to BOTH results/ and thesis/results/ so the repository
copy and the copy main.tex compiles against never drift apart, matching the
convention of scripts/plot_e6_studies.py.

Run after this script exists (once, to replace the checked-in three-panel
PNG):

    python3 scripts/plot_e6_unknown_channel_awgn.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(ROOT, "e6_unknown_channel_results", "e6_sim_ported_results.npy")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]
SETUP = "S1: unknown ISI -> AWGN"

STYLE = {
    "AF":  dict(color="tab:orange", marker="s", ls="--", label="AF"),
    "DF":  dict(color="firebrick",  marker="o", ls="-",  label="DF"),
    "MLP": dict(color="tab:blue",   marker="^", ls="-",  label="MLP (169 params)"),
}


def main():
    d = np.load(NPY_PATH, allow_pickle=True).item()
    snrs = np.asarray(d["snrs"], dtype=float)
    r = d["results"][SETUP]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, st in STYLE.items():
        mu, ci = (np.asarray(v, dtype=float) for v in r[key])
        ax.semilogy(snrs, np.maximum(mu, 1e-5), markersize=6, **st)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-6), np.maximum(mu + ci, 1e-6),
                        color=st["color"], alpha=0.18, lw=0)

    ax.axhline(0.25, color="0.4", ls=":", lw=1.2)
    ax.text(0.3, 0.25 * 1.06, "memoryless floor = 0.25", color="0.4", fontsize=9)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Unknown ISI channel: classical failure and learned mitigation")
    ax.set_ylim(1e-5, 1e0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "e6_unknown_channel.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
