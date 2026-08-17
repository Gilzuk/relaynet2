#!/usr/bin/env python3
"""Regenerate the Chapter 5 AWGN companion figure from committed JSON.

The AWGN BER falls exponentially, so above about 12 dB the true error rate
is far below what the 10 x 10,000 = 100,000 bit budget can resolve (~1e-5):
the two-hop DF BER is 2.8e-10 at 16 dB and 1.5e-23 at 20 dB. Plotting those
points shows curves diving into the resolution floor, which reads as though
the relays attain 1e-5 when in fact nothing was measured there. The figure is
therefore drawn only over the range where every plotted value is an actual
estimate, matching Table 5.5.

Usage:  python3 scripts/plot_awgn_companion.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "bpsk_comparison", "awgn.json")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]
NAME = "awgn_comparison_ci.png"

MAX_SNR = 8           # last SNR point with a statistically meaningful estimate
RESOLUTION = 1e-5     # 10 trials x 10,000 bits aggregate

# Lean relay set, matching the regenerated awgn.json.
RELAYS = [
    ("AF",               "tab:gray",   "o", "-"),
    ("DF",               "black",      "s", "-"),
    ("MLP (169p)",       "tab:blue",   "^", "-"),
    ("Transformer",      "tab:red",    "X", "-"),
    ("Mamba2 (SSD)",     "tab:pink",   ">", "-"),
]
LABEL = {"Mamba2 (SSD)": "Mamba-2 SSD"}


def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def main():
    d = json.load(open(SRC))
    snrs = np.array(d["snr_range"], dtype=float)
    keep = snrs <= MAX_SNR
    x = snrs[keep]

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, colour, marker, ls in RELAYS:
        r = d["results"][key]
        mu = np.array(r["ber_mean"], dtype=float)[keep]
        ax.semilogy(x, np.maximum(mu, RESOLUTION / 10), color=colour, marker=marker,
                    ls=ls, markersize=6, label=LABEL.get(key, key))
        lo = np.array(r["ci_lower"], dtype=float)[keep]
        hi = np.array(r["ci_upper"], dtype=float)[keep]
        ax.fill_between(x, np.maximum(lo, RESOLUTION / 10), np.maximum(hi, RESOLUTION / 10),
                        color=colour, alpha=0.15, lw=0)

    # Analytical two-hop DF reference, the curve DF should sit on.
    # P = Q(sqrt(2*Eb/N0)) -- the factor 2 matters: with Q(sqrt(Eb/N0)) this
    # curve is the 3 dB pessimistic axis and floats above the measured DF
    # instead of tracking it, which is how the error was spotted here.
    def _p(s):
        return qfunc(math.sqrt(2 * 10 ** (s / 10)))

    th = [2 * _p(s) * (1 - _p(s)) for s in x]
    ax.semilogy(x, th, color="0.35", ls=":", lw=1.6, label="Two-hop DF (theory)")

    ax.axhline(RESOLUTION, color="0.6", ls="-.", lw=1.2)
    ax.text(x.max(), RESOLUTION * 1.35, "Monte Carlo resolution floor ($10^{-5}$)  ",
            color="0.45", fontsize=9, ha="right")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("AWGN — BPSK relay comparison (companion to the canonical Rayleigh benchmark)")
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(RESOLUTION / 2, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.annotate(f"plotted to {MAX_SNR} dB: above this the true BER\n"
                f"is below the resolution floor, not measured",
                xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color="0.35")

    for out in OUT_DIRS:
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, NAME), dpi=150, bbox_inches="tight")
        print(f"  wrote {os.path.join(os.path.basename(out), NAME)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
