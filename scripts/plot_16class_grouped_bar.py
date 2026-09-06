#!/usr/bin/env python3
"""Regenerate the 4-class vs 16-class grouped bar (fig:fig51) from committed data.

The previous figure was carried over from a superseded AWGN run and showed a
cGAN that the lean re-run no longer includes. Regenerating from the JSON keeps
figure and table on one provenance -- the same mismatch has bitten this figure
once before (see Appendix F).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/all_relays_16class/all_relays_16class.json")
OUT = os.path.join(ROOT, "results/all_relays_16class/grouped_bar_16class.png")

ARCH = [("MLP", "MLP"), ("VAE", "VAE"), ("Hybrid", "Hybrid"),
        ("Transformer", "Transformer"), ("Mamba-S6", "Mamba-S6"),
        ("Mamba-2 SSD", "Mamba2")]


def main():
    d = json.load(open(SRC))
    R, i20 = d["results"], d["snr_range"].index(20)
    four = [R[f"{k} 4-cls"]["ber_mean"][i20] for _, k in ARCH]
    sixteen = [R[f"{k} 16-cls"]["ber_mean"][i20] for _, k in ARCH]

    x = np.arange(len(ARCH))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, four, w, label="4-class (per-axis I/Q)", color="#7a9cc6")
    ax.bar(x + w / 2, sixteen, w, label="16-class (joint 2D)", color="#2f5d8a")

    for xi, (a, b) in enumerate(zip(four, sixteen)):
        ax.text(xi + w / 2, b, f"{a / b:.2f}×", ha="center", va="bottom", fontsize=8)

    # DF is the bar to beat, not a relay variant, so it is a line not a column.
    ax.axhline(R["DF"]["ber_mean"][i20], ls="--", lw=1.2, color="#b03030",
               label=f"DF baseline ({R['DF']['ber_mean'][i20]:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in ARCH], rotation=15, ha="right")
    ax.set_ylabel("BER at 20 dB")
    ax.set_title("16-QAM on canonical Rayleigh: per-axis vs joint 2D classification")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
