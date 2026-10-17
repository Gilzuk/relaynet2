#!/usr/bin/env python3
"""Regenerate the analytical cost panel of fig:figE6complexity.

Count real multiplications with precomputed branch expectations, as in the
main coded-memory tables. BPSK costs 2**L residual squares; square QAM with
real taps separates into two real trellises, costing 2*M**(L/2). The joint
complex trellis (2*M**L) is shown only as a diagnostic. Additions, comparisons,
traceback and activations are excluded. The 156-multiplication line belongs
only to the deployed 170-parameter BPSK network; it does not claim that this
architecture maintains accuracy for the other modulation/memory pairs.

No experiment or timing is rerun. Writes to results/ and thesis/results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAME = "e6_per_symbol_cost_vs_modulation_memory.png"
GRID = [(2, 3), (2, 5), (2, 7), (4, 3), (4, 5), (16, 3), (16, 5)]


def main():
    factorized = np.array([2**L if M == 2 else 2 * int(M**0.5)**L
                           for M, L in GRID])
    joint = np.array([2**L if M == 2 else 2 * M**L for M, L in GRID])
    mlp_mults = 11 * 13 + 13  # deployed real 11 -> 13 -> 1 network
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    x = np.arange(len(GRID))
    # Put the corrected comparison first. No duplicated BPSK diagnostic bars.
    ax.bar(x - 0.16, factorized, width=0.32, color="#009E73",
           label="MLSE: real / I,Q-factorized")
    qam = np.array([M > 2 for M, _ in GRID])
    ax.bar(x[qam] + 0.16, joint[qam], width=0.32, facecolor="none",
           edgecolor="#777777", hatch="///", linewidth=0.7,
           label="Joint QAM (diagnostic)")
    ax.axhline(mlp_mults, color="#0072B2", ls="--", lw=1.0,
               label="MLP-170, BPSK L=3 (156)")
    ax.set_yscale("log")
    ax.set_ylim(4, 4e6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{M}, {L}" for M, L in GRID], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Modulation order M, tap count L", fontsize=8)
    ax.set_ylabel("Real multiplications / symbol", fontsize=8)
    ax.set_title("(a) Analytical arithmetic counts", fontsize=9)
    ax.grid(True, axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0, 2]  # factorized comparison before the joint diagnostic
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=6.8, framealpha=0.95)
    fig.tight_layout(pad=0.5)
    for d in (ROOT / "results", ROOT / "thesis/results"):
        d.mkdir(exist_ok=True)
        fig.savefig(d / NAME, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {NAME}: primary {factorized.tolist()}, joint {joint.tolist()}, MLP {mlp_mults}")


if __name__ == "__main__":
    main()
