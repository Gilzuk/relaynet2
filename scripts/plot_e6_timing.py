#!/usr/bin/env python3
"""Draw Fig. 6.7(b), measured inference time, from committed data.

Reads e6_unknown_channel_results/e6_complexity_ported_results.npy (written by
e6_complexity_ported.py) and plots the median wall-clock time per symbol of
relaynet's Python Viterbi and the NumPy MLP relay at each measured block
length, with the speedup annotated. Replaces the hand-drawn panel that had the
ratio typed into its title.

Usage:  python3 scripts/plot_e6_timing.py
Output: results/e6_timing_measured.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "e6_unknown_channel_results", "e6_complexity_ported_results.npy")
OUT = os.path.join(ROOT, "results", "e6_timing_measured.png")


def main():
    d = np.load(SRC, allow_pickle=True).item()
    pb = d["panel_b"]
    n = np.array(sorted(pb))
    tv = np.array([pb[k][0] for k in n]) / n * 1e3   # microseconds per symbol
    tm = np.array([pb[k][1] for k in n]) / n * 1e3
    ratio = np.array([pb[k][2] for k in n])

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.plot(n, tv, "o-", color="#c0392b", label="Viterbi (Python trellis)")
    ax.plot(n, tm, "s-", color="#2471a3", label="MLP-169 (vectorized NumPy)")
    for k, a, b, r in zip(n, tv, tm, ratio):
        ax.annotate(f"{r:.0f}×", (k, np.sqrt(a * b)), ha="center", va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Block length (symbols)")
    ax.set_ylabel("Median time per symbol (µs)")
    ax.set_title(f"Measured inference time: BPSK, $L=3$\n"
                 f"median of {d.get('timing_reps', '?')} runs, single-threaded; "
                 f"speedup {ratio.min():.0f}–{ratio.max():.0f}×")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"saved {OUT}: speedup {ratio.min():.1f}-{ratio.max():.1f}x at n={list(n)}")


if __name__ == "__main__":
    main()
