#!/usr/bin/env python3
"""Regenerate panel (a) of fig:figE6complexity (Section 6.5): per-symbol cost.

The panel was previously a checked-in PNG with no generating script. It is
analytical, so it is rebuilt here from the counting rules the thesis states:

  Viterbi MLSE  the joint trellis, M^L branch metrics per symbol at about
                eight operations each -- e6_complexity_ported.viterbi_ops,
                64 operations for BPSK with L = 3 (Section 6.5).
  MLP           the deployed BPSK, L = 3 relay: the real 11-13-1 network,
                170 parameters, 156 MACs plus 14 additions = 326 operations,
                "about 330" in the text. It is drawn as a reference line,
                not as a bar at every (M, L): Section 6.5 says this count is a
                constant of the deployed architecture, not of the channel,
                since the window and output size grow with L and M.

The earlier PNG labelled the network "MLP-169", the composite-study I/Q relay,
while the caption and text describe the 170-parameter relay; both cost about
330 operations per symbol, so the bars are unchanged and only the label moves.
Panel (b), the measured timing, is not regenerated: its timings were never
committed (e6_complexity_ported.py writes them only when re-measured).

Writes to both results/ and thesis/results/, like the other plot scripts.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from e6_complexity_ported import mlp_ops, viterbi_ops  # noqa: E402
from relaynet.relays import MLPRelay  # noqa: E402

OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]
NAME = "e6_per_symbol_cost_vs_modulation_memory.png"
GRID = [(2, 3), (2, 5), (2, 7), (4, 3), (4, 5), (16, 3), (16, 5)]


def main():
    vit = np.array([viterbi_ops(M, L)[2] for M, L in GRID], dtype=float)
    mlp = MLPRelay(input_size=11, hidden_size=13, output_size=1, window_size=None, seed=0)
    params, macs, flops = mlp_ops(mlp)
    assert params == 170 and macs == 156, (params, macs)

    fig, ax = plt.subplots(figsize=(3.0, 2.05))
    x = np.arange(len(GRID))
    ax.bar(x, vit, width=0.6, color="tab:purple", label=r"Viterbi MLSE, joint trellis ($8M^{L}$)")
    ax.axhline(flops, color="tab:blue", ls="--", lw=1.2,
               label=f"MLP-{params}, deployed BPSK $L{{=}}3$ (~330)")
    ax.set_yscale("log")
    ax.set_ylim(10, 3e8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$M{{=}}{M}, L{{=}}{L}$" for M, L in GRID], rotation=35, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("Operations per symbol", fontsize=7.5)
    ax.set_title(r"(a) Per-symbol cost vs modulation $M$, memory $L$", fontsize=8)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(loc="upper left", fontsize=5.8, framealpha=0.9)
    fig.tight_layout()
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, NAME), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {NAME}: Viterbi {vit.astype(int).tolist()}, MLP-{params} {flops} ops/symbol")


if __name__ == "__main__":
    main()
