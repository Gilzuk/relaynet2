#!/usr/bin/env python3
"""Regenerate results/unkchan_qpsk.png as a single AWGN-only panel.

The original figure (e6_qpsk_unknown_channel.py's plot()) had two panels,
AWGN and Rayleigh hop-2. Per author instruction, this table's Rayleigh-hop-2
variant is being removed (the same removal as tbl:tableE6's BPSK version),
which leaves only the AWGN panel. Other Rayleigh hop-2 studies in the same
chapter (flat unknown channels, composite cascade) are unaffected. This
script re-plots that panel from the already-committed
e6_qpsk_unknown_channel_results.npy -- the same data Table tbl:tableE6qpsk's
surviving AWGN row is checked against -- without re-running the Monte Carlo
study (which e6_qpsk_unknown_channel.py's main() would do).

Figures are written to BOTH results/ and thesis/results/, matching the
convention of scripts/plot_e6_studies.py.

Run once, to replace the checked-in two-panel PNG:

    python3 scripts/plot_e6_qpsk_unknown_channel_awgn.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(ROOT, "e6_unknown_channel_results", "e6_qpsk_unknown_channel_results.npy")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]


def main():
    d = np.load(NPY_PATH, allow_pickle=True).item()
    snrs = np.asarray(d["snrs"], dtype=float)
    summary = d["results"]["awgn"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (mu, ci) in summary.items():
        mu, ci = np.asarray(mu, dtype=float), np.asarray(ci, dtype=float)
        line, = ax.semilogy(snrs, np.maximum(mu, 1e-5), marker="o", label=name)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-6), np.maximum(mu + ci, 1e-6),
                        color=line.get_color(), alpha=0.15)

    ax.set_title("QPSK: unknown ISI $\\to$ AWGN hop 2")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "unkchan_qpsk.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
