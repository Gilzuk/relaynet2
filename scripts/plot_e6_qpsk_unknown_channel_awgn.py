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


# Same colours, markers and line styles as Fig. 6.2 (plot_e6_unknown_channel_awgn.py)
# for the relays the two figures share; the taps-only trellis borrows the style of
# that figure's other imperfect-CSI trellis.
STYLE = {
    "AF": dict(color="#E69F00", marker="s", ls="--", label="AF"),
    "DF": dict(color="#D55E00", marker="o", ls="-", label="DF"),
    "MLP-QPSK": dict(color="#0072B2", marker="^", ls="-", label="MLP-QPSK (193 parameters)"),
    "Viterbi (taps only)": dict(color="#CC79A7", marker="D", ls=":", label="Viterbi (taps only)"),
    "Viterbi (genie CSI)": dict(color="#009E73", marker="v", ls="-.",
                                label="Viterbi (genie CSI: taps and fading)"),
}


def main():
    d = np.load(NPY_PATH, allow_pickle=True).item()
    snrs = np.asarray(d["snrs"], dtype=float)
    summary = d["results"]["awgn"]
    assert set(summary) == set(STYLE), sorted(summary)

    plt.rcParams.update({"font.family": "serif", "font.size": 12.5})
    # main.tex shows this figure at full text width and 0.24\textheight (about
    # 16 x 5.7 cm); this shape fills both, with the legend beside the axes.
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    for name, st in STYLE.items():
        mu, ci = (np.asarray(a, dtype=float) for a in summary[name])
        ax.semilogy(snrs, np.maximum(mu, 1e-5), markersize=5.5, lw=1.4, **st)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-6), np.maximum(mu + ci, 1e-6),
                        color=st["color"], alpha=0.15, lw=0)

    ax.set_title("QPSK: fixed three-tap ISI with per-symbol fading, followed by AWGN", loc="left")
    ax.set_xlabel(r"Per-hop $E_s/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.set_xticks(snrs[::2] if len(snrs) > 6 else snrs)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=False)
    fig.tight_layout()

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "unkchan_qpsk.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
