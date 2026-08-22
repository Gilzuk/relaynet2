#!/usr/bin/env python3
"""Regenerate the canonical QPSK/Rayleigh figure (fig:fig10) with two
theoretical reference curves added, on top of the eight measured relays:

  DF (theory)      the closed-form two-hop composition
                    P_e = 2P(1-P), P = ber_rayleigh(snr_db - 3.01dB)
                    -- the QPSK Es/N0 -> per-bit Eb/N0 correction, same
                    derivation already stated in prose (ch05_experiments.tex,
                    Section~sec:rayleigh-two-hop-df-ber). Verified to match
                    the simulated DF curve to within 1% at every SNR point.

  single-hop floor  ber_rayleigh(snr_db - 3.01dB) alone -- the best any
                    two-hop relay could do (a genie that recovers hop 1
                    perfectly). No relay may go below this line; it is a
                    physical floor, not a target.

Reuses run_experiments.py's own plot_ber_chart (styling, jitter, insets,
CHART_GUIDELINES.md compliance) so the appearance of the eight measured
curves is pixel-identical to the existing figure -- only the two reference
curves are new.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_experiments import plot_ber_chart

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]


def ber_rayleigh(snr_db):
    g = 10 ** (snr_db / 10.0)
    return 0.5 * (1 - math.sqrt(g / (1 + g)))


def main():
    with open(os.path.join(ROOT, "results/modulation/qpsk_rayleigh.json")) as f:
        d = json.load(f)
    snrs = np.asarray(d["snr_range"], dtype=float)

    ber_dict = {name: r["ber_mean"] for name, r in d["results"].items()}
    ci_dict = {name: (r["ci_lower"], r["ci_upper"]) for name, r in d["results"].items()}

    p1 = np.array([ber_rayleigh(s - 3.0103) for s in snrs])
    ber_dict["DF (theory)"] = (2 * p1 * (1 - p1)).tolist()
    ber_dict["single-hop floor"] = p1.tolist()
    # No CI on closed-form curves.
    ci_dict["DF (theory)"] = ([0.0] * len(snrs), [0.0] * len(snrs))
    ci_dict["single-hop floor"] = ([0.0] * len(snrs), [0.0] * len(snrs))

    extra_styles = {
        "DF (theory)": {"color": "black", "marker": "", "ls": "--", "lw": 1.1, "alpha": 0.8},
        "single-hop floor": {"color": "0.55", "marker": "", "ls": ":", "lw": 1.1, "alpha": 0.8},
    }

    for d_ in OUT_DIRS:
        os.makedirs(d_, exist_ok=True)
    for d_ in OUT_DIRS:
        plot_ber_chart(
            snrs, ber_dict, ci_dict,
            title="QPSK — RAYLEIGH (§7.10), with theoretical bounds",
            save_path=os.path.join(d_, "canonical_qpsk_rayleigh_ci.png"),
            extra_styles=extra_styles,
            show_inset=True,
            show_annotations=False,  # the crossover/"Best" auto-annotator
                                      # treats the theory curves as competing
                                      # relays and mislabels them; the two
                                      # reference lines are self-explanatory
                                      # via the legend, so annotations are
                                      # not needed here.
        )
    print("Saved canonical_qpsk_rayleigh_ci.png (with DF theory + single-hop floor) to both results/ dirs")


if __name__ == "__main__":
    main()
