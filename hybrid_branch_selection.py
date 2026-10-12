#!/usr/bin/env python3
"""Which branch the canonical Hybrid relay took (Sec. 4.2, Appendix B).

The Hybrid (relaynet/relays/hybrid.py) switches between its MLP sub-network and
symbol-wise DF by comparing estimate_snr(received) with a fixed 5 dB threshold.
In the canonical QPSK/Rayleigh study it runs once per axis on the zero-forced
observation, and estimate_snr assumes unit signal power, so the estimate is not
the operating SNR: the per-axis signal power is 1/2 and the zero-forced noise is
heavy-tailed. This script replays the Hop-1 observations of the canonical run
(results/modulation/qpsk_rayleigh.json: run_experiments.py section 7.10, 10 trials
of 10,000 bits per SNR, trial seed = trial index) and records the branch taken on
each axis.

The replay is exact because Source(seed) reseeds NumPy's global generator before
the bits and the Hop-1 channel are drawn. It is checked against the published
per-trial BERs: on every trial where both axes took the DF branch, the Hybrid's
stored BER must equal DF's, since neither relay draws random numbers and the
Hop-2 draws then coincide. No BER is simulated here.
"""

import json
import os

import numpy as np

from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.nodes import Source
from relaynet.relays.hybrid import estimate_snr

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "results", "modulation", "qpsk_rayleigh.json")
OUT = os.path.join(ROOT, "results", "hybrid_branch_selection.json")
THRESHOLD_DB = 5.0          # HybridRelay(snr_threshold=5.0) in run_experiments.py
BITS_PER_TRIAL = 10_000     # run_experiments.py --bits-per-trial default
N_TRIALS = 10               # run_experiments.py --num-trials default


def main():
    pub = json.load(open(SRC))
    snrs = pub["snr_range"]
    hyb = np.array(pub["results"]["Hybrid"]["ber_trials"])
    df = np.array(pub["results"]["DF"]["ber_trials"])
    assert hyb.shape == (len(snrs), N_TRIALS)

    rows, checked, mismatches = [], 0, []
    for i, snr in enumerate(snrs):
        branches, estimates = [], []
        for trial in range(N_TRIALS):
            source = Source(seed=trial, modulation="qpsk")
            _, x = source.transmit(BITS_PER_TRIAL)
            y = rayleigh_fading_channel(x, snr)
            per_axis = []
            for axis in (y.real.copy(), y.imag.copy()):
                est = float(estimate_snr(axis))
                estimates.append(est)
                per_axis.append("MLP" if est < THRESHOLD_DB else "DF")
            branches.append(per_axis)
            if per_axis == ["DF", "DF"]:
                checked += 1
                if hyb[i, trial] != df[i, trial]:
                    mismatches.append((snr, trial, hyb[i, trial], df[i, trial]))
        n_mlp = sum(b == "MLP" for pair in branches for b in pair)
        rows.append({"snr_db": snr, "axis_blocks": 2 * N_TRIALS,
                     "mlp_axis_blocks": n_mlp,
                     "median_estimate_db": float(np.median(estimates)),
                     "branches_per_trial": branches})
        print(f"{snr:4.0f} dB  MLP branch on {n_mlp:2d}/{2 * N_TRIALS} axis-blocks   "
              f"median estimate {np.median(estimates):6.1f} dB")

    print(f"\nreplay check: {checked} all-DF trials, {len(mismatches)} BER mismatches")
    if mismatches:
        raise SystemExit(f"replay does not match the published run: {mismatches[:5]}")
    out = {"source": "results/modulation/qpsk_rayleigh.json",
           "relay": "HybridRelay(snr_threshold=5.0), estimate_snr per axis",
           "threshold_db": THRESHOLD_DB, "bits_per_trial": BITS_PER_TRIAL,
           "n_trials": N_TRIALS,
           "estimate_floor_db": 100.0,
           "replay_check": {"all_df_trials": checked, "ber_mismatches": len(mismatches)},
           "rows": rows}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
