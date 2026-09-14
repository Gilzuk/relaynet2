"""Remeasure the 16 dB MLP cell of tbl:tableE6 under a predeclared bit budget.

The 16--20 dB cells of Chapter 7 were produced by a stopping rule that
transmits to the first error at N1 bits and then continues to 10*N1. An
examiner review objected that the exposure is therefore data-dependent, which
makes the estimator biased; rare_event_estimator_bias.py measures that bias
(median 1.055x, 90% range 0.55--3.22x). This script removes the objection at
its source for the one cell it applies to.

Which cell that is matters. At 18 and 20 dB no error occurred, so the run hit
the predeclared cap of 1e10 bits and the budget was never retargeted: those
entries are already fixed-budget rule-of-three bounds and need no rework. Only
the 16 dB cell had its exposure set by the data.

Here the budget is declared before the run and does not depend on what is
observed: FIXED_BITS bits per training seed, all three seeds, errors counted
over the whole exposure. With no optional stopping the count is Binomial at
fixed n, so an exact Poisson (Garwood) interval applies.

Writes results/rare_event_fixed_budget_16db.json (into the repository, never
/tmp). Run: python rare_event_fixed_budget.py
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import chi2

import e6_sim_ported as e6

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(ROOT, "results", "rare_event_fixed_budget_16db.json")

SNR_DB = 16
FIXED_BITS = 1_000_000_000     # predeclared, per training seed
BLOCK_BITS = 2_000_000         # streaming block; does not affect the budget
SEEDS = (0, 1, 2)              # the same three training seeds as tbl:tableE6


def poisson_interval(k, exposure, conf=0.95):
    """Exact (Garwood) two-sided interval on a rate from k events."""
    a = 1.0 - conf
    lo = chi2.ppf(a / 2, 2 * k) / 2 / exposure if k > 0 else 0.0
    hi = chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2 / exposure
    return float(lo), float(hi)


def measure(mlp, hop1, hop2, source, destination, snr_db, total_bits):
    """Count errors over exactly `total_bits` bits. No early exit."""
    errors, sent = 0, 0
    while sent < total_bits:
        n = min(BLOCK_BITS, total_bits - sent)
        tx_bits, tx_symbols = source.transmit(n)
        rx = destination.receive(hop2(mlp.process(hop1(tx_symbols, snr_db)), snr_db))
        errors += int(np.count_nonzero(tx_bits != rx))
        sent += n
    return errors, sent


def main(src_base=1000, out_path=None):
    out_path = out_path or OUT_DEFAULT
    hop1 = e6.create_channel("isi", seed=1)
    hop2 = e6.create_channel("awgn", seed=2)
    destination = e6.Destination(modulation="bpsk")

    per_seed, t_start = [], time.time()
    for seed in SEEDS:
        mlp, n_params = e6.train_mlp(hop1, seed=seed)
        source = e6.Source(seed=src_base + seed, modulation="bpsk")
        t0 = time.time()
        errors, sent = measure(mlp, hop1, hop2, source, destination,
                               SNR_DB, FIXED_BITS)
        lo, hi = poisson_interval(errors, sent)
        per_seed.append({"seed": seed, "n_params": int(n_params),
                         "errors": errors, "bits": sent,
                         "ber": errors / sent, "ci95": [lo, hi],
                         "seconds": round(time.time() - t0, 1)})
        print(f"  seed {seed}: {errors} errors / {sent:,} bits "
              f"-> {errors / sent:.3e}  [{lo:.2e}, {hi:.2e}]  "
              f"({per_seed[-1]['seconds']:.0f}s)")

    tot_e = sum(r["errors"] for r in per_seed)
    tot_b = sum(r["bits"] for r in per_seed)
    lo, hi = poisson_interval(tot_e, tot_b)

    payload = {
        "description": "Fixed predeclared-budget remeasurement of the 16 dB "
                       "MLP cell of tbl:tableE6. No data-dependent stopping.",
        "snr_db": SNR_DB,
        "fixed_bits_per_seed": FIXED_BITS,
        "declared_before_run": True,
        "note_18_20_db": "Not remeasured: those cells saw no error, so the run "
                         "reached the predeclared 1e10-bit cap and the budget "
                         "was never retargeted. They are already fixed-budget "
                         "rule-of-three bounds.",
        "previous_adaptive_estimate": 4.79e-8,
        "per_seed": per_seed,
        "pooled": {"errors": tot_e, "bits": tot_b, "ber": tot_e / tot_b,
                   "ci95_garwood": [lo, hi]},
        "total_seconds": round(time.time() - t_start, 1),
    }
    payload["source_seed_base"] = src_base
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"\npooled: {tot_e} errors / {tot_b:,} bits = {tot_e / tot_b:.3e}")
    print(f"exact 95% Poisson interval [{lo:.3e}, {hi:.3e}]")
    print(f"previous adaptive estimate  {4.79e-8:.3e}")
    print(f"wrote {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-base", type=int, default=1000,
                    help="base seed for the bit source; change it to run an "
                         "independent replication on the same three networks")
    ap.add_argument("--out", default=None, help="output JSON path")
    a = ap.parse_args()
    main(src_base=a.src_base, out_path=a.out)
