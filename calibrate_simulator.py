#!/usr/bin/env python3
"""Calibrate the simulator against closed-form BPSK BER, on the Eb/N0 axis.

This is experiment E1, the foundation the rest of the thesis rests on: if the
simulator does not reproduce the textbook curves, nothing measured on top of it
means anything. It deliberately uses only AF and DF, which are pure NumPy, so
the calibration can always be re-run even where the neural relays (which need
torch) cannot.

Convention, fixed here and used throughout: ``snr_db`` is Eb/N0, noise variance
is N0/2 per real dimension, and BPSK therefore obeys

    AWGN, single hop     Pb = Q(sqrt(2*Eb/N0))
    Rayleigh, single hop Pb = 0.5*(1 - sqrt(g/(1+g))),        g = Eb/N0
    two-hop DF           Pb = 2*P*(1-P), P the single-hop value

The two-hop DF form is the odd-number-of-flips probability: the destination is
wrong when exactly one of the two hops errs.

Output: results/calibration.json, consumed by verify_thesis_tables.py so that
both the theory *and* the simulation columns of the calibration table are
checked. Previously only the theory columns were verified, which left the
agreement the table asserts entirely unchecked.

Usage:  python3 calibrate_simulator.py [--bits N] [--trials M]
"""
import argparse
import json
import math
import os

import numpy as np

from relaynet.channels import awgn_channel
from relaynet.channels.fading import rayleigh_fading_channel

SNRS = [4, 10, 16]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "calibration.json")


def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def theory_awgn(snr_db):
    return qfunc(math.sqrt(2 * 10 ** (snr_db / 10.0)))


def theory_rayleigh(snr_db):
    g = 10 ** (snr_db / 10.0)
    return 0.5 * (1.0 - math.sqrt(g / (1.0 + g)))


def two_hop(p):
    return 2 * p * (1 - p)


def _run(channel, snr_db, n_bits, rng):
    bits = rng.integers(0, 2, n_bits)
    y = channel(1.0 - 2.0 * bits, snr_db)
    return float(np.mean((np.real(y) < 0).astype(int) != bits))


def simulate(kind, snr_db, n_bits, n_trials, seed0):
    """Single-hop BER, averaged over trials, with a 95% CI."""
    per = np.zeros(n_trials)
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + t)
        np.random.seed(seed0 + t)          # fading.py uses the legacy global RNG
        ch = awgn_channel if kind == "awgn" else rayleigh_fading_channel
        per[t] = _run(ch, snr_db, n_bits, rng)
    mu = float(per.mean())
    ci = float(1.96 * per.std(ddof=1) / math.sqrt(n_trials)) if n_trials > 1 else 0.0
    return mu, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2_000_000)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    print("=" * 76)
    print("E1 SIMULATOR CALIBRATION  (Eb/N0 axis, BPSK, single hop)")
    print(f"{args.trials} trials x {args.bits:,} bits per SNR point")
    print("=" * 76)

    out = {"snrs": SNRS, "n_bits": args.bits, "n_trials": args.trials,
           "convention": "snr_db is Eb/N0; noise variance N0/2 per real dimension",
           "results": {}}

    worst = 0.0
    for kind, theory in (("awgn", theory_awgn), ("rayleigh", theory_rayleigh)):
        out["results"][kind] = {}
        print(f"\n{kind.upper()}")
        print(f"  {'SNR':>4} {'theory':>12} {'sim':>12} {'95% CI':>10} {'rel.err':>9}")
        for i, s in enumerate(SNRS):
            th = theory(s)
            mu, ci = simulate(kind, s, args.bits, args.trials, seed0=1000 * (i + 1))
            rel = abs(mu - th) / th if th > 0 else float("nan")
            # only meaningful where the budget can resolve the theoretical value
            resolvable = th * args.bits * args.trials >= 20
            if resolvable:
                worst = max(worst, rel)
            flag = "" if resolvable else "   (below resolution)"
            out["results"][kind][str(s)] = {
                "theory": th, "sim_mean": mu, "sim_ci": ci,
                "resolvable": bool(resolvable),
            }
            print(f"  {s:>4} {th:12.4e} {mu:12.4e} {ci:10.2e} {rel:8.1%}{flag}")

    out["max_relative_error_resolvable"] = worst
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nWorst relative error over resolvable points: {worst:.2%}")
    print(f"Written to {OUT}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
