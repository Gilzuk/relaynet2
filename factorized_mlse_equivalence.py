"""Check that two binary trellises reproduce the joint QPSK MLSE exactly.

Dr. Khina's review (comment 5) points out that for real channel taps and
circular white Gaussian noise the QPSK MLSE metric separates into I and Q
components, so the joint 4^(L-1)-state trellis used by the coded-memory study
(Section sec:joint-latency-memory) can be replaced by two binary trellises of
2^(L-1) states with no loss. The thesis states this analytically; this script
checks it in the implementation.

For each channel length L in {3, 5, 7} (taps h_k = 0.7^k, unit-energy
normalized, as in joint_latency_memory.py), each SNR in that study's sweep and
each traceback depth it uses (plus full-sequence decoding where the joint
reference is tractable), the same received block is decoded by the joint relay
(TruncatedViterbiQPSKRelay / ViterbiMLSEQPSKRelay) and by
FactorizedViterbiQPSKRelay, and the decisions are compared symbol by symbol.

Writes results/factorized_mlse_equivalence.json.
Run: python factorized_mlse_equivalence.py
"""

import json
import os
import subprocess

import numpy as np

from relaynet.channels.e6_channels import ComplexISIChannel
from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.relays import (FactorizedViterbiQPSKRelay, TruncatedViterbiQPSKRelay,
                             ViterbiMLSEQPSKRelay)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "factorized_mlse_equivalence.json")

LS = (3, 5, 7)
SNRS = (0, 4, 8, 12, 16, 20)          # SNRS_A of joint_latency_memory.py
TRACEBACKS = (0, 1, 2, 3, 5, 15)      # TRACEBACKS_A of joint_latency_memory.py
N_SYMBOLS = 2000                      # per (L, SNR) block
FULL_SEQUENCE_LS = (3, 5)             # joint full-sequence reference loops in Python;
                                      # at L=7 (4096 states) it is intractable here


def taps_for(L):
    h = np.array([0.7 ** k for k in range(L)])
    return h / np.linalg.norm(h)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    rng = np.random.default_rng(20261012)
    rows, total, disagree = [], 0, 0
    for L in LS:
        h = taps_for(L)
        depths = list(TRACEBACKS) + [5 * L]
        for snr in SNRS:
            x = qpsk_modulate(rng.integers(0, 2, 2 * N_SYMBOLS))
            y = ComplexISIChannel(h, seed=int(rng.integers(1 << 30)))(x, snr)
            cases = [(f"D={D}", TruncatedViterbiQPSKRelay(channel_taps=h, traceback=D),
                      FactorizedViterbiQPSKRelay(h, traceback=D)) for D in sorted(set(depths))]
            if L in FULL_SEQUENCE_LS:
                cases.append(("full", ViterbiMLSEQPSKRelay(channel_taps=h),
                              FactorizedViterbiQPSKRelay(h)))
            for label, joint, fact in cases:
                a, b = joint.process(y), fact.process(y)
                d = int(np.count_nonzero(a != b))
                ser = float(np.mean(a != x))
                rows.append({"L": L, "snr_db": snr, "decoder": label,
                             "symbols": len(a), "disagreements": d,
                             "joint_symbol_error_rate": ser})
                total += len(a); disagree += d
            print(f"  L={L} {snr:2d} dB: {len(cases)} decoder settings, "
                  f"disagreements {sum(r['disagreements'] for r in rows if r['L']==L and r['snr_db']==snr)}")
    payload = {
        "description": "Symbol-by-symbol comparison of joint QPSK MLSE and the "
                       "exact two-binary-trellis factorization, real taps, circular AWGN.",
        "taps": "h_k = 0.7^k, unit-energy normalized",
        "channel_lengths": list(LS), "snrs_db": list(SNRS),
        "tracebacks": list(TRACEBACKS) + ["5L"],
        "full_sequence_lengths": list(FULL_SEQUENCE_LS),
        "symbols_per_block": N_SYMBOLS,
        "joint_states": {str(L): 4 ** (L - 1) for L in LS},
        "factorized_states_per_axis": {str(L): 2 ** (L - 1) for L in LS},
        "total_symbols_compared": total, "total_disagreements": disagree,
        "git_rev": git_rev(), "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2); fh.write("\n")
    print(f"\ncompared {total:,} symbols; disagreements: {disagree}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
