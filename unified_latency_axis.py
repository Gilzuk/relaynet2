#!/usr/bin/env python3
"""Relay cost on two processor-independent axes: decision delay and MACs.

Section~\\ref{sec:coded-latency-throughput} reports relay cost as a
wall-clock throughput in microseconds per symbol. Those figures are NumPy
on a general-purpose CPU -- MLSE is a Python loop over symbols, the MLP a
batched matrix multiply executed in C -- so the measured ratio contains a
large implementation artifact and cannot support a deployment claim.

This script replaces the timing with two quantities that depend on the
algorithm alone:

  * structural decision delay, in symbols -- how far ahead of symbol n the
    relay must observe before it can emit symbol n;
  * arithmetic, in multiply-accumulate operations per symbol.

Neither assumes a processor, a clock rate, or a symbol rate. A platform
enters only at the point of use: it supplies some budget of MACs per
symbol period, and a relay is real-time feasible exactly when its MACs
per symbol fall inside that budget. The required budget *is* the MAC
count, so no throughput parameter is invented here.

MAC accounting, stated so it can be checked. A MAC is one real
multiply-accumulate; additions and comparisons that carry no multiply are
counted separately and reported alongside, since they dominate the
trellis searches and omitting them would flatter MLSE.

  MLSE, M-ary alphabet, L taps, traceback D
      S = M^(L-1) states, M branches per state, so M^L branch metrics per
      symbol. Expected outputs are precomputed, so each branch metric is
      |y - c|^2 on a complex difference: 2 real multiplies accumulated
      = 2 MACs. Path extension and survivor selection add M^L additions
      and S*(M-1) comparisons, and traceback costs D pointer reads.

  MLP classifier, window W, hidden H, 4 classes
      A real window of 2W (I and Q), so 2*W*H MACs into the hidden layer
      and 4*H into the output head. Biases are additions; activations and
      the output softmax are transcendental, counted separately.

  Block DF, rate-1/2 K=3 convolutional code, 2 coded bits per QPSK symbol
      One trellis step per symbol: 4 states x 2 branches = 8 branch
      metrics, each correlating 2 soft bits = 2 MACs. Re-encoding is
      shifts and XORs; re-modulation is a table lookup.

  AF: one complex scaling = 2 MACs. Symbol-wise DF: two sign tests, no MAC.
"""

import json
import numpy as np

M = 4                      # QPSK
MLP_HIDDEN = 8
FRAME_SYMBOLS = 202
MEASURED_MERGE_DEPTH_L3 = 3   # from results/joint_latency_memory.json


def mlse_cost(L, D):
    """(MACs, adds, compares) per symbol for an M-ary MLSE with L taps."""
    S = M ** (L - 1)
    branches = S * M
    return 2 * branches, branches + D, S * (M - 1)


def mlp_cost(W, H=MLP_HIDDEN):
    """(MACs, adds, nonlinearities) per symbol for the QPSK classifier."""
    return 2 * W * H + 4 * H, H + 4, H + 4


def block_df_cost():
    return 8 * 2, 8 + 4, 0


def schemes():
    """(label, delay_symbols, macs, adds, other)."""
    rows = [("AF", 0, 2, 0, 0), ("DF-hard (symbol-wise)", 0, 0, 0, 2)]
    for W in (1, 3, 5, 11, 21):
        m, a, o = mlp_cost(W)
        rows.append((f"MLP w={W}", W // 2, m, a, o))
    for L in (3, 5, 7):
        D = MEASURED_MERGE_DEPTH_L3 if L == 3 else 5 * L
        m, a, c = mlse_cost(L, D)
        rows.append((f"MLSE L={L} D={D}", D, m, a, c))
    m, a, o = block_df_cost()
    rows.append(("block DF", FRAME_SYMBOLS, m, a, o))
    m2, a2, c2 = mlse_cost(3, 15)
    rows.append(("block DF + MLSE L=3", FRAME_SYMBOLS, m + m2, a + a2, o + c2))
    return rows


def crossover_taps(W, H=MLP_HIDDEN):
    """Channel memory at which MLSE's MAC count overtakes the relay's.

    MLSE costs 2*M^L MACs per symbol; the relay costs 2*W*H + 4*H,
    independent of L. Solving 2*M^L = macs_mlp for L.
    """
    macs_mlp = 2 * W * H + 4 * H
    return float(np.log(macs_mlp / 2.0) / np.log(M)), macs_mlp


def main():
    out = {"alphabet_size": M, "mlp_hidden": MLP_HIDDEN,
           "frame_symbols": FRAME_SYMBOLS, "rows": [], "crossover": {}}

    print(f"{'scheme':22s} {'delay/sym':>9s} {'MACs/sym':>9s} "
          f"{'adds':>8s} {'other':>7s}")
    print("-" * 60)
    for label, delay, macs, adds, other in schemes():
        out["rows"].append({"scheme": label, "delay_symbols": delay,
                            "macs_per_symbol": macs, "adds_per_symbol": adds,
                            "other_ops_per_symbol": other})
        print(f"{label:22s} {delay:9d} {macs:9d} {adds:8d} {other:7d}")

    print("\nMLSE / MLP w=11 MAC ratio by channel memory:")
    macs_mlp11 = mlp_cost(11)[0]
    for L in (3, 4, 5, 6, 7):
        macs = mlse_cost(L, 5 * L)[0]
        out["crossover"][f"L={L}"] = {"mlse_macs": macs,
                                      "ratio_vs_mlp_w11": macs / macs_mlp11}
        verdict = "MLSE cheaper" if macs < macs_mlp11 else "relay cheaper"
        print(f"  L={L}:  MLSE {macs:7d} MACs   "
              f"ratio {macs / macs_mlp11:8.2f}x   ({verdict})")

    for W in (5, 11, 21):
        Lx, macs_mlp = crossover_taps(W)
        out["crossover"][f"w={W}"] = {"mlp_macs": macs_mlp,
                                      "crossover_taps": Lx}
        print(f"\nMLP w={W} costs {macs_mlp} MACs/symbol; MLSE overtakes it "
              f"at L = {Lx:.2f} taps")

    with open("results/unified_latency_axis.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/unified_latency_axis.json")


if __name__ == "__main__":
    main()
