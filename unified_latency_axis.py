#!/usr/bin/env python3
"""Put decision delay and computation on one latency axis.

Section~\\ref{sec:coded-latency-throughput} reports a relay's cost as two
unrelated quantities: a structural decision delay in symbols, and a
wall-clock throughput in microseconds per symbol. They are not unrelated.
Once a symbol rate is fixed, arithmetic per symbol *is* latency -- a relay
that needs longer than one symbol period to produce a symbol cannot run in
real time at all, and one that fits inside the period contributes a bounded
pipeline delay on top of its decision delay.

The wall-clock figures measured in joint_latency_memory.py cannot be used
for this. They are NumPy on a general-purpose CPU: MLSE is a Python loop
over symbols while the MLP is a batched matrix multiply executed in C, so
the measured ratio contains a large implementation artifact that has
nothing to do with the algorithms. This script counts operations per
symbol instead, which is implementation-independent, and converts to time
under a stated processor throughput.

Operation accounting, stated so it can be checked:

  MLSE, M-ary alphabet, L taps, traceback D
      S = M^(L-1) states, M branches per state.
      branch metric   S*M * 5     (complex subtract 2, |.|^2 3)
      add-compare-sel S*M * 1  +  S*(M-1) * 1
      traceback       D           (one pointer chase per step)

  MLP classifier, window W, hidden H, 4 classes
      2*(2*W*H) + 2*(4*H) multiply-accumulates, H activations,
      4 output exponentials.

  Block DF, rate-1/2 K=3 code, 2 coded bits per QPSK symbol
      code trellis has 4 states, 2 branches; per coded bit
      4*2 * (metric + add + compare) = 24, twice per symbol, plus
      re-encode and re-modulate.

  AF: one complex scale. Symbol-wise DF: two sign tests.
"""

import json
import numpy as np

M = 4                      # QPSK
MLP_HIDDEN = 8
FRAME_SYMBOLS = 202
SYMBOL_RATES_MSPS = (0.1, 1.0, 10.0)
THROUGHPUT_GOPS = 1.0      # a modest embedded DSP, stated not measured


def ops_mlse(L, D):
    S = M ** (L - 1)
    return S * M * 5 + S * M + S * (M - 1) + D


def ops_mlp(W, H=MLP_HIDDEN):
    return 2 * (2 * W * H) + 2 * (4 * H) + H + 4


def ops_block_df():
    return 2 * 24 + 6


def schemes():
    """(label, structural_delay_symbols, ops_per_symbol)."""
    rows = [("AF", 0, 2), ("DF-hard (symbol-wise)", 0, 4)]
    for W in (1, 3, 5, 11, 21):
        rows.append((f"MLP w={W}", W // 2, ops_mlp(W)))
    for L in (3, 5, 7):
        D = 3 if L == 3 else 5 * L          # measured merge depth at L=3
        rows.append((f"MLSE L={L} D={D}", D, ops_mlse(L, D)))
    rows.append(("block DF", FRAME_SYMBOLS, ops_block_df()))
    rows.append((f"block DF + MLSE L=3", FRAME_SYMBOLS,
                 ops_block_df() + ops_mlse(3, 15)))
    return rows


def main():
    ops_per_us = THROUGHPUT_GOPS * 1e3       # giga-ops/s -> ops per microsecond
    out = {"symbol_rates_msps": list(SYMBOL_RATES_MSPS),
           "throughput_gops": THROUGHPUT_GOPS, "mlp_hidden": MLP_HIDDEN,
           "rows": []}

    print(f"processor throughput {THROUGHPUT_GOPS} Gop/s\n")
    header = f"{'scheme':22s} {'delay':>6s} {'ops/sym':>9s} {'t_comp':>9s}"
    for rs in SYMBOL_RATES_MSPS:
        header += f" | {rs:>4g} Msym/s"
    print(header)
    print("-" * len(header))

    for label, delay_sym, ops in schemes():
        t_comp_us = ops / ops_per_us
        row = {"scheme": label, "delay_symbols": delay_sym,
               "ops_per_symbol": ops, "compute_us_per_symbol": t_comp_us,
               "by_symbol_rate": {}}
        line = f"{label:22s} {delay_sym:6d} {ops:9d} {t_comp_us:8.3f}us"
        for rs in SYMBOL_RATES_MSPS:
            ts_us = 1.0 / rs                       # symbol period, microseconds
            feasible = t_comp_us <= ts_us
            total_us = delay_sym * ts_us + t_comp_us
            row["by_symbol_rate"][f"{rs}Msps"] = {
                "symbol_period_us": ts_us, "realtime_feasible": bool(feasible),
                "total_latency_us": total_us if feasible else None,
                "compute_over_budget_x": None if feasible else t_comp_us / ts_us}
            line += (f" | {total_us:9.2f}us" if feasible
                     else f" | {t_comp_us/ts_us:7.0f}x over")
        out["rows"].append(row)
        print(line)

    print("\n'Nx over' = cannot run in real time; arithmetic per symbol exceeds")
    print("the symbol period, so the backlog grows without bound.")

    with open("results/unified_latency_axis.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/unified_latency_axis.json")


if __name__ == "__main__":
    main()
