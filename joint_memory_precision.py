#!/usr/bin/env python3
"""Re-measure the memory sweep at a trial count that can resolve 1e-5.

joint_latency_memory.py runs Part B at 90,000 information bits per point.
At the MLSE error rates that produces, the L=7 cell is a single bit error
and the L=3 and L=5 cells are three and six -- counts from which no ratio
can honestly be quoted, and from which Chapter 8's "at most a factor of
23" was in fact quoted. This script re-runs that sweep only, with fifty
times the bits and five seeds instead of three, and reports a Wilson
score interval on every cell so the resolution limit is visible in the
table rather than hidden behind a point estimate.

A ratio between two cells is reported as resolvable only when their
intervals do not overlap.
"""

import json
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays import TruncatedViterbiQPSKRelay
import joint_latency_memory as J

N_FRAMES = 1500          # 300,000 info bits per trial
SEEDS = (0, 1, 2, 3, 4)  # 1,500,000 per point
SNR_DB = 12


def wilson(errors, n, z=1.96):
    """Wilson score interval -- correct at the small error counts here."""
    if n == 0:
        return (0.0, 0.0)
    p = errors / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def main():
    J.N_FRAMES = N_FRAMES
    enc, dec = ConvolutionalEncoder(3), ViterbiCodeDecoder(3)
    fs = J.FRAME_INFO_BITS + dec.num_tail
    bits_per_trial = N_FRAMES * J.FRAME_INFO_BITS

    out = {"n_frames": N_FRAMES, "seeds": list(SEEDS), "snr_db": SNR_DB,
           "info_bits_per_point": bits_per_trial * len(SEEDS), "rows": []}

    for L in (3, 5, 7):
        taps = J.taps_for(L)
        mlps = {s: J.train_mlp(L, 11, s) for s in SEEDS}
        cases = [
            ("MLP w=11", 5, None, lambda s: mlps[s]),
            (f"MLSE D={5*L}", 5 * L, 4 ** (L - 1),
             lambda s, t=taps, L=L: TruncatedViterbiQPSKRelay(channel_taps=t,
                                                              traceback=5 * L)),
            ("block DF", fs, None,
             lambda s: CodedDecodeAndForwardRelay(frame_info_bits=J.FRAME_INFO_BITS)),
        ]
        for label, latency, states, factory in cases:
            errs = 0
            for s in SEEDS:
                ber, _ = J.run_trial(factory(s), SNR_DB, s, L, enc, dec, fs)
                errs += int(round(ber * bits_per_trial))
            n = bits_per_trial * len(SEEDS)
            lo, hi = wilson(errs, n)
            row = {"channel_taps_L": L, "scheme": label, "latency_symbols": latency,
                   "states": states, "bit_errors": errs, "info_bits": n,
                   "ber": errs / n, "ci95": [lo, hi]}
            out["rows"].append(row)
            print(f"L={L} {label:12s} errors={errs:6d}/{n}  BER={errs/n:.3e}  "
                  f"95% CI [{lo:.3e}, {hi:.3e}]", flush=True)

    # a ratio is only quotable when the two intervals are disjoint
    for L in (3, 5, 7):
        r = {x["scheme"].split()[0]: x for x in out["rows"]
             if x["channel_taps_L"] == L and not x["scheme"].startswith("block")}
        mlp, mlse = r["MLP"], r["MLSE"]
        disjoint = mlse["ci95"][1] < mlp["ci95"][0]
        ratio = (mlp["ber"] / mlse["ber"]) if mlse["ber"] > 0 else float("inf")
        lo = mlp["ci95"][0] / mlse["ci95"][1] if mlse["ci95"][1] > 0 else float("inf")
        hi = mlp["ci95"][1] / mlse["ci95"][0] if mlse["ci95"][0] > 0 else float("inf")
        out.setdefault("ratios", []).append(
            {"channel_taps_L": L, "point_ratio": ratio,
             "ratio_range": [lo, hi], "intervals_disjoint": bool(disjoint)})
        print(f"L={L} MLP/MLSE ratio {ratio:.1f} (range {lo:.1f}-{hi:.1f}) "
              f"{'resolvable' if disjoint else 'NOT RESOLVABLE'}", flush=True)

    with open("results/joint_memory_precision.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/joint_memory_precision.json")


if __name__ == "__main__":
    main()
