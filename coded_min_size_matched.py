#!/usr/bin/env python3
"""Coded size sweep against a matched-metric block-DF baseline (C4b).

Reruns only the block-DF baseline of coded_min_size.py with |h|^2-weighted relay
and destination metrics, on the same seeds, and recomputes every penalty from the
learned-relay curves stored in results/coded_min_size.json. Writes
results/coded_min_size_matched.json. Run with CODED_METRIC=matched.
"""

import json

import coded_df_experiment as CE
import coded_metric as CM
from coded_min_size import (SNRS, N_FRAMES, OPERATIONAL_FER, evaluate, summarize,
                            ConvolutionalEncoder, ViterbiCodeDecoder,
                            CodedDecodeAndForwardRelay)


def main():
    """Matched metric: rerun only the block-DF baseline with |h|^2-weighted
    relay and destination metrics, on the same seeds, and recompute every
    penalty from the stored learned-relay curves. The learned relays keep the
    published (unweighted) destination decoder, so their curves are unchanged
    and are not retrained."""
    CE.N_FRAMES = N_FRAMES
    encoder, decoder = ConvolutionalEncoder(), ViterbiCodeDecoder()
    frame_symbols = CE.FRAME_INFO_BITS + decoder.num_tail
    pub = json.load(open("results/coded_min_size.json"))
    df = CodedDecodeAndForwardRelay(frame_info_bits=CE.FRAME_INFO_BITS)
    df_ber, df_fer = evaluate(df, encoder, decoder, frame_symbols,
                              "block DF, matched (0 params)")
    usable = [i for i, f in enumerate(df_fer) if f < OPERATIONAL_FER]
    rows = []
    for r in pub["sweep"]:
        runs = []
        for run in r["seed_runs"]:
            ber, fer = run["ber"], run["fer"]
            rel_all = [(ber[i] - df_ber[i]) / df_ber[i] if df_ber[i] > 0 else float("inf")
                       for i in range(len(SNRS))]
            rel_op = [rel_all[i] for i in usable]
            runs.append(dict(run, worst_rel_all=float(max(rel_all)),
                             worst_rel_operational=float(max(rel_op)) if rel_op else float("nan"),
                             beats_df_fer_at=[SNRS[i] for i in range(len(SNRS)) if fer[i] < df_fer[i]]))
        rows.append(dict(r, seed_runs=runs,
                         worst_rel_operational=float(max(x["worst_rel_operational"] for x in runs)),
                         best_rel_operational=float(min(x["worst_rel_operational"] for x in runs)),
                         worst_rel_all=float(max(x["worst_rel_all"] for x in runs))))
    out = dict(pub, metric="matched (block DF only; learned relays unweighted destination)",
               operational_snrs=[SNRS[i] for i in usable],
               df_ber=[float(x) for x in df_ber], df_fer=[float(x) for x in df_fer], sweep=rows)
    with open(CM.path("results/coded_min_size.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  saved {CM.path('results/coded_min_size.json')}")
    summarize(out)



if __name__ == "__main__":
    if not CM.MATCHED:
        raise SystemExit("run with CODED_METRIC=matched")
    main()
