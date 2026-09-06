"""Coded scenario: the same two analyses the uncoded channels got.

The coded study reported worst-case relative BER penalty against block DF and
nothing else, which left two gaps the uncoded channels had filled:

  1. no dB reading, so the coded numbers were still on the metric that
     divides by a shrinking denominator
  2. no comparison against the scenario's own incumbent. Everywhere else the
     question "how small for how much degradation" is asked against the
     thesis's published relay, which here is MLP-756 (w=21, hidden 16), not
     against the classical baseline.

Both are computed here from results/coded_min_size.json; nothing is re-run.

WHY BOTH COMPARATORS MATTER HERE MORE THAN ELSEWHERE. On the uncoded channels
the published relay matches its classical baseline, so the two comparisons
give similar answers. In the coded scenario it does not: MLP-756 trails block
DF by tens of percent. "Within X of MLP-756" and "within X of block DF" are
therefore genuinely different questions, and a relay can do well on the first
while both it and MLP-756 lose the second.

Reference for the incumbent comparison is MLP-756's mean over its three
initializations; each candidate is held to its worst, as in the uncoded
tables.
"""

import json

import numpy as np

from ber_metrics import penalty_table

REF_W, REF_H = 21, 16
THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main(path="results/coded_min_size.json"):
    d = json.load(open(path))
    snrs = d["snr_db"]
    df = np.asarray(d["df_ber"], dtype=float)
    ref_row = [r for r in d["sweep"]
               if r["window"] == REF_W and r["hidden"] == REF_H][0]
    ref = np.mean([s["ber"] for s in ref_row["seed_runs"]], axis=0)
    ref_params = ref_row["params"]

    print("=" * 96)
    print("  CODED SCENARIO: dB penalty vs block DF")
    print(f"  block DF BER  " + "  ".join(f"{x:.5f}" for x in df))
    print(f"  MLP-756 BER   " + "  ".join(f"{x:.5f}" for x in ref))
    print("=" * 96)
    print(f"  {'params':>7} {'architecture':<18} {'dB vs block DF':>15} "
          f"{'rel % vs block DF':>18}")
    rows = []
    for r in sorted(d["sweep"], key=lambda z: z["params"]):
        worst = np.max([s["ber"] for s in r["seed_runs"]], axis=0)
        p_df = penalty_table(snrs, worst, df)
        p_ref = penalty_table(snrs, worst, ref)
        rows.append({
            "params": r["params"], "window": r["window"], "hidden": r["hidden"],
            "db_df": p_df["worst_db_penalty"] if p_df["targets_reached"] else np.nan,
            "db_ref": p_ref["worst_db_penalty"] if p_ref["targets_reached"] else np.nan,
            "rel_df": r["worst_rel_operational"],
        })
        tag = "   <- MLP-756" if r["params"] == ref_params else ""
        arch = "w=%d h1-%dp" % (r["window"], r["hidden"])
        print(f"  {r['params']:>7} {arch:<18} {rows[-1]['db_df']:>+15.2f} "
              f"{100 * rows[-1]['rel_df']:>+17.1f}%{tag}")

    print("\n" + "=" * 96)
    print(f"  SMALLEST CODED RELAY WITHIN A GIVEN DEGRADATION OF MLP-756 "
          f"({ref_params}p, w=21 h1-16p)")
    print("=" * 96)
    smaller = [z for z in rows
               if z["params"] < ref_params and z["db_ref"] == z["db_ref"]]
    print(f"  {'budget':>10}  {'smallest':<24} {'dB vs 756':>10} "
          f"{'shrink':>8}  {'but vs block DF':>16}")
    for th in THRESHOLDS:
        ok = [z for z in smaller if z["db_ref"] <= th]
        if not ok:
            print(f"  {'<=' + f'{th:g}' + ' dB':>10}  {'-- none --':<24}")
            continue
        b = min(ok, key=lambda z: z["params"])
        arch = "%dp  w=%d h1-%dp" % (b["params"], b["window"], b["hidden"])
        print(f"  {'<=' + f'{th:g}' + ' dB':>10}  {arch:<24} {b['db_ref']:>+10.3f} "
              f"{ref_params / b['params']:>7.1f}x  {b['db_df']:>+15.2f} dB")

    best_df = min((z for z in rows if z["db_df"] == z["db_df"]),
                  key=lambda z: z["db_df"])
    print(f"\n  Closest any size gets to block DF: {best_df['params']}p "
          f"(w={best_df['window']} h1-{best_df['hidden']}p) at "
          f"{best_df['db_df']:+.2f} dB")
    print(f"  MLP-756 itself against block DF: "
          f"{[z for z in rows if z['params'] == ref_params][0]['db_df']:+.2f} dB")
    print("\n  The incumbent and the classical baseline are far apart here, so"
          "\n  'within X dB of MLP-756' does not imply anything about block DF.")


if __name__ == "__main__":
    main()
