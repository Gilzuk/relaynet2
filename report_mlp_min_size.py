"""Cross-channel summary of the MLP minimum-size study.

Reads results/mlp_min_size_all_channels.json and prints two views, because
neither alone is honest:

  relative : worst-case mean-BER penalty vs DF, the criterion the sweep uses
             to decide "matches". It gets harsher as BER falls, so at high
             SNR a negligible absolute gap can read as a several-percent
             penalty (Table 5.2's own published MLP-169 lands at +2.06% at
             20 dB this way, against an absolute gap of 0.0002).
  absolute : worst-case absolute BER gap vs DF, which is the criterion the
             thesis prose actually uses ("within a few thousandths").

A configuration's headline verdict stays the relative one -- changing the
criterion after seeing the numbers would be tuning to an answer -- but the
absolute column is printed alongside so the reader can see when a "failure"
is a few ten-thousandths of BER.
"""

import json
import sys


def worst_abs_gap(seed_runs):
    g = 0.0
    for s in seed_runs:
        for r in s["per_snr"]:
            g = max(g, r["mlp_ber"] - r["df_ber"])
    return g


def main(path="results/mlp_min_size_all_channels.json"):
    d = json.load(open(path))
    chans = d["channels"]

    print("=" * 96)
    print("  MINIMUM MLP SIZE MATCHING DF, BY CHANNEL")
    print(f"  tolerance {100*d['tolerance_rel']:.0f}% relative at every SNR, "
          f"all {len(d['train_seeds'])} inits; Wilcoxon alpha {d['alpha']}")
    print("=" * 96)
    print(f"  {'channel':<14} {'mod':<5} {'mem':>3} {'tol':>5} {'both':>5}  "
          f"{'config':<12} {'DF BER @0dB':>11} {'abs gap':>9}")
    print("  " + "-" * 92)
    for name, r in chans.items():
        # fall back to the tolerance-best config so the absolute gap is
        # always shown -- a channel where nothing passes both criteria is
        # exactly where the reader most needs to see how big the gap is
        b = r.get("best_config_both")
        if b is None:
            tol = [x for x in r["sweep"] if x["matches_tolerance_all_seeds"]]
            pool = tol or r["sweep"]
            b = min(pool, key=lambda x: (x["worst_rel_penalty_over_seeds"], x["params"]))
            mark = "~"          # not a pass, shown for scale only
        else:
            mark = " "
        cfg = f"{mark}w={b['window']} h={b['hidden']}"
        gap = worst_abs_gap(b["seed_runs"])
        print(f"  {name:<14} {r['modulation']:<5} {r['memory']:>3} "
              f"{str(r['min_params_tolerance']):>5} {str(r['min_params_both_criteria']):>5}  "
              f"{cfg:<12} {r['df_ber'][0]:>11.4f} {gap:>9.4f}")

    print("\n" + "=" * 96)
    print("  BEST CONFIGURATION PER WINDOW, PER CHANNEL")
    print("  (lowest worst-case relative penalty achieved at each window width)")
    print("=" * 96)
    windows = sorted({g["window"] for g in d["grid"]})
    print(f"  {'channel':<14} {'mem':>3} " + "".join(f"{'w=' + str(w):>16}" for w in windows))
    print("  " + "-" * 92)
    for name, r in chans.items():
        cells = []
        for w in windows:
            rows = [x for x in r["sweep"] if x["window"] == w]
            if not rows:
                cells.append(f"{'--':>16}")
                continue
            # rank by the same worst-case-over-seeds figure the verdicts use
            best = min(rows, key=lambda x: x["worst_rel_penalty_over_seeds"])
            cells.append(f"{100*best['worst_rel_penalty_over_seeds']:>+13.1f}% "
                         f"{'*' if best['matches_tolerance_all_seeds'] else ' '}")
        print(f"  {name:<14} {r['memory']:>3} " + "".join(cells))
    print("\n  * = that window's best configuration is within tolerance at every SNR")
    print("  ~ in the config column = nothing passed both criteria; shown for scale only")
    print("  Read across a row: does a wider window help on this channel?")
    print("  Memoryless channels (mem 1) should not benefit; ISI (mem 3) should.")


if __name__ == "__main__":
    main(*sys.argv[1:])
