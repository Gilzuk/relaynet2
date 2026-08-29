"""Per channel: the smallest relay that costs almost nothing against MLP-169.

Comparator is the thesis's own 169-parameter relay (w=5, h1-24p), not a
classical baseline, so the answer does not depend on how good DF or MLSE
happen to be on a given channel. Both sides are the same kind of object,
trained the same way, on identical channel draws and payload bits.

Reference is MLP-169's mean BER over its three initializations; a single one
is a draw, and the question is about the architecture. Each candidate is held
to its *worst* initialization, so a size counts only if it works whichever way
it initializes. Worst-of-three against a mean is deliberately unfair to the
candidate: a size that clears it has cleared a conservative bar.

Degradation is reported in dB -- the extra SNR the smaller relay needs to
reach the same BER -- because a relative BER penalty divides by a shrinking
denominator and turns negligible absolute gaps into large percentages. The
relative figure is printed alongside for continuity with the earlier tables.

"<= 0 dB" means the smaller relay is at least as good as MLP-169 everywhere
the comparison is defined, not merely close to it.
"""

import json

import numpy as np

from ber_metrics import penalty_table

REF_W, REF_H = 5, 24
THRESHOLDS = [0.0, 0.1, 0.25, 0.5, 1.0]


def analyse(path="results/mlp_min_size_all_channels.json"):
    d = json.load(open(path))
    snrs = d["snr_db"]
    table = {}

    for name, r in d["channels"].items():
        ref_row = [x for x in r["sweep"]
                   if x["window"] == REF_W and x["hidden"] == REF_H]
        if not ref_row:
            continue
        ref = np.mean([s["ber"] for s in ref_row[0]["seed_runs"]], axis=0)
        ref_params = ref_row[0]["params"]

        rows = []
        for x in r["sweep"]:
            if x["params"] >= ref_params:
                continue                      # only interested in smaller
            worst = np.max([s["ber"] for s in x["seed_runs"]], axis=0)
            p = penalty_table(snrs, worst, ref)
            if not p["targets_reached"]:
                continue
            usable = [i for i in range(len(ref)) if ref[i] > 1e-5]
            rel = max((worst[i] - ref[i]) / ref[i] for i in usable) if usable else np.nan
            rows.append({"params": x["params"], "window": x["window"],
                         "hidden": x["hidden"], "db": p["worst_db_penalty"],
                         "rel": rel})
        table[name] = {"memory": r["memory"], "modulation": r["modulation"],
                       "ref_params": ref_params, "rows": rows}
    return table, snrs


def main():
    table, _ = analyse()

    print("=" * 104)
    print("  SMALLEST RELAY WITHIN A GIVEN DEGRADATION OF MLP-169 (w=5 h1-24p)")
    print("  cell = parameters (architecture); '--' = no smaller size stays "
          "within that budget")
    print("=" * 104)
    hdr = "".join(f"{'<=' + f'{t:g}' + ' dB':>22}" for t in THRESHOLDS)
    print(f"  {'channel':<14} {'mem':>3}" + hdr)
    print("  " + "-" * 100)

    for name, t in table.items():
        cells = []
        for th in THRESHOLDS:
            ok = [z for z in t["rows"] if z["db"] <= th]
            if ok:
                b = min(ok, key=lambda z: z["params"])
                cells.append(f"{b['params']:>5}p w{b['window']} h1-{b['hidden']}p".rjust(22))
            else:
                cells.append(f"{'--':>22}")
        print(f"  {name:<14} {t['memory']:>3}" + "".join(cells))

    print("\n" + "=" * 104)
    print("  DETAIL: the smallest size with no degradation at all (<= 0 dB)")
    print("=" * 104)
    print(f"  {'channel':<14} {'mem':>3}  {'architecture':<22} "
          f"{'params':>7} {'dB':>7} {'rel %':>9} {'shrink':>8}")
    for name, t in table.items():
        ok = [z for z in t["rows"] if z["db"] <= 0.0]
        if not ok:
            if t["rows"]:
                b = min(t["rows"], key=lambda z: z["db"])
                note = (f"best is {b['db']:+.2f} dB at {b['params']}p "
                        f"(w={b['window']} h1-{b['hidden']}p)")
            else:
                note = "no smaller size has a defined comparison"
            print(f"  {name:<14} {t['memory']:>3}  {'-- none --':<22} {note}")
            continue
        b = min(ok, key=lambda z: z["params"])
        arch = f"w={b['window']} h1-{b['hidden']}p"
        shrink = t["ref_params"] / b["params"]
        print(f"  {name:<14} {t['memory']:>3}  {arch:<22} {b['params']:>7} "
              f"{b['db']:>+7.2f} {100 * b['rel']:>+8.1f}% {shrink:>7.1f}x")
    print()


if __name__ == "__main__":
    main()
