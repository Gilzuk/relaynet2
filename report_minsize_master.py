"""One table: channel taps, window, and the parameter count per dB budget.

Consolidates the uncoded and coded analyses into a single view. Each cell is
the smallest relay whose degradation against that scenario's own published
relay stays within the column's dB budget, written as "<params>p w<window>"
so the window is visible next to the size -- window is the axis that actually
moves, and a parameter count alone hides it.

Reference per scenario is the thesis's own relay: MLP-169 (w=5, h1-24p) on
the uncoded channels, MLP-756 (w=21, h1-16p) coded. Reference uses its mean
over three initializations; every candidate is held to its worst, so a size
that appears in a cell works whichever way it initializes.

Degradation is measured as extra SNR needed to reach the same BER. The final
column carries what that cell costs against the *classical* baseline, which
is a different question and on the coded row a very different answer.
"""

import json

import numpy as np

from ber_metrics import penalty_table

BUDGETS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def rows_for(sweep, snrs, ref_w, ref_h, base_ber):
    ref_row = [x for x in sweep if x["window"] == ref_w and x["hidden"] == ref_h]
    if not ref_row:
        return None, None, None
    ref = np.mean([s["ber"] for s in ref_row[0]["seed_runs"]], axis=0)
    ref_params = ref_row[0]["params"]
    out = []
    for x in sweep:
        if x["params"] >= ref_params:
            continue
        worst = np.max([s["ber"] for s in x["seed_runs"]], axis=0)
        p_ref = penalty_table(snrs, worst, ref)
        if not p_ref["targets_reached"]:
            continue
        p_base = penalty_table(snrs, worst, base_ber)
        out.append({"params": x["params"], "window": x["window"],
                    "hidden": x["hidden"],
                    "db_ref": p_ref["worst_db_penalty"],
                    "db_base": (p_base["worst_db_penalty"]
                                if p_base["targets_reached"] else np.nan)})
    return out, ref_params, ref


def main():
    d = json.load(open("results/mlp_min_size_all_channels.json"))
    snrs = d["snr_db"]
    table = []

    for name, r in d["channels"].items():
        base = np.asarray(r.get("baseline_ber", r["df_ber"]), dtype=float)
        rows, refp, _ = rows_for(r["sweep"], snrs, 5, 24, base)
        if rows is None:
            continue
        table.append({"name": name, "taps": r["memory"], "mod": r["modulation"],
                      "ref": f"MLP-{refp}", "baseline": r["baseline"],
                      "rows": rows})

    c = json.load(open("results/coded_min_size.json"))
    crows, crefp, _ = rows_for(c["sweep"], c["snr_db"], 21, 16,
                               np.asarray(c["df_ber"], dtype=float))
    if crows is not None:
        table.append({"name": "coded", "taps": "code", "mod": "qpsk",
                      "ref": f"MLP-{crefp}", "baseline": "block DF",
                      "rows": crows})

    head = (f"| channel | taps | mod | reference | "
            + " | ".join(f"≤{b:g} dB" for b in BUDGETS)
            + " | vs classical at ≤0.5 dB cell |")
    sep = "|---|---|---|---|" + "---|" * (len(BUDGETS) + 1)
    lines = [head, sep]
    for t in table:
        cells = []
        half = None
        for b in BUDGETS:
            ok = [z for z in t["rows"] if z["db_ref"] <= b]
            if ok:
                z = min(ok, key=lambda q: q["params"])
                cells.append(f"{z['params']}p w{z['window']}")
                if b == 0.5:
                    half = z
            else:
                cells.append("—")
        tail = (f"{half['db_base']:+.2f} dB"
                if half is not None and half["db_base"] == half["db_base"]
                else "—")
        lines.append(f"| {t['name']} | {t['taps']} | {t['mod']} | {t['ref']} | "
                     + " | ".join(cells) + f" | {tail} |")

    md = "\n".join(lines)
    print(md)
    with open("results/minsize_master_table.md", "w") as fh:
        fh.write("# Smallest relay per dB budget, against each scenario's "
                 "own published relay\n\n"
                 "Cell = smallest configuration meeting the budget, as "
                 "`<parameters>p w<window>`.\nDash = no smaller size stays "
                 "within that budget.\n\n" + md + "\n")
    print("\nwrote results/minsize_master_table.md")


if __name__ == "__main__":
    main()
