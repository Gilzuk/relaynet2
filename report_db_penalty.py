"""Re-read the stored sweeps under the dB-penalty metric.

No re-run: every BER curve is already in the result JSONs, so this is a
change of metric on existing measurements, not new data.
"""

import json
import sys

import numpy as np

from ber_metrics import penalty_table, DEFAULT_TARGETS


def main(path="results/mlp_min_size_all_channels.json"):
    d = json.load(open(path))
    snrs = d["snr_db"]

    print("=" * 100)
    print("  SNR PENALTY IN dB AT A TARGET BER, vs each channel's own baseline")
    print("  (worst over the reachable targets and over 3 initializations)")
    print("=" * 100)
    print(f"  {'channel':<14} {'base':>5} {'valid':>13}  "
          f"{'best config':<22} {'dB pen':>7}  {'rel %':>9}  targets")
    for name, r in d["channels"].items():
        base = np.asarray(r.get("baseline_ber", r["df_ber"]), dtype=float)
        best = None
        for x in r["sweep"]:
            for s in x["seed_runs"]:
                p = penalty_table(snrs, s["ber"], base)
                if not p["targets_reached"]:
                    continue
                # worst target for this model, then best model over the grid
                w = p["worst_db_penalty"]
                if best is None or w < best[0]:
                    best = (w, x, p, s)
        diag = r.get("baseline_diagnostics", {})
        if best is None:
            print(f"  {name:<14} {r.get('baseline','DF'):>5} "
                  f"{diag.get('verdict','?'):>13}  "
                  f"{'-- no target reachable':<22} {'--':>7}  {'--':>9}")
            continue
        w, x, p, s = best
        cfg = f"w={x['window']} h1-{x['hidden']}p ({x['params']}p)"
        rel = 100 * max(s["worst_rel_penalty"], -9.99)
        tg = ",".join(f"{t:g}" for t in p["targets_reached"])
        print(f"  {name:<14} {r.get('baseline','DF'):>5} "
              f"{diag.get('verdict','?'):>13}  {cfg:<22} {w:>+7.2f}  "
              f"{rel:>+8.1f}%  {tg}")

    print("\n  dB pen = extra SNR the best configuration needs to hit the same BER.")
    print("  rel %  = the same model's worst-case relative BER penalty, for contrast.")
    print("  Targets are 1e-1, 1e-2, 1e-3; unreachable ones are excluded and listed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
