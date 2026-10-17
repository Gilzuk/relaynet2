"""Paired tests behind the Chapter 5 statements about which canonical relays differ.

Every relay of Tables 5.2 and 5.3 is evaluated on the same ten trials (seeds 0-9,
shared by every relay at every SNR), so each pair of relays has ten paired
per-trial BERs per SNR. Section 4.3.2 declares the two-sided Wilcoxon
signed-rank test on those pairs, uncorrected for multiplicity; with ten pairs
its smallest attainable p is 2/1024 ~ 0.002. This script applies it to

* every learned relay against symbol-wise DF, in both tables;
* the three feedforward relays of Table 5.2 against one another;
* every learned relay against the 169-parameter MLP;

and records, per SNR, the mean paired difference (first minus second, so a
positive value means the first has the higher BER), the p-value and whether
p < 0.05. It reads only the committed result files; nothing is retrained.

Output: results/canonical_paired_tests.json
Usage:  python canonical_paired_tests.py
"""
import json
import warnings

import numpy as np
from scipy.stats import wilcoxon

OUT = "results/canonical_paired_tests.json"
FILES = {"table_5_2": "results/modulation/qpsk_rayleigh.json",
         "table_5_3": "results/normalized_3k/3k_rayleigh.json"}
ALPHA = 0.05


def paired(a, b, snrs):
    """Per-SNR mean paired difference a - b and two-sided Wilcoxon p."""
    a, b = np.asarray(a), np.asarray(b)
    rows = []
    for i, s in enumerate(snrs):
        d = a[i] - b[i]
        if np.allclose(d, 0):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(wilcoxon(a[i], b[i]).pvalue)
        rows.append({"snr_db": int(s), "mean_diff": float(d.mean()), "p": p, "resolved": bool(p < ALPHA)})
    return rows


def main():
    out = {"description": __doc__.split("\n\n")[0], "test": "two-sided Wilcoxon signed-rank, 10 paired trials, "
           "uncorrected", "alpha": ALPHA, "tables": {}}
    mlp169 = None
    for key, path in FILES.items():
        res = json.load(open(path))["results"]
        snrs = list(range(0, 21, 2))
        if key == "table_5_2":
            mlp169 = res["MLP (169p)"]["ber_trials"]
        learned = [k for k in res if k not in ("AF", "DF")]
        t = {"vs_DF": {k: paired(res[k]["ber_trials"], res["DF"]["ber_trials"], snrs) for k in learned},
             "vs_MLP169": {k: paired(res[k]["ber_trials"], mlp169, snrs) for k in learned if k != "MLP (169p)"}}
        if key == "table_5_2":
            ff = ["MLP (169p)", "Hybrid", "VAE"]
            t["feedforward_pairs"] = {f"{a} - {b}": paired(res[a]["ber_trials"], res[b]["ber_trials"], snrs)
                                      for i, a in enumerate(ff) for b in ff[i + 1:]}
        t["max_abs_mean_diff_vs_DF"] = {k: max(abs(r["mean_diff"]) for r in v) for k, v in t["vs_DF"].items()}
        out["tables"][key] = t
        for k, v in t["vs_DF"].items():
            worse = [r["snr_db"] for r in v if r["resolved"] and r["mean_diff"] > 0]
            better = [r["snr_db"] for r in v if r["resolved"] and r["mean_diff"] < 0]
            print(f"{key} {k:15s} DF ahead at {worse}; relay ahead at {better}; "
                  f"max |diff| {t['max_abs_mean_diff_vs_DF'][k]:.4f}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
