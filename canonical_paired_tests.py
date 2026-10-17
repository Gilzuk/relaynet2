"""Paired tests behind the Chapter 5 statements about which canonical relays differ.

Every relay of Tables 5.2 and 5.3 is evaluated on the same ten trials (seeds 0-9,
shared by every relay at every SNR), so each pair of relays has ten paired
per-trial BERs per SNR. Section 4.3.2 declares the two-sided Wilcoxon
signed-rank test on those pairs, uncorrected for multiplicity; with ten pairs
its smallest attainable p is 2/1024 ~ 0.002. Integer error-count differences
preserve ties that floating-point subtraction of BERs can otherwise split.
This script applies it to

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
CANONICAL_BITS_PER_TRIAL = 10000


def paired(a, b, snrs, *, bits_per_trial):
    """Test paired BERs using integer error differences at a common bit budget.

    SciPy 1.17 auto uses exhaustive sign permutations for ten-pair tests with
    ties/zeros; larger coded samples can use its asymptotic method. The
    generating SciPy version is recorded in the outputs.
    """
    if (isinstance(bits_per_trial, (bool, np.bool_))
            or not isinstance(bits_per_trial, (int, np.integer))
            or bits_per_trial <= 0):
        raise ValueError("bits_per_trial must be a positive integer")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(snrs) or not a.shape[1]:
        raise ValueError("BER arrays must have matching (SNR, trial) shapes")
    counts = []
    for values in (a, b):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("BERs must be finite and in [0, 1]")
        scaled = values * bits_per_trial
        rounded = np.rint(scaled)
        if not np.allclose(scaled, rounded, rtol=0, atol=1e-7):
            raise ValueError("BERs do not represent integer counts at bits_per_trial")
        counts.append(rounded.astype(np.int64))
    rows = []
    for i, s in enumerate(snrs):
        d = a[i] - b[i]
        error_difference = counts[0][i] - counts[1][i]
        if not np.any(error_difference):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(wilcoxon(error_difference, zero_method="wilcox",
                                   correction=False, alternative="two-sided",
                                   method="auto").pvalue)
        rows.append({"snr_db": int(s), "mean_diff": float(d.mean()), "p": p, "resolved": bool(p < ALPHA)})
    return rows


def main():
    import scipy

    out = {"description": __doc__.split("\n\n")[0], "test": "two-sided Wilcoxon signed-rank, 10 paired trials, "
           "integer error-count differences, zeros omitted, method=auto, uncorrected",
           "scipy_version": scipy.__version__, "bits_per_trial": CANONICAL_BITS_PER_TRIAL,
           "alpha": ALPHA, "tables": {}}
    def canonical_pair(a, b, snrs):
        return paired(a, b, snrs, bits_per_trial=CANONICAL_BITS_PER_TRIAL)
    mlp169 = None
    for key, path in FILES.items():
        res = json.load(open(path))["results"]
        snrs = list(range(0, 21, 2))
        if key == "table_5_2":
            mlp169 = res["MLP (169p)"]["ber_trials"]
        learned = [k for k in res if k not in ("AF", "DF")]
        t = {"vs_DF": {k: canonical_pair(res[k]["ber_trials"], res["DF"]["ber_trials"], snrs) for k in learned},
             "vs_MLP169": {k: canonical_pair(res[k]["ber_trials"], mlp169, snrs) for k in learned if k != "MLP (169p)"}}
        if key == "table_5_2":
            ff = ["MLP (169p)", "Hybrid", "VAE"]
            t["feedforward_pairs"] = {f"{a} - {b}": canonical_pair(res[a]["ber_trials"], res[b]["ber_trials"], snrs)
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
