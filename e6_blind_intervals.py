"""Correct intervals for the blind-regime comparison (thesis Sec. 6.4, Table 6.1 layer 4).

e6_blind_ported.py stores, for every relay, a mean and an interval
1.96 * std / sqrt(150) over 150 columns: 3 trained MLP instances x 50 trials.
Two things are wrong with that interval as a confidence statement:

* the classical relays (CMA, decision-directed Viterbi, differential DF) have
  no training seed, and their trial seed (8000 * si + tr) does not depend on
  the MLP instance, so their 150 columns are three identical copies of the same
  50 channel draws. Dividing by sqrt(150) overstates their precision by sqrt(3);
* the MLP's 150 columns pool three trained networks, so the interval resamples
  Monte Carlo noise but not training variance (ber_metrics.hierarchical_ci
  explains why that understates it).

The thesis text describes the intended rule (classical intervals over the 50
draws, the MLP treated hierarchically) but quotes the stored pooled values.
This script reruns the identical protocol -- same channel, MLP training seeds
(2, 3, 4) and trial seeds -- keeps every column, checks that the means reproduce
the stored ones, and reports:

* classical intervals on the 50 unique draws (Student t, 49 d.o.f.);
* the MLP interval computed hierarchically over the three trained instances;
* the paired MLP - CMA difference: per instance, the mean over the 50 shared
  draws, then a Student-t interval over the three instances (training variance
  included), and the draw-level paired interval for reference.

Output: results/e6_blind_intervals.json
Usage:  python e6_blind_intervals.py
"""
import json
import os
import subprocess

import numpy as np
from scipy import stats

import e6_blind_ported as B
from ber_metrics import hierarchical_ci
from relaynet.channels import RandomISICompositeChannel, ComplexISIRayleighChannel

OUT = "results/e6_blind_intervals.json"
STORED = "e6_unknown_channel_results/e6_blind_ported_results.npy"
CLASSICAL = ("DF-diff", "CMA-blind", "Viterbi-blind")


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def t_ci(x, conf=0.95):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.5 + conf / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def main():
    channel = RandomISICompositeChannel(pa_sat=1.2, seed=1)
    hop2 = ComplexISIRayleighChannel(taps=np.array([1.0]), seed=2)
    mlps = [B.train_mlp(hidden_size=7, seed=2 + ti) for ti in range(B.N_TRAIN)]
    snrs = [int(s) for s in B.SNRS]
    nt = B.N_TRIALS
    cls = {k: np.zeros((len(snrs), nt)) for k in CLASSICAL}
    mlp = np.zeros((len(snrs), B.N_TRAIN, nt))
    for si, snr in enumerate(snrs):
        for tr in range(nt):
            seed = 8000 * si + tr
            for k in CLASSICAL:
                cls[k][si, tr] = B.run_ber_trial(k, None, channel, hop2, B.N_BITS, snr, seed=seed)
            for ti, m in enumerate(mlps):
                mlp[si, ti, tr] = B.run_ber_trial("MLP-169", m, channel, hop2, B.N_BITS, snr, seed=seed)
        print(f"{snr:2d} dB  MLP {mlp[si].mean():.5f}  CMA {cls['CMA-blind'][si].mean():.5f}", flush=True)

    stored = np.load(STORED, allow_pickle=True).item()["summary"]
    out = {"description": __doc__.split("\n\n")[0], "git_rev": git_rev(), "snr_db": snrs,
           "n_trials": nt, "n_train": B.N_TRAIN, "n_bits": B.N_BITS, "rows": {}}
    for si, snr in enumerate(snrs):
        row = {}
        for k in CLASSICAL:
            v = cls[k][si]
            row[k] = {"mean": float(v.mean()), "ci_unique_draws": t_ci(v),
                      "ci_stored_pooled": float(stored[k][1][si]),
                      "mean_matches_stored": bool(abs(v.mean() - stored[k][0][si]) < 1e-12)}
        v = mlp[si]
        h = hierarchical_ci(v.reshape(-1), B.N_TRAIN, nt)
        row["MLP-169"] = {"mean": float(v.mean()), "ci_hierarchical": float(h["ci"]),
                          "seed_means": [float(x) for x in v.mean(axis=1)],
                          "ci_stored_pooled": float(stored["MLP-169"][1][si]),
                          "mean_matches_stored": bool(abs(v.mean() - stored["MLP-169"][0][si]) < 1e-12)}
        d_inst = (v - cls["CMA-blind"][si][None, :]).mean(axis=1)      # per trained instance
        d_draw = v.mean(axis=0) - cls["CMA-blind"][si]                 # per shared draw
        row["MLP_minus_CMA"] = {"mean": float(d_inst.mean()),
                                "per_instance": [float(x) for x in d_inst],
                                "ci_over_instances": t_ci(d_inst),
                                "ci_over_draws": t_ci(d_draw),
                                "draws_mlp_better": int((d_draw < 0).sum()),
                                "wilcoxon_p_draws": float(stats.wilcoxon(d_draw).pvalue)}
        out["rows"][str(snr)] = row
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
