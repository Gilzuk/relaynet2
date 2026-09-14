"""Bias of the stopping rule used for the 16--20 dB cells of tbl:tableE6.

Chapter 7 estimates rare-event BER by transmitting until the first error at
$N_1$ bits, continuing to $10N_1$ bits, and dividing accumulated errors by
total exposure. The exposure is therefore a function of the data, which makes
the estimator biased -- an examiner review raised this against the headline
4.79e-8. This script measures the bias and, more importantly, its direction.

The model is the conservative idealisation: bit errors i.i.d. Bernoulli(p),
so the waiting time to the first error is Geometric(p) and the errors in the
remaining 9*N1 bits are Poisson(9*N1*p). Real ISI errors are bursty, which
inflates the variance of both terms; the direction of the bias is set by
Jensen's inequality on E[1/N1] and does not depend on that.

Writes results/rare_event_estimator_bias.json (into the repository, never
/tmp). Run: python rare_event_estimator_bias.py
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "rare_event_estimator_bias.json")

# The operating points the rule is actually used at in tbl:tableE6, plus two
# decades either side so the trend in the bias is visible rather than asserted.
TRUE_P = [5e-6, 5e-7, 5e-8, 5e-9]
REPLICATIONS = 200_000
SEED = 20260914


def simulate(p, reps, rng):
    """Return the estimator's realisations under the thesis's stopping rule."""
    n1 = rng.geometric(p, size=reps)                  # bits to the first error
    extra = rng.poisson(9.0 * n1 * p)                 # errors in the next 9*N1
    return (1 + extra) / (10.0 * n1)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for p in TRUE_P:
        est = simulate(p, REPLICATIONS, rng)
        rows.append({
            "true_ber": p,
            "mean_estimate": float(est.mean()),
            "mean_ratio": float(est.mean() / p),
            "median_estimate": float(np.median(est)),
            "median_ratio": float(np.median(est) / p),
            "p_estimate_below_truth": float((est < p).mean()),
            "q05_ratio": float(np.quantile(est, 0.05) / p),
            "q95_ratio": float(np.quantile(est, 0.95) / p),
        })

    # Inverting the ratio quantiles turns the sampling spread into a
    # calibrated 90% interval on the true BER given one observed estimate,
    # which is what the thesis actually needs to quote alongside 4.79e-8.
    r = rows[TRUE_P.index(5e-8)]
    observed = 4.79e-8
    interval = [observed / r["q95_ratio"], observed / r["q05_ratio"]]

    payload = {
        "description": "Bias of the first-error stopping rule of Section 7 "
                       "(transmit to first error at N1, continue to 10*N1, "
                       "divide accumulated errors by total exposure).",
        "model": "i.i.d. Bernoulli(p); N1 ~ Geometric(p); "
                 "errors in remaining 9*N1 bits ~ Poisson(9*N1*p)",
        "replications": REPLICATIONS,
        "seed": SEED,
        "rows": rows,
        "tbl_tableE6_16dB": {
            "reported_ber": observed,
            "implied_90pct_interval": interval,
            "note": "The reported value is 16 errors over 334,002,040 bits. "
                    "The interval is the estimator's own sampling spread "
                    "inverted, not a Poisson interval at fixed exposure.",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"{'true BER':>10} {'mean/p':>9} {'median/p':>10} {'P(est<p)':>9}")
    for r in rows:
        print(f"{r['true_ber']:>10.1e} {r['mean_ratio']:>9.2f} "
              f"{r['median_ratio']:>10.2f} {r['p_estimate_below_truth']:>9.3f}")
    print(f"\n16 dB cell: {observed:.3g} reported, "
          f"90% interval [{interval[0]:.2g}, {interval[1]:.2g}]")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
