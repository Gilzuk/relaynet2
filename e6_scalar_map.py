"""Scalar MAP relay on the unknown-ISI channel: decision rule vs temporal context.

Dr. Khina's review (comment 9) notes that the large gain of the windowed MLP
over zero-threshold DF on the selected ISI channel does not by itself
establish a new relay principle: the thesis itself observes that a
non-monotonic scalar detector could remove the channel's noiseless sign-error
floor. This script measures that reference.

ScalarMAPISIRelay decides each symbol from y[n] alone, comparing the two
four-component Gaussian mixtures; it is told the taps and the noise variance,
the same information genie-CSI Viterbi receives. It is run beside
zero-threshold DF, the 170-parameter MLP over its three training seeds and
genie Viterbi MLSE under the matched protocol of e6_matched_protocol.py:
11 SNR points, 10 trials x 100,000 bits, the same channel objects, the same
per-trial source seeds, the current hop-2 noise convention.

For each relay the hop-1 (relay-output) error rate is recorded as well as the
destination BER. As an independent check, the scalar MAP relay's hop-1 error
rate is also computed by numerical integration of the Bayes error,
0.5 * integral of min(p(y | +1), p(y | -1)) dy.

Writes results/e6_scalar_map.json. Run: python e6_scalar_map.py
"""

import json
import os
import subprocess
import time

import numpy as np

import e6_sim_ported as e6
from relaynet.relays import DecodeAndForwardRelay, ScalarMAPISIRelay, ViterbiMLSERelay

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "e6_scalar_map.json")

H_CHANNEL = np.array([1.0, 0.7, 0.5])
H_ISI = H_CHANNEL / np.linalg.norm(H_CHANNEL)   # ISIChannel normalizes internally
SNRS = list(range(0, 21, 2))
SEEDS = (0, 1, 2)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def bayes_error(snr_db, n_grid=400_001):
    """Scalar-MAP hop-1 error by numerical integration (equal priors)."""
    r = ScalarMAPISIRelay(H_CHANNEL, snr_db)
    s = np.sqrt(r.sigma2)
    y = np.linspace(-3 - 8 * s, 3 + 8 * s, n_grid)
    g = lambda m: np.exp(-(y[:, None] - m[None, :]) ** 2 / (2 * r.sigma2)).mean(axis=1) \
        / np.sqrt(2 * np.pi * r.sigma2)
    return float(0.5 * np.trapezoid(np.minimum(g(r.means_plus), g(r.means_minus)), y))


def run(n_bits, n_trials):
    hop1 = e6.create_channel("isi", seed=1)
    hop2 = e6.create_channel("awgn", seed=2)
    dst = e6.Destination(modulation="bpsk")
    mlps = [e6.train_mlp(hop1, seed=s)[0] for s in SEEDS]

    names = ("DF", "scalar-MAP", "MLP", "VIT-genie")
    res = {k: {"errors": [], "bits": [], "relay_errors": [], "relay_bits": []} for k in names}
    t0 = time.time()
    for snr in SNRS:
        acc = {k: [0, 0, 0, 0] for k in names}
        smap = ScalarMAPISIRelay(H_CHANNEL, snr)
        for tr in range(n_trials):
            src = e6.Source(seed=10_000 + 97 * snr + tr, modulation="bpsk")
            tx_bits, tx_sym = src.transmit(n_bits)
            y = hop1(tx_sym, snr)
            relays = [("DF", DecodeAndForwardRelay(target_power=1.0)),
                      ("scalar-MAP", smap),
                      ("VIT-genie", ViterbiMLSERelay(channel_taps=H_ISI))]
            relays += [("MLP", m) for m in mlps]
            for name, relay in relays:
                out = relay.process(y)
                acc[name][2] += int(np.count_nonzero(np.sign(out) != tx_sym))
                acc[name][3] += tx_bits.size
                rx = dst.receive(hop2(out, snr))
                acc[name][0] += int(np.count_nonzero(tx_bits != rx))
                acc[name][1] += tx_bits.size
        for k in names:
            for j, f in enumerate(("errors", "bits", "relay_errors", "relay_bits")):
                res[k][f].append(acc[k][j])
        print(f"  {snr:2d} dB  " + "  ".join(
            f"{k}={acc[k][0] / acc[k][1]:.3e}" for k in names))

    analytic = {str(s): bayes_error(s) for s in SNRS}
    payload = {
        "description": "Scalar MAP relay vs zero-threshold DF, MLP and genie Viterbi "
                       "on the unknown-ISI (BPSK, AWGN hop 2) channel, matched protocol.",
        "taps_effective": H_ISI.tolist(),
        "noiseless_means_given_plus1": sorted(ScalarMAPISIRelay(H_CHANNEL, 0).means_plus.tolist()),
        "hop2": "awgn, sigma^2 = N0/2 per real dimension (current convention)",
        "snrs": SNRS, "bits_per_trial": n_bits, "trials": n_trials,
        "mlp_training_seeds": list(SEEDS),
        "scalar_map_relay_error_analytic": analytic,
        "git_rev": git_rev(), "seconds": round(time.time() - t0, 1),
        "results": {k: dict(v, ber=[e / b for e, b in zip(v["errors"], v["bits"])],
                            relay_ser=[e / b for e, b in zip(v["relay_errors"], v["relay_bits"])])
                    for k, v in res.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2); fh.write("\n")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    run(a.bits, a.trials)
