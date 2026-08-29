"""Minimum MLP size that matches symbol-wise DF on the canonical Rayleigh channel.

Question: Table 5.2 shows the 169-parameter MLP (window 5, hidden 24) matching
symbol-wise DF on the canonical setup. How much smaller can the relay be before
it stops matching?

Why a very small network should suffice, and what that predicts. The canonical
channel is memoryless, so after coherent compensation y_R[i] is already a
sufficient statistic for x[i] and a window adds nothing (Remark on window
causality, Chapter 4). The MMSE-optimal relay estimate is the scalar map
tanh(y/sigma^2) applied per axis. Approximating one smooth monotone scalar
function needs very little capacity, so the floor should be set by whether the
network can represent a soft threshold at all, not by the window.

The sweep therefore varies window and hidden size independently:

    params = hidden * (window + 2) + 1        (real-valued, per-axis relay)

Every configuration is trained and evaluated exactly as the canonical
comparison trains and evaluates MLP-169 -- same channel, same modulation, same
multi-SNR training recipe, same Monte Carlo budget -- so the only variable is
size. DF is measured in the same run rather than quoted, so the comparison is
paired against the same channel draws.

Matching is decided by a paired Wilcoxon signed-rank test against DF on
per-trial BER, the test used elsewhere in this thesis, plus the requirement
that the mean BER penalty stay within a stated tolerance. Reporting only "the
CIs overlap" would let a large but noisy gap pass.

SNR convention follows memory-bank/techContext.md: gamma = 10^(SNR_dB/10).

GOTCHA (see techContext): rayleigh_fading_channel() draws from the global
numpy RNG, so seeding a local default_rng is not enough -- the global seed must
be set too or the fading draws are not reproducible.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relaynet.channels import rayleigh_fading_channel
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.simulation.runner import run_monte_carlo

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 20
BITS_PER_TRIAL = 20000
TRAIN_SNRS = [5, 10, 15]      # the canonical MLP-169 recipe
TRAIN_SAMPLES = 25000
EPOCHS = 100
SEED = 0

# (window, hidden). Includes the canonical 169-parameter point as the control.
GRID = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 8), (1, 16),
    (3, 1), (3, 2), (3, 4), (3, 8),
    (5, 1), (5, 2), (5, 4), (5, 8), (5, 16),
    (5, 24),                                    # = MLP-169, the control
]

# Mean-BER penalty against DF, relative, that still counts as "matching".
TOL_REL = 0.02


def n_params(window, hidden):
    return hidden * (window + 2) + 1


def evaluate(relay, tag):
    np.random.seed(SEED % (2 ** 31))          # global RNG: the fading draws
    _, ber, trials = run_monte_carlo(
        relay, SNRS,
        num_bits_per_trial=BITS_PER_TRIAL,
        num_trials=N_TRIALS,
        channel_fn=rayleigh_fading_channel,
        modulation="qpsk",
        seed_offset=SEED,
    )
    print(f"    {tag:<22} " + "  ".join(f"{b:.4f}" for b in ber), flush=True)
    return np.asarray(ber), np.asarray(trials)


def main():
    print(f"SNRs {SNRS} | {N_TRIALS} trials x {BITS_PER_TRIAL} bits | QPSK/Rayleigh\n")
    print("  " + " " * 22 + "  ".join(f"{s:>6}dB" for s in SNRS))

    print("\n  classical baselines")
    df_ber, df_trials = evaluate(DecodeAndForwardRelay(), "DF (0 params)")
    af_ber, _ = evaluate(AmplifyAndForwardRelay(), "AF (0 params)")

    print("\n  MLP sweep")
    rows = []
    for window, hidden in GRID:
        p = n_params(window, hidden)
        relay = MinimalGenAIRelay(window_size=window, hidden_size=hidden,
                                  prefer_gpu=False)
        np.random.seed(SEED % (2 ** 31))
        relay.train(training_snrs=TRAIN_SNRS, num_samples=TRAIN_SAMPLES,
                    epochs=EPOCHS, seed=SEED)
        ber, trials = evaluate(relay, f"w={window} h={hidden} ({p}p)")

        # paired per-SNR comparison against DF on identical channel draws
        per_snr = []
        for i, snr in enumerate(SNRS):
            d = trials[i] - df_trials[i]
            if np.allclose(d, 0):
                stat, pval = np.nan, 1.0
            else:
                stat, pval = wilcoxon(trials[i], df_trials[i])
            per_snr.append({
                "snr_db": snr,
                "mlp_ber": float(ber[i]),
                "df_ber": float(df_ber[i]),
                "rel_penalty": float((ber[i] - df_ber[i]) / df_ber[i]),
                "wilcoxon_p": float(pval),
                "wins": int(np.sum(d < 0)),
                "losses": int(np.sum(d > 0)),
            })

        worst = max(r["rel_penalty"] for r in per_snr)
        # "matches DF" = never worse than TOL_REL relative at any SNR
        matches = worst <= TOL_REL
        rows.append({
            "window": window, "hidden": hidden, "params": p,
            "ber": [float(b) for b in ber],
            "worst_rel_penalty": float(worst),
            "matches_df": bool(matches),
            "per_snr": per_snr,
        })
        print(f"      -> worst penalty vs DF {100*worst:+.1f}%   "
              f"{'MATCHES' if matches else 'does not match'}")

    matching = [r for r in rows if r["matches_df"]]
    out = {
        "snr_db": SNRS,
        "n_trials": N_TRIALS,
        "bits_per_trial": BITS_PER_TRIAL,
        "tolerance_rel": TOL_REL,
        "df_ber": [float(b) for b in df_ber],
        "af_ber": [float(b) for b in af_ber],
        "sweep": rows,
        "min_matching_params": min((r["params"] for r in matching), default=None),
    }
    path = "results/mlp_min_size_rayleigh.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 62)
    if matching:
        best = min(matching, key=lambda r: r["params"])
        print(f"  smallest MLP matching DF: {best['params']} parameters "
              f"(window {best['window']}, hidden {best['hidden']})")
        print(f"  against the canonical 169-parameter relay: "
              f"{169 / best['params']:.1f}x smaller")
    else:
        print("  no configuration in the grid matched DF within tolerance")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
