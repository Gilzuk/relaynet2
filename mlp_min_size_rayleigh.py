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

SEEDING (three separate RNGs, all of which had to be pinned):

  1. Payload bits.   run_monte_carlo seeds Source(seed=seed_offset+trial).
  2. Channel draws.  rayleigh_fading_channel() draws from the *global* numpy
     RNG, so seeding a local default_rng is not enough (techContext gotcha).
     np.random.seed() is re-applied before every relay's evaluation, so DF, AF
     and every MLP see bit-identical h1, h2, n1, n2 sequences.
  3. Network init and batch shuffling.  MinimalGenAIRelay runs the *PyTorch*
     backend on CPU (_use_torch is forced True whenever torch imports), and
     neither the constructor's weight init nor train()'s torch.randperm()
     touches numpy. train(seed=...) seeds numpy only, which for this path
     controls nothing. Torch must be seeded explicitly, before construction
     (init) and before train() (shuffling).

Because (3) was uncontrolled, an earlier version of this sweep produced
verdicts that moved between runs -- a 41-parameter network could fail while an
8-parameter one passed. Initialization is therefore treated as a variable, not
a nuisance: every configuration is trained under TRAIN_SEEDS independent
initializations and a size only counts as matching if it matches under all of
them.

Matching is reported under two criteria, side by side:

  tolerance : mean BER penalty vs DF <= TOL_REL at every SNR, for every seed.
  Wilcoxon  : paired signed-rank test on per-trial BER vs DF, per SNR. A
              configuration "loses" at an SNR if the test is significant
              (p < 0.05) *and* the MLP is the worse of the pair.

Neither is sufficient alone. The tolerance test ignores per-trial pairing and
at 20 dB the Monte Carlo relative standard error (~1.6% here) sits just under
TOL_REL, so it is barely able to resolve a real 2% gap. The Wilcoxon test is
paired and sensitive but, with 20 trials on identical channel draws, will flag
a consistent gap far smaller than anything that matters. Reported together
they bracket the answer.

SNR convention follows memory-bank/techContext.md: gamma = 10^(SNR_dB/10).
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

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
SEED = 0                      # bits + channel; identical for every relay
TRAIN_SEEDS = [0, 1, 2]       # independent torch initializations per config

# (window, hidden). Includes the canonical 169-parameter point as the control.
GRID = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 8), (1, 16),
    (3, 1), (3, 2), (3, 4), (3, 8),
    (5, 1), (5, 2), (5, 4), (5, 8), (5, 16),
    (5, 24),                                    # = MLP-169, the control
]

# Mean-BER penalty against DF, relative, that still counts as "matching".
TOL_REL = 0.02
ALPHA = 0.05


def n_params(window, hidden):
    return hidden * (window + 2) + 1


def evaluate(relay, tag):
    """Run the Monte Carlo from a fixed global RNG state, so every relay in
    this script sees the same fading and noise realizations."""
    np.random.seed(SEED % (2 ** 31))          # global RNG: the fading draws
    _, ber, trials = run_monte_carlo(
        relay, SNRS,
        num_bits_per_trial=BITS_PER_TRIAL,
        num_trials=N_TRIALS,
        channel_fn=rayleigh_fading_channel,
        modulation="qpsk",
        seed_offset=SEED,
    )
    print(f"    {tag:<26} " + "  ".join(f"{b:.4f}" for b in ber), flush=True)
    return np.asarray(ber), np.asarray(trials)


def train_relay(window, hidden, train_seed):
    """Build and train one relay with every RNG it touches pinned."""
    torch.manual_seed(train_seed)             # weight init happens in __init__
    relay = MinimalGenAIRelay(window_size=window, hidden_size=hidden,
                              prefer_gpu=False)
    torch.manual_seed(train_seed)             # batch shuffling inside train()
    np.random.seed(train_seed)
    relay.train(training_snrs=TRAIN_SNRS, num_samples=TRAIN_SAMPLES,
                epochs=EPOCHS, seed=train_seed)
    return relay


def compare(ber, trials, df_ber, df_trials):
    """Per-SNR paired comparison of one trained relay against DF."""
    per_snr = []
    for i, snr in enumerate(SNRS):
        d = trials[i] - df_trials[i]
        if np.allclose(d, 0):
            pval = 1.0
        else:
            _stat, pval = wilcoxon(trials[i], df_trials[i])
        rel = float((ber[i] - df_ber[i]) / df_ber[i])
        per_snr.append({
            "snr_db": snr,
            "mlp_ber": float(ber[i]),
            "df_ber": float(df_ber[i]),
            "rel_penalty": rel,
            "wilcoxon_p": float(pval),
            # a significant loss: MLP reliably the worse of the pair
            "wilcoxon_loses": bool(pval < ALPHA and rel > 0),
            "wins": int(np.sum(d < 0)),
            "losses": int(np.sum(d > 0)),
        })
    return per_snr


def main():
    print(f"SNRs {SNRS} | {N_TRIALS} trials x {BITS_PER_TRIAL} bits | QPSK/Rayleigh")
    print(f"torch {torch.__version__} | init seeds {TRAIN_SEEDS} per config\n")
    print("  " + " " * 26 + "  ".join(f"{s:>6}dB" for s in SNRS))

    print("\n  classical baselines")
    df_ber, df_trials = evaluate(DecodeAndForwardRelay(), "DF (0 params)")
    af_ber, _ = evaluate(AmplifyAndForwardRelay(), "AF (0 params)")

    print("\n  MLP sweep")
    rows = []
    for window, hidden in GRID:
        p = n_params(window, hidden)
        seed_runs = []
        for ts in TRAIN_SEEDS:
            relay = train_relay(window, hidden, ts)
            ber, trials = evaluate(relay, f"w={window} h={hidden} ({p}p) seed {ts}")
            per_snr = compare(ber, trials, df_ber, df_trials)
            seed_runs.append({
                "train_seed": ts,
                "ber": [float(b) for b in ber],
                "worst_rel_penalty": max(r["rel_penalty"] for r in per_snr),
                "tolerance_ok": max(r["rel_penalty"] for r in per_snr) <= TOL_REL,
                "wilcoxon_ok": not any(r["wilcoxon_loses"] for r in per_snr),
                "per_snr": per_snr,
            })

        worst = max(s["worst_rel_penalty"] for s in seed_runs)
        best = min(s["worst_rel_penalty"] for s in seed_runs)
        tol_all = all(s["tolerance_ok"] for s in seed_runs)
        tol_any = any(s["tolerance_ok"] for s in seed_runs)
        wil_all = all(s["wilcoxon_ok"] for s in seed_runs)

        rows.append({
            "window": window, "hidden": hidden, "params": p,
            "worst_rel_penalty_over_seeds": float(worst),
            "best_rel_penalty_over_seeds": float(best),
            "matches_tolerance_all_seeds": bool(tol_all),
            "matches_tolerance_any_seed": bool(tol_any),
            "matches_wilcoxon_all_seeds": bool(wil_all),
            "seed_runs": seed_runs,
        })
        verdict = ("MATCHES (both)" if tol_all and wil_all else
                   "matches tolerance only" if tol_all else
                   "seed-dependent" if tol_any else
                   "does not match")
        print(f"      -> penalty vs DF over seeds {100*best:+.1f}% .. {100*worst:+.1f}%"
              f"   tol {'ok' if tol_all else 'no'}"
              f"   wilcoxon {'ok' if wil_all else 'no'}"
              f"   {verdict}", flush=True)

    tol_match = [r for r in rows if r["matches_tolerance_all_seeds"]]
    both_match = [r for r in rows if r["matches_tolerance_all_seeds"]
                  and r["matches_wilcoxon_all_seeds"]]
    out = {
        "snr_db": SNRS,
        "n_trials": N_TRIALS,
        "bits_per_trial": BITS_PER_TRIAL,
        "tolerance_rel": TOL_REL,
        "alpha": ALPHA,
        "train_seeds": TRAIN_SEEDS,
        "torch_version": torch.__version__,
        "df_ber": [float(b) for b in df_ber],
        "af_ber": [float(b) for b in af_ber],
        "sweep": rows,
        "min_params_tolerance": min((r["params"] for r in tol_match), default=None),
        "min_params_both_criteria": min((r["params"] for r in both_match), default=None),
    }
    path = "results/mlp_min_size_rayleigh.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 70)
    if tol_match:
        b = min(tol_match, key=lambda r: r["params"])
        print(f"  smallest MLP within {100*TOL_REL:.0f}% of DF at every SNR, all seeds:"
              f" {b['params']} params (window {b['window']}, hidden {b['hidden']})"
              f" -- {169 / b['params']:.1f}x smaller than MLP-169")
    else:
        print(f"  no configuration stayed within {100*TOL_REL:.0f}% of DF at all seeds")
    if both_match:
        b = min(both_match, key=lambda r: r["params"])
        print(f"  smallest also never losing the paired Wilcoxon test:"
              f" {b['params']} params (window {b['window']}, hidden {b['hidden']})")
    else:
        print("  no configuration passed the paired Wilcoxon test at every SNR")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
