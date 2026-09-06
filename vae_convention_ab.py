#!/usr/bin/env python3
"""Controlled A/B: did the 3 dB SNR-convention bug cause the VAE relay's failure?

Background. Up to commit 97ca397, awgn_channel's real-valued branch used
sigma = sqrt(noise_power) instead of sqrt(noise_power/2) -- 3 dB more noise
than the label claimed. The published thesis reported the VAE relay "pinned at
0.25-0.40 at every SNR, meaning it never learns the task at any operating
point". After the fix the failure does not reproduce in any of three
configurations (QPSK/Rayleigh, normalized-3K, 16-class). That is suggestive,
not conclusive: those runs changed the convention and re-trained at the same
time, so neither is a controlled comparison.

Design. Evaluation is on the Rayleigh channel, which lives in fading.py and was
never touched by the fix, so the evaluation path is bit-identical in both arms.
The only thing that differs is the convention used to synthesise the relay's
*training* data, which does flow through awgn_channel
(relaynet/utils/activations.py:214). The experiment therefore isolates the
training-data hypothesis and nothing else. Same seed, same architecture, same
budget, same evaluation, in both arms.

Interpretation. If the old arm reproduces the 0.25-0.40 plateau and the new arm
does not, the convention bug caused the failure. If both arms behave, something
else fixed it and the thesis's explanation must not name the bug.
"""
import argparse
import json
import os

import numpy as np

import relaynet.channels.awgn as awgn_mod
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.relays.vae import VAERelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import compute_confidence_interval

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "vae_convention_ab.json")

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS, N_BITS = 10, 10_000
SEED = 42


def awgn_old(signal, snr_db):
    """The pre-97ca397 channel: full N0 in the single real dimension (3 dB hot)."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    if np.iscomplexobj(signal):
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(len(signal))
                             + 1j * np.random.randn(len(signal)))
    else:
        noise_std = np.sqrt(noise_power)          # <-- the bug
        noise = noise_std * np.random.randn(len(signal))
    return signal + noise


awgn_new = awgn_mod.awgn_channel


def run_arm(name, train_channel, modulation):
    """Train a VAE with *train_channel* supplying its data, evaluate on Rayleigh."""
    # activations.py imports awgn_channel into its own namespace at call time,
    # so patching the module attribute is what actually redirects training data.
    original = awgn_mod.awgn_channel
    awgn_mod.awgn_channel = train_channel
    try:
        np.random.seed(SEED)
        relay = VAERelay(window_size=7, latent_size=8, beta=0.1, prefer_gpu=False)
        relay.train(training_snrs=[5, 10, 15], num_samples=50_000,
                    epochs=100, seed=SEED)
        _, ber, trials = run_monte_carlo(
            relay, SNRS, num_bits_per_trial=N_BITS, num_trials=N_TRIALS,
            channel_fn=rayleigh_fading_channel, modulation=modulation)
    finally:
        awgn_mod.awgn_channel = original
    lo, hi = compute_confidence_interval(trials)
    print(f"  {name:24} " + "  ".join(f"{b:.4f}" for b in ber))
    return {"ber_mean": list(ber), "ci_lower": list(lo), "ci_upper": list(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulation", default="qpsk", choices=["bpsk", "qpsk"])
    args = ap.parse_args()

    print(f"VAE convention A/B  ({args.modulation.upper()} on Rayleigh, "
          f"{N_TRIALS}x{N_BITS} bits/point, seed {SEED})")
    print("  SNR (dB)                 " + "  ".join(f"{s:>6}" for s in SNRS))
    res = {
        "old_convention": run_arm("trained @ old (3 dB hot)", awgn_old, args.modulation),
        "new_convention": run_arm("trained @ corrected", awgn_new, args.modulation),
    }

    old = np.array(res["old_convention"]["ber_mean"])
    new = np.array(res["new_convention"]["ber_mean"])
    plateau = old.min() > 0.15 and old.max() < 0.45 and (old.max() - old.min()) < 0.2
    print()
    print(f"  old-arm range {old.min():.4f}-{old.max():.4f}; "
          f"new-arm range {new.min():.4f}-{new.max():.4f}")
    print(f"  old arm reproduces the published 0.25-0.40 plateau: {plateau}")
    print(f"  verdict: {'convention bug explains the failure' if plateau else 'NOT explained by the convention bug'}")

    res["meta"] = {"snrs": SNRS, "n_trials": N_TRIALS, "n_bits": N_BITS,
                   "seed": SEED, "modulation": args.modulation,
                   "eval_channel": "rayleigh (untouched by the fix)",
                   "old_arm_reproduces_plateau": bool(plateau)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
