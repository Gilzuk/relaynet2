"""Across-seed spread of the *canonical* cross-architecture comparison.

Limitations item 8 concedes that the cross-architecture comparison behind
Table~tbl:table2 is "still a single trained instance per architecture with
unequal training budgets", so the claim that architecture is secondary to
capacity "remains an observation from one run rather than a seed-robust
result". `seed_spread_architectures.py` closed that gap for the equal-budget
3K family; this closes it for the architectures the canonical table actually
reports.

BUDGET. Every architecture gets an identical 100 epochs at 25,000 samples,
which is the point: the published run gives the MLP and Hybrid 25,000 samples
and the VAE and sequence models 50,000, so the native comparison confounds
architecture with training budget. Equalising removes that confound. The
absolute BER may therefore sit slightly above the published figures, which does
not matter here because the quantity of interest is the spread across seeds at
one common budget, not the level.

SEEDING. torch.manual_seed() before construction fixes the weights and again
before train() fixes batch shuffling, because the sequence checkpoints seed
nothing internally. Every architecture is evaluated on identical channel draws
and payload bits, with the global numpy RNG re-seeded before each evaluation,
so what is reported is training variance and not evaluation variance.

Written incrementally after every (architecture, seed) pair: this is a
multi-hour CPU run and a partial result is worth more than a lost one.

Output: results/seed_spread_native_architectures.json
"""

import contextlib
import io as _io
import json
import os
import time

import numpy as np
import torch

from ber_metrics import penalty_table
from relaynet.channels import rayleigh_fading_channel
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from run_experiments import build_base_relays

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS, BITS = 20, 20_000
EVAL_SEED = 0
TRAIN_SEEDS = [0, 1, 2]
TRAIN_SNRS = [5, 10, 15]
SAMPLES, EPOCHS = 25_000, 100          # identical for every architecture
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "seed_spread_native_architectures.json")

# name -> (training_snrs, extra train kwargs). Hybrid trains at low SNR by
# design; the sequence models take an explicit lr and modulation.
ARCHS = {
    "MLP (169p)":   (TRAIN_SNRS, {}),
    "Hybrid":       ([2, 4, 6], {}),
    "VAE":          (TRAIN_SNRS, {}),
    "Transformer":  (TRAIN_SNRS, {"lr": 0.001, "training_modulation": "qpsk"}),
    "Mamba S6":     (TRAIN_SNRS, {"lr": 0.001, "training_modulation": "qpsk"}),
    "Mamba2 (SSD)": (TRAIN_SNRS, {"lr": 0.001, "training_modulation": "qpsk"}),
}


def evaluate(relay):
    np.random.seed(EVAL_SEED % (2 ** 31))
    _, ber, _ = run_monte_carlo(
        relay, SNRS, num_bits_per_trial=BITS, num_trials=N_TRIALS,
        channel_fn=rayleigh_fading_channel, modulation="qpsk",
        seed_offset=EVAL_SEED)
    return np.asarray(ber)


def n_params(relay):
    m = getattr(relay, "model", None)
    try:
        return sum(q.numel() for q in m.parameters()) if m is not None else -1
    except Exception:
        return -1


def main():
    print(f"Across-seed spread, canonical architectures | {N_TRIALS} trials x "
          f"{BITS} bits | seeds {TRAIN_SEEDS} | {EPOCHS} epochs x {SAMPLES} "
          f"samples (equalised)\n", flush=True)

    df = evaluate(DecodeAndForwardRelay())
    af = evaluate(AmplifyAndForwardRelay())
    print("  DF  " + "  ".join(f"{b:.5f}" for b in df), flush=True)
    print("  AF  " + "  ".join(f"{b:.5f}" for b in af) + "\n", flush=True)

    out = {"snr_db": SNRS, "n_trials": N_TRIALS, "bits_per_trial": BITS,
           "train_seeds": TRAIN_SEEDS, "epochs": EPOCHS, "samples": SAMPLES,
           "budget_note": "equalised across architectures; published run uses "
                          "25k for MLP/Hybrid and 50k for VAE/sequence models",
           "df_ber": [float(x) for x in df], "af_ber": [float(x) for x in af],
           "architectures": {}}

    def flush():
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")

    flush()
    for name, (snrs, kw) in ARCHS.items():
        out["architectures"][name] = {"params": -1, "runs": [],
                                      "spread_db": float("nan")}
        for ts in TRAIN_SEEDS:
            t0 = time.time()
            torch.manual_seed(ts)
            relay = build_base_relays(gpu=False)[name]
            torch.manual_seed(ts)
            np.random.seed(ts)
            with contextlib.redirect_stdout(_io.StringIO()):
                relay.train(training_snrs=snrs, num_samples=SAMPLES,
                            epochs=EPOCHS, **kw)
            ber = evaluate(relay)
            p = penalty_table(SNRS, ber, df)
            pen = (p["worst_db_penalty"] if p["targets_reached"]
                   else float("nan"))
            rec = out["architectures"][name]
            rec["runs"].append({"train_seed": ts,
                                "ber": [float(x) for x in ber],
                                "db_penalty": pen})
            rec["params"] = n_params(relay)
            vals = [r["db_penalty"] for r in rec["runs"]
                    if r["db_penalty"] == r["db_penalty"]]
            rec["spread_db"] = float(max(vals) - min(vals)) if len(vals) > 1 \
                else float("nan")
            flush()
            print(f"  {name:<14} seed {ts}  penalty {pen:+7.3f} dB   "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  {name:<14} spread {rec['spread_db']:+.3f} dB "
              f"({rec['params']} params)\n", flush=True)

    print("=" * 74)
    sp = {k: v["spread_db"] for k, v in out["architectures"].items()
          if v["spread_db"] == v["spread_db"]}
    if sp:
        worst = max(sp, key=sp.get)
        print(f"  largest across-seed spread: {worst} at {sp[worst]:.3f} dB")
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
