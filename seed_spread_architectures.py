"""Across-initialization spread for the equal-budget architectures (H4).

Section~\\ref{sec:capacity-and-overfitting} defends H4's single-seed
equal-budget comparison by pointing at the across-initialization spread
measured in the size sweep: 0.008 dB median on the canonical channel, well
below the ~0.18 dB that H4's 4.1% high-SNR architecture spread corresponds
to. That defence was explicitly labelled reassurance rather than proof,
because the spread was measured on MLPs and nothing licensed assuming the
sequence models initialize as stably. This script measures them directly.

WHY THIS IS WORTH MEASURING RATHER THAN ASSUMED. None of the three sequence
checkpoints seeds torch: grep manual_seed over checkpoint_18_transformer_relay,
checkpoint_20_mamba_s6_relay and checkpoint_23_mamba2_relay returns nothing,
the same defect found earlier in MinimalGenAIRelay. The published equal-budget
comparison was therefore run from an uncontrolled initialization, and the
variance it carries has never been quantified. Seeding here is external:
torch.manual_seed() before construction, which fixes the weights, and again
before train(), which fixes batch shuffling.

BUDGET. 100 epochs at 25,000 samples, against the thesis protocol's 100
epochs at 50,000. Epochs are kept and samples halved so the optimizer sees
the same number of passes; the absolute BER may therefore sit slightly above
the published figures, which does not matter here because the quantity of
interest is the spread between seeds at an identical budget, not the level.

Every architecture is evaluated on identical channel draws and payload bits,
with the global numpy RNG re-seeded before each run, so the spread reported
is training variance and not evaluation variance.
"""

import contextlib
import io as _io
import json
import sys
import time

import numpy as np
import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import (
    make_mlp_3k, make_vae_3k, make_transformer_3k, make_mamba_3k, make_mamba2_3k,
)
from relaynet.channels import rayleigh_fading_channel
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 20
BITS = 20000
SEED = 0
TRAIN_SEEDS = [0, 1, 2]
TRAIN_SNRS = [5, 10, 15]
SAMPLES = 25000
EPOCHS = 100

ARCHS = [
    ("MLP-3K", make_mlp_3k),
    ("VAE-3K", make_vae_3k),
    ("Transformer-3K", make_transformer_3k),
    ("Mamba-S6-3K", make_mamba_3k),
    ("Mamba2-3K", make_mamba2_3k),
]


def evaluate(relay):
    np.random.seed(SEED % (2 ** 31))
    _, ber, _ = run_monte_carlo(
        relay, SNRS, num_bits_per_trial=BITS, num_trials=N_TRIALS,
        channel_fn=rayleigh_fading_channel, modulation="qpsk", seed_offset=SEED)
    return np.asarray(ber)


def main():
    print("Across-initialization spread of the equal-budget architectures")
    print(f"canonical QPSK/Rayleigh | {N_TRIALS} trials x {BITS} bits | "
          f"{len(TRAIN_SEEDS)} seeds | {EPOCHS} epochs x {SAMPLES} samples\n")

    df = evaluate(DecodeAndForwardRelay())
    af = evaluate(AmplifyAndForwardRelay())
    print("  DF  " + "  ".join(f"{b:.5f}" for b in df))
    print("  AF  " + "  ".join(f"{b:.5f}" for b in af))

    out = {"snr_db": SNRS, "n_trials": N_TRIALS, "bits_per_trial": BITS,
           "train_seeds": TRAIN_SEEDS, "epochs": EPOCHS, "samples": SAMPLES,
           "df_ber": [float(x) for x in df], "af_ber": [float(x) for x in af],
           "architectures": {}}

    print(f"\n  {'architecture':<16} {'params':>7}  per-seed dB penalty vs DF"
          f"        spread")
    for name, factory in ARCHS:
        runs, t0 = [], time.time()
        for ts in TRAIN_SEEDS:
            torch.manual_seed(ts)                 # weight initialization
            relay = factory(prefer_gpu=False)
            torch.manual_seed(ts)                 # batch shuffling in train()
            np.random.seed(ts)
            with contextlib.redirect_stdout(_io.StringIO()):
                relay.train(training_snrs=TRAIN_SNRS, num_samples=SAMPLES,
                            epochs=EPOCHS, training_modulation="qpsk")
            ber = evaluate(relay)
            p = penalty_table(SNRS, ber, df)
            runs.append({"train_seed": ts, "ber": [float(x) for x in ber],
                         "db_penalty": (p["worst_db_penalty"]
                                        if p["targets_reached"] else float("nan"))})
        n = -1
        try:
            m = getattr(relay, "model", None)
            n = sum(q.numel() for q in m.parameters()) if m is not None else -1
        except Exception:
            pass
        vals = [r["db_penalty"] for r in runs if r["db_penalty"] == r["db_penalty"]]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        out["architectures"][name] = {"params": n, "runs": runs,
                                      "spread_db": float(spread)}
        cells = "  ".join(f"{v:+6.3f}" for v in vals) if vals else "  --"
        print(f"  {name:<16} {n:>7}  {cells:<34} {spread:+.3f} dB"
              f"   [{time.time()-t0:.0f}s]", flush=True)
        with open("results/seed_spread_architectures.json", "w") as fh:
            json.dump(out, fh, indent=2)

    print("\n" + "=" * 78)
    sp = {k: v["spread_db"] for k, v in out["architectures"].items()
          if v["spread_db"] == v["spread_db"]}
    if sp:
        worst = max(sp, key=sp.get)
        print(f"  largest across-seed spread: {worst} at {sp[worst]:.3f} dB")
        print(f"  H4's high-SNR architecture spread (4.1% BER) is about 0.18 dB")
        print(f"  -> single-seed H4 comparison is {'SAFE' if sp[worst] < 0.18 else 'NOT SAFE'}"
              f": the largest seed spread is "
              f"{'below' if sp[worst] < 0.18 else 'above'} the effect it must resolve")
    print("  saved results/seed_spread_architectures.json")


if __name__ == "__main__":
    main()
