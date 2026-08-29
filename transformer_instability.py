"""Why is Transformer-3K unstable across initializations?

The equal-budget seed sweep found Transformer-3K spread 0.906 dB across three
initializations -- five times the ~0.18 dB that H4's high-SNR architecture
spread corresponds to -- while Mamba-S6-3K spread 0.041 dB and MLP-3K
0.055 dB at the same budget. The instability is specific to the Transformer,
not generic to sequence models, and this script asks what produces it.

Two mechanical explanations are already ruled out. Dropout is not leaking
into evaluation: process() calls model.eval() before inference
(checkpoint_18_transformer_relay.py:451). And the divergence is not an
evaluation artifact, since every seed is scored on identical channel draws
and payload bits.

What the three-seed data shows is that the outlier tracks the others at 0 dB
and separates progressively as SNR rises, ending at an error floor: seed 0
reaches 0.01125 at 20 dB where seeds 1 and 2 reach 0.00996 and 0.01005. That
is the shape of a model that never found the sharp decision rather than one
that found a different valid one.

This script distinguishes the two candidate causes by recording the training
objective alongside the BER:

  optimization failure   the bad seed ends at a visibly worse training loss,
                         so the run simply did not converge
  objective mismatch     all seeds reach comparable loss but differ in BER,
                         so the loss the model minimizes is not tracking the
                         error rate the relay is judged on

Eight initializations are run rather than three, so the frequency of the bad
mode is estimated rather than inferred from a single occurrence.
"""

import contextlib
import io as _io
import json
import re
import time

import numpy as np
import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import make_transformer_3k, make_mamba_3k
from relaynet.channels import rayleigh_fading_channel
from relaynet.relays import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS, BITS, SEED = 20, 20000, 0
TRAIN_SNRS, SAMPLES, EPOCHS = [5, 10, 15], 25000, 100
SEEDS = list(range(8))

_LOSS = re.compile(r"Epoch (\d+)/\d+, Loss: ([0-9.]+).*?Val Acc: ([0-9.]+)")


def evaluate(relay):
    np.random.seed(SEED % (2 ** 31))
    _, ber, _ = run_monte_carlo(relay, SNRS, num_bits_per_trial=BITS,
                                num_trials=N_TRIALS, channel_fn=rayleigh_fading_channel,
                                modulation="qpsk", seed_offset=SEED)
    return np.asarray(ber)


def one_run(factory, ts):
    torch.manual_seed(ts)
    relay = factory(prefer_gpu=False)
    torch.manual_seed(ts)
    np.random.seed(ts)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        relay.train(training_snrs=TRAIN_SNRS, num_samples=SAMPLES,
                    epochs=EPOCHS, training_modulation="qpsk")
    rows = _LOSS.findall(buf.getvalue())
    losses = [float(r[1]) for r in rows]
    vaccs = [float(r[2]) for r in rows]
    return relay, losses, vaccs


def main():
    df = evaluate(DecodeAndForwardRelay())
    out = {"snr_db": SNRS, "seeds": SEEDS, "epochs": EPOCHS, "samples": SAMPLES,
           "df_ber": [float(x) for x in df], "runs": {}}

    print("Transformer-3K across 8 initializations, canonical QPSK/Rayleigh")
    print(f"{'seed':>5} {'final loss':>11} {'best loss':>10} {'final val':>10} "
          f"{'dB vs DF':>9} {'BER@20dB':>10}")
    for arch, factory in (("Transformer-3K", make_transformer_3k),
                          ("Mamba-S6-3K", make_mamba_3k)):
        print(f"\n  --- {arch}")
        recs = []
        for ts in SEEDS:
            t0 = time.time()
            relay, losses, vaccs = one_run(factory, ts)
            ber = evaluate(relay)
            p = penalty_table(SNRS, ber, df)
            db = p["worst_db_penalty"] if p["targets_reached"] else float("nan")
            rec = {"seed": ts, "final_loss": losses[-1] if losses else None,
                   "best_loss": min(losses) if losses else None,
                   "final_val_acc": vaccs[-1] if vaccs else None,
                   "loss_curve": losses, "ber": [float(x) for x in ber],
                   "db_penalty": float(db)}
            recs.append(rec)
            print(f"{ts:>5} {rec['final_loss']:>11.6f} {rec['best_loss']:>10.6f} "
                  f"{rec['final_val_acc']:>10.4f} {db:>+9.3f} {ber[-1]:>10.5f}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
            out["runs"].setdefault(arch, []).append(rec)
            with open("results/transformer_instability.json", "w") as fh:
                json.dump(out, fh, indent=2)

        dbs = np.array([r["db_penalty"] for r in recs])
        fl = np.array([r["final_loss"] for r in recs], dtype=float)
        print(f"  spread {dbs.max()-dbs.min():.3f} dB   "
              f"loss range {fl.min():.6f}-{fl.max():.6f}")
        if len(dbs) > 2 and np.std(fl) > 0:
            print(f"  correlation(final loss, dB penalty) = "
                  f"{np.corrcoef(fl, dbs)[0,1]:+.3f}")
    print("\n  saved results/transformer_instability.json")


if __name__ == "__main__":
    main()
