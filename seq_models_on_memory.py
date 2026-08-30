"""Do sequence architectures earn their inductive bias on channels with memory?

The thesis reports that Transformer, Mamba-S6 and Mamba-2 buy nothing on the
canonical channel and attributes this to the channel itself: "the memoryless
channel simply offers no temporal structure for their inductive bias to
exploit" (Section~\\ref{sec:siso-bpsk-performance-baseline-relay-comparison}).
That is a prediction, and it has never been tested, because no sequence model
appears anywhere in the unknown-channel work -- not in any e6 script, and not
once in Chapter 7.

The minimum-size sweep sharpens the prediction. It finds that on channels with
memory the binding resource is the relay's window, worth 8 to 11 dB between
window 1 and window 7, while width and depth buy almost nothing. A window is
a crude way of giving a feedforward network temporal context; sequence models
are built for exactly that. If the Chapter 5 explanation is right, this is the
setting where they should finally win.

POSITIVE HYPOTHESIS UNDER TEST. On channels with memory, sequence
architectures outperform a feedforward relay of equal parameter count, having
temporal structure to exploit that the memoryless channel did not offer.

TRAINING ON THE ACTUAL CHANNEL. The wrappers' own train() generates its data
internally from AWGN or Rayleigh and cannot be handed an ISI channel, so this
script reimplements the wrapper's loop -- windows in, clean symbols out, Adam
on MSE, identical hyperparameters -- against data drawn from the channel under
test. Without that, a sequence model would be trained on a memoryless
surrogate and evaluated on a channel with memory, which would test mismatch
rather than the hypothesis.

Every architecture is seeded (torch.manual_seed before construction and before
training) and run over three initializations, following the finding that
Transformer-3K spreads 0.906 dB across seeds while every other architecture
stays within 0.16 dB.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import (
    make_mlp_3k, make_transformer_3k, make_mamba_3k, make_mamba2_3k)
from mlp_min_size_all_channels import (CHANNELS, evaluate_two_hop,
                                       baseline_diagnostics, SNRS)
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay

TRAIN_SNRS = [5, 10, 15]
SAMPLES, EPOCHS, BATCH, LR = 20000, 40, 64, 1e-3
TRAIN_SEEDS = [0, 1, 2]
CH = ["isi", "isi_complex", "composite"]

ARCHS = [("MLP-3K", make_mlp_3k), ("Transformer-3K", make_transformer_3k),
         ("Mamba-S6-3K", make_mamba_3k), ("Mamba2-3K", make_mamba2_3k)]


def make_data(channel, mod, window, rng):
    """Windows of the channel's output paired with the clean symbol, per axis,
    mirroring how the runner applies these relays at test time."""
    per = SAMPLES // len(TRAIN_SNRS)
    pad = window // 2
    X, T = [], []
    for snr in TRAIN_SNRS:
        if mod == "bpsk":
            x = 1.0 - 2.0 * rng.integers(0, 2, per).astype(float)
            y = np.asarray(channel(x, snr))
            axes = [(y.real if np.iscomplexobj(y) else y, x)]
        else:
            b = rng.integers(0, 2, (per, 2))
            xr = (1.0 - 2.0 * b[:, 0]) / np.sqrt(2.0)
            xi = (1.0 - 2.0 * b[:, 1]) / np.sqrt(2.0)
            y = np.asarray(channel(xr + 1j * xi, snr))
            axes = ([(y.real, xr), (y.imag, xi)] if np.iscomplexobj(y)
                    else [(y, xr)])
        for yy, tt in axes:
            yp = np.pad(np.asarray(yy, float), (pad, pad), mode="constant")
            X.append(np.lib.stride_tricks.sliding_window_view(yp, window))
            T.append(np.asarray(tt, float))
    return np.vstack(X), np.concatenate(T)


def torch_module(relay):
    """The relays expose their torch net under different names: the sequence
    wrappers as .model, MinimalGenAIRelay as ._torch_model."""
    m = getattr(relay, "model", None)
    if m is None:
        m = getattr(relay, "_torch_model", None)
    if m is None:
        raise AttributeError(f"{type(relay).__name__} exposes no torch module")
    return m


def train_on_channel(relay, channel, mod, seed):
    """The wrapper's own loop, driven by data from the channel under test."""
    rng = np.random.default_rng(1000 + seed)
    X, T = make_data(channel, mod, relay.window_size, rng)
    dev = getattr(relay, "device", "cpu")
    net = torch_module(relay)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Tt = torch.as_tensor(T, dtype=torch.float32, device=dev).unsqueeze(-1)
    # the sequence models take (batch, window, 1); the MLP takes (batch, window)
    with torch.no_grad():
        try:
            net(Xt[:2])
        except Exception:
            Xt = Xt.unsqueeze(-1)
            net(Xt[:2])
    opt = optim.Adam(net.parameters(), lr=LR)
    crit = nn.MSELoss()
    net.train()
    for _ in range(EPOCHS):
        idx = torch.randperm(Xt.size(0), device=dev)
        for i in range(0, Xt.size(0), BATCH):
            sl = idx[i:i + BATCH]
            opt.zero_grad()
            crit(net(Xt[sl]).reshape(-1, 1), Tt[sl]).backward()
            opt.step()
    relay.is_trained = True
    return relay


def main():
    only = sys.argv[1:] or CH
    out = {"snr_db": SNRS, "train_seeds": TRAIN_SEEDS, "epochs": EPOCHS,
           "samples": SAMPLES, "channels": {}}
    print("Sequence architectures on channels with memory")
    print(f"{len(ARCHS)} architectures x {len(TRAIN_SEEDS)} seeds, "
          f"trained on the channel under test\n")

    for name in only:
        spec = CHANNELS[name]
        ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
        bn, brelay = spec["baseline"]()
        base, _ = evaluate_two_hop(brelay, ch, hop2, mod, None)
        af, _ = evaluate_two_hop(AmplifyAndForwardRelay(), ch, hop2, mod, None)
        diag = baseline_diagnostics(base, af)
        print(f"  === {name} ({mod}, baseline {bn}: {diag['verdict']}) ===")
        rec = {"baseline": bn, "baseline_diagnostics": diag,
               "baseline_ber": [float(x) for x in base], "archs": {}}
        for aname, factory in ARCHS:
            runs, t0 = [], time.time()
            for ts in TRAIN_SEEDS:
                torch.manual_seed(ts)
                relay = factory(prefer_gpu=False)
                torch.manual_seed(ts)
                train_on_channel(relay, ch, mod, ts)
                ber, _ = evaluate_two_hop(relay, ch, hop2, mod, None)
                p = penalty_table(SNRS, ber, base)
                runs.append({"seed": ts, "ber": [float(x) for x in ber],
                             "db": (p["worst_db_penalty"]
                                    if p["targets_reached"] else float("nan"))})
            dbs = [r["db"] for r in runs if r["db"] == r["db"]]
            rec["archs"][aname] = {"runs": runs,
                                   "best_db": min(dbs) if dbs else float("nan"),
                                   "spread": (max(dbs) - min(dbs)) if len(dbs) > 1
                                             else float("nan")}
            print(f"    {aname:<16} dB vs {bn}: "
                  + "  ".join(f"{v:+6.2f}" for v in dbs)
                  + f"   best {min(dbs):+.2f}  spread {max(dbs)-min(dbs):.3f}"
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
        out["channels"][name] = rec
        with open("results/seq_models_on_memory.json", "w") as fh:
            json.dump(out, fh, indent=2)

    print("\n" + "=" * 74)
    for name, rec in out["channels"].items():
        best = min(rec["archs"].items(), key=lambda kv: kv[1]["best_db"])
        mlp = rec["archs"]["MLP-3K"]["best_db"]
        seq = min(v["best_db"] for k, v in rec["archs"].items() if k != "MLP-3K")
        verdict = ("sequence wins" if seq < mlp - 0.05 else
                   "feedforward wins" if mlp < seq - 0.05 else "tie")
        print(f"  {name:<14} best {best[0]} at {best[1]['best_db']:+.2f} dB  |  "
              f"MLP {mlp:+.2f} vs best sequence {seq:+.2f}  -> {verdict}")
    print("  saved results/seq_models_on_memory.json")


if __name__ == "__main__":
    main()
