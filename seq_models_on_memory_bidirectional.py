"""Mamba relays on channels with memory, with a bidirectional centre readout.

Follow-up to seq_models_on_memory_readout.py (committee-audit finding A2).
There, reading the causal scan after the whole window (readout="last") did not
give the Mamba relays usable look-ahead: on the BPSK ISI channel both stalled
near BER 0.12 even at 20 dB, far worse than the published centre readout.

The usual way to give a causal sequence model both sides of a window is a
bidirectional scan. Here the blocks run over the window and over the reversed
window with shared weights, both outputs are read at the centre and averaged,
and the residual still adds the centre sample. The parameter count is
unchanged (3,027 and 3,004 at the 3K budget), so the equal-budget comparison
of the sequence-on-memory study (thesis Table 6.7) is preserved.

The protocol, seeds, training data and evaluation draws are those of
seq_models_on_memory.py; a self-check first confirms that the bidirectional
models respond to the samples on both sides of the centre.

Output: results/seq_models_on_memory_bidirectional.json (written after every channel).
Usage: python seq_models_on_memory_bidirectional.py [channel ...]   (default: all three)
"""
import json
import os
import subprocess
import sys
import time

import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import make_mamba_3k, make_mamba2_3k
from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS
from seq_models_on_memory import CH, TRAIN_SEEDS, train_on_channel

OUT = "results/seq_models_on_memory_bidirectional.json"
ARCHS = [("Mamba-S6-3K", make_mamba_3k), ("Mamba2-3K", make_mamba2_3k)]
READOUT = "bidirectional"


def self_check():
    """The bidirectional models must respond to the samples on both sides of the centre."""
    out = {}
    for name, factory in ARCHS:
        torch.manual_seed(0)
        net = factory(prefer_gpu=False, readout=READOUT).model.eval()
        torch.manual_seed(1)
        x = torch.randn(8, 11, 1)
        xf, xp = x.clone(), x.clone()
        xf[:, 6:, :] += torch.randn(8, 5, 1)
        xp[:, :5, :] += torch.randn(8, 5, 1)
        with torch.no_grad():
            out[name] = {"future": float((net(x) - net(xf)).abs().max()),
                         "past": float((net(x) - net(xp)).abs().max()),
                         "params": sum(p.numel() for p in net.parameters())}
        assert out[name]["future"] > 0 and out[name]["past"] > 0, name
    return out


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def main():
    only = sys.argv[1:] or CH
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update({"description": __doc__.split("\n\n")[0], "snr_db": SNRS,
                "train_seeds": TRAIN_SEEDS, "readout": READOUT, "git_rev": git_rev(),
                "self_check": self_check()})
    out.setdefault("channels", {})
    print("self-check:", out["self_check"])
    for name in only:
        spec = CHANNELS[name]
        ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
        bn, brelay = spec["baseline"]()
        base, _ = evaluate_two_hop(brelay, ch, hop2, mod, None)
        print(f"\n=== {name} ({mod}, baseline {bn}) ===")
        rec = {"baseline": bn, "baseline_ber": [float(x) for x in base], "archs": {}}
        for aname, factory in ARCHS:
            runs, t0 = [], time.time()
            for ts in TRAIN_SEEDS:
                torch.manual_seed(ts)
                relay = factory(prefer_gpu=False, readout=READOUT)
                torch.manual_seed(ts)
                train_on_channel(relay, ch, mod, ts)
                ber, _ = evaluate_two_hop(relay, ch, hop2, mod, None)
                p = penalty_table(SNRS, ber, base)
                runs.append({"seed": ts, "ber": [float(x) for x in ber],
                             "db": (p["worst_db_penalty"]
                                    if p["targets_reached"] else float("nan"))})
            dbs = [r["db"] for r in runs if r["db"] == r["db"]]
            rec["archs"][f"{aname}|{READOUT}"] = {
                "runs": runs, "best_db": min(dbs) if dbs else float("nan"),
                "spread": (max(dbs) - min(dbs)) if len(dbs) > 1 else float("nan")}
            print(f"  {aname:<12} readout={READOUT} dB vs {bn}: "
                  + "  ".join(f"{v:+6.2f}" for v in dbs)
                  + (f"   best {min(dbs):+.2f}  spread {max(dbs) - min(dbs):.2f}" if dbs else "")
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
        out["channels"][name] = rec
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
