"""Mamba relays on channels with memory, read after the whole window.

The committee audit (docs/committee_audit_2026-10-13.md, finding A2) found that
both Mamba relays scan their 11-sample window causally and are read at its
centre, so the decision on the centre symbol never sees the five samples after
it. On the three-tap channels of the sequence-on-memory study
(seq_models_on_memory.py, thesis Table 6.7) those samples carry 42.5% (ISI) and
34% (composite) of the centre symbol's energy, so the published Mamba rows test
relays without look-ahead.

This script reruns those rows with readout="last": the same models, parameter
counts and training protocol as seq_models_on_memory.py, but the output is read
after the scan has consumed the whole window (the residual skip still adds the
centre sample). The published centre readout is rerun alongside as a control,
on the same seeds, training data and evaluation draws, so the comparison
isolates the readout; the control also shows whether the published centre
values reproduce in this environment.

Before any training, a self-check confirms the property under test: perturbing
the samples after the centre leaves a centre-read model's output unchanged and
changes a last-read model's output.

Output: results/seq_models_on_memory_readout.json (written after every channel).
Usage: python seq_models_on_memory_readout.py [channel ...]   (default: all three)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import make_mamba_3k, make_mamba2_3k
from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS
from seq_models_on_memory import CH, TRAIN_SEEDS, train_on_channel

OUT = "results/seq_models_on_memory_readout.json"
PUBLISHED = "results/seq_models_on_memory.json"
ARCHS = [("Mamba-S6-3K", make_mamba_3k), ("Mamba2-3K", make_mamba2_3k)]
READOUTS = ["center", "last"]


def causality_self_check():
    """Centre readout must ignore the samples after the centre; last readout must not."""
    out = {}
    for name, factory in ARCHS:
        out[name] = {}
        for ro in READOUTS:
            torch.manual_seed(0)
            net = factory(prefer_gpu=False, readout=ro).model.eval()
            torch.manual_seed(1)
            x = torch.randn(8, 11, 1)
            x2 = x.clone()
            x2[:, 6:, :] += torch.randn(8, 5, 1)
            with torch.no_grad():
                d = float((net(x) - net(x2)).abs().max())
            out[name][ro] = d
        assert out[name]["center"] == 0.0, f"{name}: centre readout sees future samples"
        assert out[name]["last"] > 0.0, f"{name}: last readout ignores future samples"
    return out


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def main():
    only = sys.argv[1:] or CH
    published = json.load(open(PUBLISHED)) if os.path.exists(PUBLISHED) else {"channels": {}}
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update({"description": __doc__.split("\n\n")[0],
                "snr_db": SNRS, "train_seeds": TRAIN_SEEDS,
                "readouts": READOUTS, "git_rev": git_rev(),
                "causality_self_check": causality_self_check()})
    out.setdefault("channels", {})
    print("causality self-check (max output change when the 5 samples after the centre move):")
    for a, v in out["causality_self_check"].items():
        print(f"  {a:<12} center {v['center']:.3g}   last {v['last']:.3g}")

    for name in only:
        spec = CHANNELS[name]
        ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
        bn, brelay = spec["baseline"]()
        base, _ = evaluate_two_hop(brelay, ch, hop2, mod, None)
        pub = published["channels"].get(name, {})
        pub_base = np.asarray(pub.get("baseline_ber", []), float)
        base_diff = (float(np.max(np.abs(pub_base - base))) if pub_base.size == base.size
                     else None)
        print(f"\n=== {name} ({mod}, baseline {bn}; max |baseline - published| = {base_diff}) ===")
        rec = {"baseline": bn, "baseline_ber": [float(x) for x in base],
               "baseline_max_abs_diff_vs_published": base_diff, "archs": {}}
        for aname, factory in ARCHS:
            for ro in READOUTS:
                runs, t0 = [], time.time()
                for ts in TRAIN_SEEDS:
                    torch.manual_seed(ts)
                    relay = factory(prefer_gpu=False, readout=ro)
                    torch.manual_seed(ts)
                    train_on_channel(relay, ch, mod, ts)
                    ber, _ = evaluate_two_hop(relay, ch, hop2, mod, None)
                    p = penalty_table(SNRS, ber, base)
                    runs.append({"seed": ts, "ber": [float(x) for x in ber],
                                 "db": (p["worst_db_penalty"]
                                        if p["targets_reached"] else float("nan"))})
                dbs = [r["db"] for r in runs if r["db"] == r["db"]]
                entry = {"runs": runs,
                         "best_db": min(dbs) if dbs else float("nan"),
                         "spread": (max(dbs) - min(dbs)) if len(dbs) > 1 else float("nan")}
                if ro == "center" and aname in pub.get("archs", {}):
                    pruns = {r["seed"]: r["db"] for r in pub["archs"][aname]["runs"]}
                    entry["published_db_by_seed"] = pruns
                    entry["max_abs_diff_vs_published_db"] = max(
                        abs(r["db"] - pruns[r["seed"]]) for r in runs
                        if r["seed"] in pruns and r["db"] == r["db"])
                rec["archs"][f"{aname}|{ro}"] = entry
                print(f"  {aname:<12} readout={ro:<6} dB vs {bn}: "
                      + "  ".join(f"{v:+6.2f}" for v in dbs)
                      + (f"   best {min(dbs):+.2f}  spread {max(dbs) - min(dbs):.2f}" if dbs else "")
                      + (f"   (published centre best {pub['archs'][aname]['best_db']:+.2f})"
                         if ro == "center" and aname in pub.get("archs", {}) else "")
                      + f"   [{time.time() - t0:.0f}s]", flush=True)
        out["channels"][name] = rec
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
