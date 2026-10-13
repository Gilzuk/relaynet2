"""Same-environment rerun of the MLP-3K and Transformer-3K rows of Table 6.7.

Companion to seq_models_on_memory_readout.py. That script reruns the two
Mamba relays with the published centre readout and with a full-window readout.
Its centre-readout control showed that the published per-seed Mamba penalties
are not reproduced in a different software environment even with the same
seeds: the best-of-three value agrees closely, the per-seed values and the
spread do not.

Comparing the new Mamba numbers with published MLP and Transformer values
would therefore mix environments. This script reruns the two other
architectures of the sequence-on-memory study (seq_models_on_memory.py) in the
current environment, with the same protocol, seeds, training data and
evaluation draws, so all four architectures can be compared in one
environment. The published values are stored alongside for reference.

Output: results/seq_models_on_memory_rerun_controls.json (written after every channel).
Usage: python seq_models_on_memory_rerun_controls.py [channel ...]   (default: all three)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import make_mlp_3k, make_transformer_3k
from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS
from seq_models_on_memory import CH, TRAIN_SEEDS, train_on_channel

OUT = "results/seq_models_on_memory_rerun_controls.json"
PUBLISHED = "results/seq_models_on_memory.json"
ARCHS = [("MLP-3K", make_mlp_3k), ("Transformer-3K", make_transformer_3k)]


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
    out.update({"description": __doc__.split("\n\n")[0], "snr_db": SNRS,
                "train_seeds": TRAIN_SEEDS, "git_rev": git_rev()})
    out.setdefault("channels", {})
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
            entry = {"runs": runs,
                     "best_db": min(dbs) if dbs else float("nan"),
                     "spread": (max(dbs) - min(dbs)) if len(dbs) > 1 else float("nan")}
            if aname in pub.get("archs", {}):
                entry["published_db_by_seed"] = {r["seed"]: r["db"]
                                                 for r in pub["archs"][aname]["runs"]}
            rec["archs"][aname] = entry
            print(f"  {aname:<15} dB vs {bn}: " + "  ".join(f"{v:+6.2f}" for v in dbs)
                  + (f"   best {min(dbs):+.2f}  spread {max(dbs) - min(dbs):.2f}" if dbs else "")
                  + (f"   (published best {pub['archs'][aname]['best_db']:+.2f})"
                     if aname in pub.get("archs", {}) else "")
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
        out["channels"][name] = rec
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
