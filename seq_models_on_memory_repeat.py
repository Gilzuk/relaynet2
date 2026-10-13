"""Is sequence-on-memory training deterministic within one environment?

The readout rerun (seq_models_on_memory_readout.py) and its controls
(seq_models_on_memory_rerun_controls.py) reproduced the published MLSE
baselines exactly but not the published per-seed penalties, so the spreads of
thesis Table 6.7 differ between the two software environments. This script
separates the two possible causes. It retrains three configurations, in a
fresh process, with the protocol, seeds, training data and evaluation draws of
seq_models_on_memory.py, and compares each with the value committed by the
earlier run in this environment. Identical BER curves mean training is
deterministic here, so the published-versus-rerun differences come from the
environment rather than from run-to-run noise.

The three configurations are the ones whose published and rerun values differ
most or matter most: Mamba-S6 seed 1 on BPSK ISI (published 5.45 dB, rerun
4.32), Transformer seed 2 on QPSK ISI (5.98 against 4.25), and MLP seed 0 on
BPSK ISI as a fast control.

Output: results/seq_models_on_memory_repeat.json
Usage: python seq_models_on_memory_repeat.py
"""
import json
import subprocess
import time

import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import make_mamba_3k, make_mlp_3k, make_transformer_3k
from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS
from seq_models_on_memory import train_on_channel

OUT = "results/seq_models_on_memory_repeat.json"
PUBLISHED = "results/seq_models_on_memory.json"
# (architecture, factory, channel, seed, committed rerun file, key in that file)
CASES = [
    ("MLP-3K", make_mlp_3k, "isi", 0,
     "results/seq_models_on_memory_rerun_controls.json", "MLP-3K"),
    ("Mamba-S6-3K", make_mamba_3k, "isi", 1,
     "results/seq_models_on_memory_readout.json", "Mamba-S6-3K|center"),
    ("Transformer-3K", make_transformer_3k, "isi_complex", 2,
     "results/seq_models_on_memory_rerun_controls.json", "Transformer-3K"),
]


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def main():
    published = json.load(open(PUBLISHED))
    out = {"description": __doc__.split("\n\n")[0], "snr_db": SNRS, "git_rev": git_rev(),
           "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cases": []}
    for aname, factory, name, ts, ref_file, ref_key in CASES:
        spec = CHANNELS[name]
        ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
        _, brelay = spec["baseline"]()
        base, _ = evaluate_two_hop(brelay, ch, hop2, mod, None)
        t0 = time.time()
        torch.manual_seed(ts)
        relay = factory(prefer_gpu=False)
        torch.manual_seed(ts)
        train_on_channel(relay, ch, mod, ts)
        ber, _ = evaluate_two_hop(relay, ch, hop2, mod, None)
        p = penalty_table(SNRS, ber, base)
        db = p["worst_db_penalty"] if p["targets_reached"] else float("nan")
        ref = next(r for r in json.load(open(ref_file))["channels"][name]["archs"][ref_key]["runs"]
                   if r["seed"] == ts)
        pub = next(r for r in published["channels"][name]["archs"][aname]["runs"]
                   if r["seed"] == ts)
        rec = {"arch": aname, "channel": name, "seed": ts, "ber": [float(x) for x in ber],
               "db": db, "committed_rerun_file": ref_file, "committed_rerun_db": ref["db"],
               "identical_to_committed_rerun": [float(x) for x in ber] == ref["ber"],
               "published_db": pub["db"],
               "identical_to_published": [float(x) for x in ber] == pub["ber"],
               "seconds": round(time.time() - t0)}
        out["cases"].append(rec)
        print(f"{aname:<15} {name:<12} seed {ts}: repeat {db:+.2f} dB, committed rerun "
              f"{ref['db']:+.2f} (identical BER: {rec['identical_to_committed_rerun']}), "
              f"published {pub['db']:+.2f} (identical BER: {rec['identical_to_published']})"
              f"   [{rec['seconds']}s]", flush=True)
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
