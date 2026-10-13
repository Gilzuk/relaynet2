"""Retrain one sequence-on-memory configuration and fingerprint the result.

Companion to seq_models_on_memory_repeat.py. That script found that two of
three configurations did not reproduce the committed rerun values. The
committed values came from an earlier machine of the same cloud environment;
the session moved to a new machine (same filesystem, so the same installed
PyTorch) before the repeat. This probe separates the remaining possibilities
by retraining one configuration in a fresh process and recording a hash of
the trained weights, the penalty at full precision, the BER curve, and the
machine it ran on. Identical hashes across processes on one machine mean
training is deterministic there; a different value on another machine is
then a machine effect.

The protocol, seeds, training data and evaluation draws are those of
seq_models_on_memory.py.

Output: one JSON line per run appended to results/seq_models_on_memory_repeat_probe.jsonl
Usage: python seq_models_on_memory_repeat_probe.py ARCH CHANNEL SEED [note]
       ARCH in MLP-3K, Transformer-3K, Mamba-S6-3K, Mamba2-3K
"""
import datetime
import hashlib
import json
import platform
import subprocess
import sys

import torch

from ber_metrics import penalty_table
from checkpoints.checkpoint_22_normalized_3k import (
    make_mlp_3k, make_transformer_3k, make_mamba_3k, make_mamba2_3k)
from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS
from seq_models_on_memory import torch_module, train_on_channel

OUT = "results/seq_models_on_memory_repeat_probe.jsonl"
FACTORIES = {"MLP-3K": make_mlp_3k, "Transformer-3K": make_transformer_3k,
             "Mamba-S6-3K": make_mamba_3k, "Mamba2-3K": make_mamba2_3k}


def machine():
    """What the run executed on: CPU model and vector extensions, boot time, threads."""
    info = {"platform": platform.platform(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    try:
        cpu = open("/proc/cpuinfo").read()
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in cpu.splitlines()
                                 if l.startswith("model name"))
        flags = next(l.split(":", 1)[1].split() for l in cpu.splitlines()
                     if l.startswith("flags"))
        info["cpu_flags"] = sorted(f for f in flags if f.startswith(("avx", "amx", "fma")))
        info["cpus"] = cpu.count("processor\t:")
    except Exception:  # noqa
        pass
    try:
        info["booted"] = subprocess.run(["uptime", "-s"], capture_output=True, text=True,
                                        check=True).stdout.strip()
    except Exception:  # noqa
        pass
    return info


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def main():
    aname, name, ts = sys.argv[1], sys.argv[2], int(sys.argv[3])
    note = " ".join(sys.argv[4:])
    spec = CHANNELS[name]
    ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
    _, brelay = spec["baseline"]()
    base, _ = evaluate_two_hop(brelay, ch, hop2, mod, None)
    torch.manual_seed(ts)
    relay = FACTORIES[aname](prefer_gpu=False)
    torch.manual_seed(ts)
    train_on_channel(relay, ch, mod, ts)
    weights = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                      for p in torch_module(relay).parameters())).hexdigest()
    ber, _ = evaluate_two_hop(relay, ch, hop2, mod, None)
    p = penalty_table(SNRS, ber, base)
    rec = {"arch": aname, "channel": name, "seed": ts, "note": note,
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git_rev": git_rev(), "weights_sha256": weights,
           "db": p["worst_db_penalty"] if p["targets_reached"] else float("nan"),
           "ber": [float(x) for x in ber], "machine": machine()}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{aname} {name} seed {ts}: {rec['db']!r} dB, weights {weights[:16]}"
          + (f"  ({note})" if note else ""), flush=True)


if __name__ == "__main__":
    main()
