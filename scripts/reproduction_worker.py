#!/usr/bin/env python3
"""Execute a frozen suite job inside its disposable repository worktree."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_round2 import jobs, write_json, git


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(x) for x in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def baselines():
    import numpy as np
    import run_experiments as R
    from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
    snrs = np.arange(0, 21, 2)
    data = R.evaluate_relays({"AF": AmplifyAndForwardRelay(), "DF": DecodeAndForwardRelay()},
                            snrs, channel_fn=R._CHANNELS["rayleigh"][1], modulation="qpsk",
                            bits_per_trial=10000, num_trials=10)
    out = {"snr_range": snrs.tolist(), "results": {
        k: {field: v[field] for field in ("ber_mean", "ber_trials")} for k, v in data.items()}}
    write_json(ROOT / "results/reproduction_canonical_af_df.json", safe(out))


def sequence():
    import torch
    from seq_models_on_memory import (ARCHS, CH, TRAIN_SEEDS, TRAIN_SNRS, SAMPLES, EPOCHS,
                                      BATCH, LR, train_on_channel, torch_module)
    from mlp_min_size_all_channels import CHANNELS, evaluate_two_hop, SNRS, N_TRIALS, BITS_PER_TRIAL
    from relaynet.relays import AmplifyAndForwardRelay
    from penalty_common_targets import seeds_summary
    from ber_metrics import penalty_common
    out_path = ROOT / "results/reproduction_seq_memory.json"
    out = {"snr_db": SNRS, "train_seeds": TRAIN_SEEDS, "epochs": EPOCHS, "samples": SAMPLES,
           "source_commit": git("rev-parse", "HEAD"), "channels": {}, "config": {
               "optimizer": "Adam", "lr": LR, "batch": BATCH, "checkpoint": "final epoch",
               "train_snrs": TRAIN_SNRS, "train_source_seeds": [1000 + s for s in TRAIN_SEEDS],
               "eval_trials": N_TRIALS, "eval_bits_per_trial": BITS_PER_TRIAL,
               "padding": "training: zero; inference: edge repeat", "torch_version": torch.__version__,
               "dtype": "float32", "device": "cpu", "threads": torch.get_num_threads()},
           "penalty_policy": "Common baseline-reachable targets; failed targets retained as bounds. Legacy db/best_db/spread fields are not authoritative."}
    checkpoints = ROOT / "reproduction_checkpoints/sequence"
    checkpoints.mkdir(parents=True)
    for name in CH:
        spec = CHANNELS[name]
        ch, hop2, mod = spec["make"](1), spec["hop2"](), spec["mod"]
        bn, brelay = spec["baseline"]()
        base, base_trials = evaluate_two_hop(brelay, ch, hop2, mod, "baseline")
        # Preserve the original driver's call order and random-number use.
        af, af_trials = evaluate_two_hop(AmplifyAndForwardRelay(), ch, hop2, mod, "AF")
        rec = {"baseline": bn, "baseline_ber": base.tolist(), "baseline_trials": base_trials.tolist(),
               "af_ber": af.tolist(), "af_trials": af_trials.tolist(), "archs": {}}
        out["channels"][name] = rec
        for arch, factory in ARCHS:
            runs = []
            rec["archs"][arch] = {"runs": runs}
            for seed in TRAIN_SEEDS:
                print(f"START {name} {arch} seed={seed}", flush=True)
                torch.manual_seed(seed)
                relay = factory(prefer_gpu=False)
                torch.manual_seed(seed)
                train_on_channel(relay, ch, mod, seed)
                net = torch_module(relay)
                checkpoint = checkpoints / f"{name}_{arch}_{seed}.pt"
                torch.save({"state_dict": net.state_dict(), "architecture": str(net), "seed": seed,
                            "torch_rng_after_training": torch.get_rng_state(), "config": out["config"]}, checkpoint)
                ber, trials = evaluate_two_hop(relay, ch, hop2, mod, f"{name} {arch} seed {seed}")
                p = penalty_common(SNRS, ber, base)
                runs.append({"seed": seed, "ber": ber.tolist(), "ber_trials": trials.tolist(),
                             "penalty_common": p, "checkpoint": str(checkpoint.relative_to(ROOT))})
                rec["archs"][arch]["common_summary"] = seeds_summary(SNRS, [r["ber"] for r in runs], base)
                write_json(out_path, safe(out))
                print(f"END {name} {arch} seed={seed}; penalty={p['worst_db']}; lower_bound={p['lower_bound']}", flush=True)
        summaries = {a: v["common_summary"] for a, v in rec["archs"].items()}
        best = summaries["MLP-3K"]
        rec["ranking_check"] = {
            "point_estimate_order": sorted(summaries, key=lambda a: summaries[a]["new_best"]),
            "mlp_best_resolved_by_bounds": not best["new_best_flag"] and all(
                best["new_best"] < s["new_best"] for a, s in summaries.items() if a != "MLP-3K"),
            "interpretation": "Ordering of best-of-three penalties, not statistical superiority across training populations."}
        write_json(out_path, safe(out))


def canonical(job):
    import numpy as np
    import torch
    import run_experiments as R
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    checkpoint_dir = ROOT / "reproduction_checkpoints" / job["id"]
    checkpoint_dir.mkdir(parents=True)
    torch.save(torch.get_rng_state(), checkpoint_dir / "torch_rng_before_construction.pt")
    write_json(checkpoint_dir / "protocol.json", {
        "stage": job["stage"], "new_python_numpy_torch_seed": seed,
        "training_data_seed": 42, "epochs": 100, "trials": 10, "bits_per_trial": 10000,
        "snrs_db": list(range(0, 21, 2)), "torch_version": torch.__version__,
        "threads": torch.get_num_threads(), "checkpoint_rule": "unchanged original per-family rule",
        "historical_limit": "Original PyTorch seed and learned checkpoint absent; this is new protocol replication."})
    original = R.evaluate_relays
    saved = set()

    def capture(relays, *args, **kwargs):
        for name, relay in relays.items():
            if name in ("AF", "DF") or name in saved:
                continue
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
            target = getattr(relay, "mlp_relay", relay)
            net = next((getattr(target, k) for k in ("model", "_torch_model", "_torch_vae")
                        if getattr(target, k, None) is not None), None)
            if isinstance(net, dict):
                net = net.get("model")
            if net is None:
                raise RuntimeError(f"Cannot preserve the learned checkpoint for {name}")
            torch.save({"state_dict": net.state_dict(), "architecture": str(net), "name": name,
                        "torch_rng_after_training": torch.get_rng_state(), "source_commit": git("rev-parse", "HEAD")},
                       checkpoint_dir / (stem + ".pt"))
            saved.add(name)
        return original(relays, *args, **kwargs)

    R.evaluate_relays = capture
    sys.argv = ["run_experiments.py", "--exp", job["stage"], "--retrain", "--skip-relays", "cgan",
                "--num-trials", "10", "--bits-per-trial", "10000", "--snr-min", "0", "--snr-max", "20", "--snr-step", "2"]
    R.main()
    expected = {"MLP (169p)", "Hybrid", "VAE", "Transformer", "Mamba S6", "Mamba2 (SSD)"} if job["stage"] == "7.10" else {
        "MLP-3K", "Hybrid-3K", "VAE-3K", "Transformer-3K", "Mamba-3K", "Mamba2-3K"}
    if saved != expected:
        raise RuntimeError(f"Missing/unexpected learned models: {saved ^ expected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=[j["id"] for j in jobs()])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    job = next(j for j in jobs() if j["id"] == args.job)
    try:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    except ImportError:
        if job["worker"] in ("canonical", "sequence"):
            raise
    if job["worker"] == "script":
        sys.argv = [job["script"]]
        runpy.run_path(str(ROOT / job["script"]), run_name="__main__")
    elif job["worker"] == "baselines":
        baselines()
    elif job["worker"] == "sequence":
        sequence()
    elif job["worker"] == "canonical":
        canonical(job)


if __name__ == "__main__":
    main()
