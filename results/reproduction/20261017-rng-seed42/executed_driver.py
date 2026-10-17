#!/usr/bin/env python3
"""Check historical --seed 42 against an explicitly seeded full-budget MLP replay.

The source directory must be a clean checkout of the recorded historical commit.
Four fresh processes use the original native-model construction order, followed
by the first MLP's original 100-epoch training and complete QPSK evaluation.
Only that MLP is retrained; this is an RNG control, not a replacement thesis run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", *args], cwd=source, text=True).strip()


def worker(args):
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import run_experiments as experiment
    import relaynet.relays.genai as genai

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    if args.worker == "all_seeded":
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
    initial_seed = torch.initial_seed()

    def array_hash(value):
        value = np.asarray(value)
        h = hashlib.sha256()
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes(order="C"))
        return h.hexdigest()

    def weights(net, filename):
        arrays = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        np.savez(out / filename, **arrays)
        return {k: array_hash(v) for k, v in arrays.items()}

    def rng_record(label):
        state = np.random.get_state()
        torch.save(torch.get_rng_state(), out / f"torch_rng_{label}.pt")
        save(out / f"rng_{label}.json", {
            "python_state": random.getstate(),
            "numpy_legacy_state": [state[0], state[1].tolist(), *state[2:]],
            "torch_initial_seed": torch.initial_seed(),
            "torch_state_sha256": array_hash(torch.get_rng_state().numpy())})

    rng_record("before_construction")
    # Construction of the other relays consumes RNG draws before the first
    # MLP is trained. Keep that order, including the subsequently skipped cGAN.
    experiment.SKIP_RELAYS = {"cgan"}
    relays = experiment.build_base_relays(gpu=False)
    expected_relays = {"AF", "DF", "MLP (169p)", "Hybrid", "VAE", "Transformer", "Mamba S6", "Mamba2 (SSD)"}
    if set(relays) != expected_relays:
        raise RuntimeError(f"Historical construction order is incomplete: {set(relays) ^ expected_relays}")
    relay = relays["MLP (169p)"]
    if not relay._use_torch:
        raise RuntimeError("This control requires the original Torch MLP path")
    initial = weights(relay._torch_model, "initial_weights.npz")
    rng_record("before_training")
    generated = []
    original_generator = genai.generate_training_targets

    def record_data(*a, **kw):
        clean, noisy = original_generator(*a, **kw)
        generated.append({"samples": a[0], "snr_db": a[1], "seed": kw["seed"],
                          "clean_sha256": array_hash(clean), "noisy_sha256": array_hash(noisy)})
        return clean, noisy

    genai.generate_training_targets = record_data
    # Identical to the first training call in the historical native protocol.
    relay.train(training_snrs=[5, 10, 15], num_samples=25000, epochs=100, seed=42)
    final = weights(relay._torch_model, "final_weights.npz")
    rng_record("after_training")
    result = experiment.evaluate_relays(
        {"MLP (169p)": relay}, np.arange(0, 21, 2),
        channel_fn=experiment._CHANNELS["rayleigh"][1], modulation="qpsk",
        bits_per_trial=10000, num_trials=10)["MLP (169p)"]
    save(out / "result.json", {
        "mode": args.worker, "source_commit": git(source, "rev-parse", "HEAD"),
        "torch_initial_seed": initial_seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "initial_weights": initial, "final_weights": final,
        "training_data": generated, "snr_db": list(range(0, 21, 2)),
        "ber_mean": np.asarray(result["ber_mean"]).tolist(),
        "ber_trials": np.asarray(result["ber_trials"]).tolist(),
        "protocol": {"train_seed_argument": 42, "samples_requested": 25000,
                     "training_windows": 24987, "epochs": 100, "batch": 32,
                     "optimizer": "Adam", "lr": 0.01, "checkpoint": "final epoch",
                     "evaluation_trials": 10, "bits_per_trial": 10000,
                     "evaluation_seeds": list(range(10))},
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "torch": torch.__version__, "platform": platform.platform(),
                        "threads": torch.get_num_threads(), "device": "cpu"},
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": datetime.now(timezone.utc).isoformat()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--worker", choices=["legacy_default", "all_seeded"])
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    if out.exists():
        ap.error("refusing to overwrite prior evidence")
    if git(source, "diff", "HEAD", "--", "*.py"):
        ap.error("historical source code must be clean")
    expected = git(source, "rev-parse", "b4059e6")
    if git(source, "rev-parse", "HEAD") != expected:
        ap.error("expected the native canonical source b4059e6")
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / "executed_driver.py")
    published = subprocess.check_output(
        ["git", "show", "69a470c:results/modulation/qpsk_rayleigh.json"], cwd=source)
    (out / "published.json").write_bytes(published)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
           "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""}
    commands = []
    for mode in ("all_seeded", "legacy_default"):
        for repeat in (1, 2):
            case = out / f"{mode}_{repeat}"
            case.mkdir()
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--source-dir", str(source), "--output-dir", str(case), "--worker", mode]
            commands.append(command)
            print(f"START {case.name}", flush=True)
            with (case / "run.log").open("w") as log:
                process = subprocess.run(command, cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=1800)
            save(case / "execution.json", {"command": command, "returncode": process.returncode})
            if process.returncode:
                raise RuntimeError(f"{case.name} failed; keep its log")
            print(f"END {case.name}", flush=True)
    import numpy as np
    cases = {p.name: json.loads((p / "result.json").read_text())
             for p in out.iterdir() if p.is_dir()}
    reference = json.loads(published)["results"]["MLP (169p)"]

    def equality(a, b):
        trials_a, trials_b = np.asarray(a["ber_trials"]), np.asarray(b["ber_trials"])
        return {"initial_weights_exact": a["initial_weights"] == b["initial_weights"],
                "training_data_exact": a["training_data"] == b["training_data"],
                "final_weights_exact": a["final_weights"] == b["final_weights"],
                "ber_trials_exact": bool(np.array_equal(trials_a, trials_b)),
                "differing_trial_ber_cells": int(np.count_nonzero(trials_a != trials_b)),
                "max_abs_mean_ber_difference": float(np.max(np.abs(
                    np.asarray(a["ber_mean"]) - np.asarray(b["ber_mean"]))))}

    comparison = {mode: equality(cases[f"{mode}_1"], cases[f"{mode}_2"])
                  for mode in ("legacy_default", "all_seeded")}
    comparison["versus_historical"] = {name: {
        "ber_trials_exact": bool(np.array_equal(case["ber_trials"], reference["ber_trials"])),
        "max_abs_mean_ber_difference": float(np.max(np.abs(
            np.asarray(case["ber_mean"]) - np.asarray(reference["ber_mean"]))))}
        for name, case in cases.items()}
    comparison["status"] = ("SEEDED_REPLAY_EXACT_ON_THIS_HOST" if all(
        comparison["all_seeded"][k] for k in (
            "initial_weights_exact", "training_data_exact", "final_weights_exact", "ber_trials_exact"))
        else "SEEDED_REPLAY_FAILED")
    comparison["limits"] = (
        "Native MLP only, full original budget and construction order. Explicit seed 42 plus "
        "deterministic algorithms is a new controlled run, not a recovered historical RNG state. "
        "Same-host repetition is not a guarantee across hardware or library versions.")
    save(out / "comparison.json", comparison)
    save(out / "manifest.json", {
        "historical_source": expected, "published_data_commit": git(source, "rev-parse", "69a470c"),
        "driver_sha256": sha(out / "executed_driver.py"), "commands": commands,
        "environment_overrides": {k: env[k] for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "PYTHONHASHSEED", "MPLBACKEND", "CUDA_VISIBLE_DEVICES")},
        "artifacts": [{"path": str(p.relative_to(out)), "sha256": sha(p)}
                      for p in sorted(out.rglob("*")) if p.is_file()]})
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
