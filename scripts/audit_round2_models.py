#!/usr/bin/env python3
"""Inspect retained round-2 weights without training or rewriting experiment data.

Strict state-dict loading checks compatibility, not historical checkpoint identity.
The finite-input causality probe supplements source inspection; it is not a proof
for arbitrary architectures or a test of block-normalized transmission causality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import coded_matched_metrics as coded

from checkpoints.checkpoint_22_normalized_3k import build_all_3k
from run_experiments import build_base_relays
from seq_models_on_memory import ARCHS

ARCHIVE = "dc7c02c00cbebdfb566362a810ab4b45cb6abbe0"
SOURCE = "c0394f0da5dac449594b89a9794fe251fad75e76"
AUDITED_BASE = "d29f8ee96664e5df8baa21a7e02e2a91ca9c665d"
OUTPUT = ROOT / "results/round2_model_checkpoint_audit.json"
RUN = "gha-39024026594-1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def module(relay):
    relay = getattr(relay, "mlp_relay", relay)
    for attr in ("model", "_torch_model", "_torch_vae"):
        net = getattr(relay, attr, None)
        if net is not None:
            return net["model"] if isinstance(net, dict) else net
    raise ValueError(f"No Torch model: {type(relay).__name__}")


def check_checkpoint(path, relay, name, expected, expected_seed=None):
    # Archived sequence metadata stores torch.__version__ as TorchVersion.
    # Permit only this known string subclass; never enable unrestricted pickle.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    net = module(relay)
    if saved["architecture"] != str(net):
        raise ValueError(f"Architecture representation differs: {path}")
    if expected_seed is not None and saved["seed"] != expected_seed:
        raise ValueError(f"Seed metadata differs: {path}")
    if expected_seed is None and saved["source_commit"] != SOURCE:
        raise ValueError(f"Checkpoint source differs: {path}")
    net.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(t).all().item() for t in net.state_dict().values()):
        raise ValueError(f"Nonfinite saved tensor: {path}")
    count = sum(p.numel() for p in net.parameters())
    window = getattr(relay, "mlp_relay", relay).window_size
    if count != expected["params"] or window != expected["window"]:
        raise ValueError(f"Parameter/window inventory differs: {name}")
    net.eval()
    x = torch.linspace(-1.2, 1.3, 4 * window).reshape(4, window)
    sequential = name.startswith(("Transformer", "Mamba"))
    if sequential:
        x = x.unsqueeze(-1)
    infer = net.infer if name.startswith("VAE") else net
    with torch.no_grad():
        y = infer(x)
        again = infer(x)
    if y.shape != (4, 1) or not torch.isfinite(y).all() or not torch.equal(y, again):
        raise ValueError(f"Nonfinite, wrong-shape or stochastic eval output: {path}")
    row = {"model": name, "path": str(path.relative_to(ROOT)), "sha256": digest(path),
           "parameter_count": count, "window": window,
           "state_tensors": len(net.state_dict()), "strict_load": True,
           "architecture_text_matches": True, "finite_tensors": True,
           "same_input_eval_repeat_equal": True,
           "parameter_dtypes": sorted({str(p.dtype) for p in net.parameters()})}
    if expected_seed is not None:
        row["seed"] = expected_seed
    if name.startswith("Mamba"):
        changed = x.clone()
        changed[:, window // 2 + 1:, :] += 0.73
        with torch.no_grad():
            later = infer(changed)
        row["future_input_probe_max_abs_delta"] = float((y - later).abs().max())
        if not torch.equal(y, later):
            raise ValueError(f"Centre readout responds to future inputs: {path}")
        net.zero_grad(set_to_none=True)
        infer(x).square().sum().backward()
        unused = {k: p.numel() for k, p in net.named_parameters() if p.grad is None}
        expected_unused = {k: p.numel() for k, p in net.named_parameters()
                           if ".C_proj." in k} if not name.startswith("Mamba2") else {}
        if unused != expected_unused:
            raise ValueError(f"Unexpected disconnected parameter: {path}: {unused}")
        row["parameters_disconnected_from_output"] = unused
        row["disconnected_parameter_count"] = sum(unused.values())
    return row


def audit():
    torch.set_num_threads(2)
    torch.manual_seed(42)  # audit construction only; never replaces saved weights
    inventory = read(ROOT / "results/canonical_run_provenance.json")
    report = {"audited_base_revision": AUDITED_BASE, "evidence_archive_commit": ARCHIVE,
              "experiment_source_revision": SOURCE,
              "audit_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                    "torch": str(torch.__version__), "threads": 2},
              "scope": "Saved-weight compatibility and finite-input probes; no training or BER simulation.",
              "limitations": ["Historical canonical checkpoints and Torch states remain missing.",
                              "Successful loading does not recover historical checkpoints or prove BER agreement.",
                              "Causal local outputs still undergo block-wide normalization.",
                              "The coded run did not retain classifier weights or optimizer state.",
                              "Saved inference weights and RNG snapshots do not implement mid-training resume."],
              "checkpoint_checks": [], "jobs": {}, "inputs": {}}
    for job_name in ("canonical_native", "canonical_3k", "sequence_memory", "coded_matched"):
        base = ROOT / "results/reproduction" / f"{RUN}-{job_name}" / job_name
        job = read(base / "job.json")
        if job["source_commit"] != SOURCE or job["returncode"] != 0:
            raise ValueError(f"Wrong/unsuccessful archived job: {job_name}")
        report["inputs"][str((base / "job.json").relative_to(ROOT))] = digest(base / "job.json")
        artifacts = {}
        for item in job["artifacts"]:
            path = base / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Artifact hash differs: {path}")
            artifacts[item["path"]] = path
        rows = []
        if job_name.startswith("canonical_"):
            table = "table_5_2" if job_name == "canonical_native" else "table_5_3"
            relays = build_base_relays(gpu=False) if table == "table_5_2" else build_all_3k(prefer_gpu=False)
            for name, expected in inventory["relays"][table].items():
                stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
                key = f"outputs/reproduction_checkpoints/{job_name}/{stem}.pt"
                rows.append(check_checkpoint(artifacts[key], relays[name], name, expected))
        elif job_name == "sequence_memory":
            factories = dict(ARCHS)
            for channel in ("isi", "isi_complex", "composite"):
                for name, factory in factories.items():
                    canonical_name = "Mamba-3K" if name == "Mamba-S6-3K" else name
                    expected = inventory["relays"]["table_5_3"][canonical_name]
                    for seed in (0, 1, 2):
                        key = f"outputs/reproduction_checkpoints/sequence/{channel}_{name}_{seed}.pt"
                        row = check_checkpoint(artifacts[key], factory(prefer_gpu=False), name, expected, seed)
                        row["channel"] = channel
                        rows.append(row)
        else:
            # Current archived inventory is one results JSON, with no trained weights.
            if set(artifacts) != {"outputs/results/coded_matched_metrics.json"}:
                raise ValueError("Coded artifact inventory changed; review weight availability")
            classifier = coded.MLPQPSKClassifierRelay(window_size=coded.MLP_WINDOW, hidden_size=16, seed=0)
            examples = len((5, 10, 15)) * coded.MLP_TRAIN_FRAMES * coded.FRAME_SYMBOLS
            if classifier.n_params() != 756 or examples != 1212000:
                raise ValueError("Coded model or training-example inventory differs")
            report["coded_model_inventory"] = {
                "scope": "Untrained construction and source constants only; trained weights absent",
                "parameter_count": classifier.n_params(), "input_features": classifier.input_size,
                "hidden_units": classifier.hidden_size, "classes": classifier.num_classes,
                "parameter_dtype": str(classifier.W1.dtype), "window": classifier.window_size,
                "training_frames_per_snr": coded.MLP_TRAIN_FRAMES,
                "frame_symbols": coded.FRAME_SYMBOLS, "training_windows": examples,
                "input_order": "I window followed by Q window"}
        report["checkpoint_checks"].extend(rows)
        report["jobs"][job_name] = {"checked_checkpoints": len(rows),
                                    "artifact_hashes_verified": len(artifacts),
                                    "weight_replay_available": bool(rows)}
    if len(report["checkpoint_checks"]) != 48:
        raise ValueError("Expected twelve canonical and thirty-six sequence checkpoints")
    for rel in ("results/canonical_run_provenance.json", "run_experiments.py", "seq_models_on_memory.py",
                "relaynet/relays/genai.py", "relaynet/relays/hybrid.py", "relaynet/relays/vae.py",
                "checkpoints/checkpoint_18_transformer_relay.py", "checkpoints/checkpoint_20_mamba_s6_relay.py",
                "checkpoints/checkpoint_22_normalized_3k.py", "checkpoints/checkpoint_23_mamba2_relay.py",
                "coded_matched_metrics.py", "coded_learned_relay.py", "relaynet/relays/mlp.py",
                "relaynet/coding/convolutional.py"):
        report["inputs"][rel] = digest(ROOT / rel)
    report["status"] = "PASS_WITH_DOCUMENTED_LIMITS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with the committed audit without overwriting it")
    args = parser.parse_args()
    result = audit()
    if args.check:
        saved = read(OUTPUT)
        # Record the actual audit environment but don't require the same host version.
        saved.pop("audit_environment", None)
        comparable = {k: v for k, v in result.items() if k != "audit_environment"}
        if comparable != saved:
            raise SystemExit("Model checkpoint audit differs from saved evidence")
    else:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS_WITH_DOCUMENTED_LIMITS: 48 checkpoints; coded weights absent; no training performed")


if __name__ == "__main__":
    main()
