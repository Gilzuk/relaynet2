#!/usr/bin/env python
"""
Benchmark: Transformer vs Mamba S6 vs Mamba2 (SSD) at window_size=256.

Tests the hypothesis that Mamba-2's chunk-parallel SSD becomes faster
than Mamba S6's sequential scan when the context length is large.

- 1 000 symbols for training (quick)
- window_size = 256
- Same d_model=32, d_state=16, num_layers=2 for all
- Times: model build, training (20 epochs), inference (10 000 symbols)
- Reports param count, training time, inference time, BER @ 10 dB
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "checkpoints"))

import numpy as np
import torch

from checkpoints.checkpoint_01_channel import awgn_channel
from checkpoints.checkpoint_02_modulation import bpsk_modulate, calculate_ber
from checkpoints.checkpoint_03_nodes import Source, Destination

# ── Model imports ────────────────────────────────────────────────────
from checkpoints.checkpoint_18_transformer_relay import (
    TransformerRelay, PositionalEncoding,
)
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelay
from checkpoints.checkpoint_23_mamba2_relay import Mamba2Relay

# ── Config ───────────────────────────────────────────────────────────
WINDOW      = 255        # odd so center token is unambiguous
D_MODEL     = 32
D_STATE     = 16
NUM_LAYERS  = 2
NUM_HEADS   = 4          # Transformer only
CHUNK_SIZE  = 32         # Mamba2 only
TRAIN_SYMS  = 1_000      # tiny dataset — we only care about speed
TRAIN_EPOCHS = 20
TRAIN_SNR   = [10]       # single SNR for quick training
INFER_BITS  = 10_000     # inference workload
INFER_SNR   = 10
BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEP = "=" * 72

# ── Helpers ──────────────────────────────────────────────────────────

def make_training_data(window_size, num_samples, snrs, device):
    """Create windowed training tensors."""
    half = window_size // 2
    X_all, y_all = [], []
    for snr in snrs:
        np.random.seed(42 + int(snr))
        bits = np.random.randint(0, 2, num_samples)
        clean = bpsk_modulate(bits)
        noisy = awgn_channel(clean, snr)
        for i in range(half, len(noisy) - half):
            X_all.append(noisy[i - half: i + half + 1])
            y_all.append(clean[i])
    X = torch.FloatTensor(np.array(X_all)).unsqueeze(-1).to(device)
    y = torch.FloatTensor(np.array(y_all).reshape(-1, 1)).to(device)
    return X, y


def train_model(model, X, y, epochs, batch_size, device):
    """Train for *epochs* and return wall-clock seconds."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for ep in range(epochs):
        idx = torch.randperm(len(X), device=device)
        Xs, ys = X[idx], y[idx]
        for i in range(0, len(X), batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def infer_ber(model, window_size, num_bits, snr, device):
    """Run batched inference and return (wall_seconds, ber)."""
    model.eval()
    np.random.seed(42)
    bits = np.random.randint(0, 2, num_bits)
    clean = bpsk_modulate(bits)
    noisy = awgn_channel(clean, snr)

    half = window_size // 2
    padded = np.pad(noisy, half, mode="edge")
    n = len(noisy)
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n, window_size),
        strides=(padded.strides[0], padded.strides[0]),
    ).copy()

    inp = torch.as_tensor(windows, dtype=torch.float32,
                          device=device).unsqueeze(-1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Process in batches to avoid OOM on large windows
    INFER_BATCH = 512
    parts = []
    with torch.no_grad():
        for i in range(0, len(inp), INFER_BATCH):
            parts.append(model(inp[i:i+INFER_BATCH]).cpu())
    out = torch.cat(parts).numpy().flatten()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # power-normalise
    pw = np.mean(out ** 2)
    if pw > 0:
        out *= np.sqrt(1.0 / pw)

    # second hop
    rx2 = awgn_channel(out, snr)
    rx_bits = (rx2 < 0).astype(int)
    ber, _ = calculate_ber(bits, rx_bits)
    return elapsed, ber


# ── Build models ─────────────────────────────────────────────────────

def build_models():
    models = {}

    # Transformer  (max_len must cover window size)
    models["Transformer"] = TransformerRelay(
        window_size=WINDOW, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_MODEL * 2, dropout=0.1,
    ).to(DEVICE)

    # Mamba S6
    models["Mamba S6"] = MambaRelay(
        window_size=WINDOW, d_model=D_MODEL, d_state=D_STATE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    # Mamba2 (SSD)
    models["Mamba2 (SSD)"] = Mamba2Relay(
        window_size=WINDOW, d_model=D_MODEL, d_state=D_STATE,
        num_layers=NUM_LAYERS, chunk_size=CHUNK_SIZE,
    ).to(DEVICE)

    return models


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  CONTEXT-256 BENCHMARK: Transformer vs Mamba S6 vs Mamba2 (SSD)")
    print(SEP)
    print(f"  Device         : {DEVICE}")
    print(f"  Window size    : {WINDOW}")
    print(f"  d_model        : {D_MODEL}")
    print(f"  d_state        : {D_STATE}")
    print(f"  Layers         : {NUM_LAYERS}")
    print(f"  Mamba2 chunk   : {CHUNK_SIZE}")
    print(f"  Train samples  : {TRAIN_SYMS:,}")
    print(f"  Train epochs   : {TRAIN_EPOCHS}")
    print(f"  Inference bits : {INFER_BITS:,}")
    print(f"  Inference SNR  : {INFER_SNR} dB")
    print(SEP)

    models = build_models()

    # Param count
    print("\n  Model parameters:")
    for name, m in models.items():
        nparams = sum(p.numel() for p in m.parameters())
        print(f"    {name:<20s}  {nparams:>8,}")

    # Prepare training data
    print(f"\n  Generating training data (window={WINDOW}, samples={TRAIN_SYMS}) …")
    X, y = make_training_data(WINDOW, TRAIN_SYMS, TRAIN_SNR, DEVICE)
    print(f"  Training windows: {len(X):,}")

    # ── Warm-up GPU ──────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        print("\n  GPU warm-up …")
        dummy = torch.randn(2, WINDOW, 1, device=DEVICE)
        for m in models.values():
            m.eval()
            with torch.no_grad():
                m(dummy)
        torch.cuda.synchronize()

    # ── Training ─────────────────────────────────────────────────────
    print(f"\n  Training ({TRAIN_EPOCHS} epochs each):")
    train_times = {}
    for name, m in models.items():
        t = train_model(m, X, y, TRAIN_EPOCHS, BATCH_SIZE, DEVICE)
        train_times[name] = t
        print(f"    {name:<20s}  {t:>8.3f}s")

    # ── Inference ────────────────────────────────────────────────────
    print(f"\n  Inference ({INFER_BITS:,} bits, SNR={INFER_SNR} dB):")
    infer_times = {}
    bers = {}
    for name, m in models.items():
        t, ber = infer_ber(m, WINDOW, INFER_BITS, INFER_SNR, DEVICE)
        infer_times[name] = t
        bers[name] = ber
        print(f"    {name:<20s}  {t:>8.4f}s   BER={ber:.6f}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  {'Model':<20s}  {'Params':>8s}  {'Train':>9s}  {'Infer':>9s}  {'BER':>10s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")
    for name, m in models.items():
        nparams = sum(p.numel() for p in m.parameters())
        print(f"  {name:<20s}  {nparams:>8,}  {train_times[name]:>8.3f}s"
              f"  {infer_times[name]:>8.4f}s  {bers[name]:>10.6f}")

    # ── Speed ratios ─────────────────────────────────────────────────
    s6_train = train_times["Mamba S6"]
    s6_infer = infer_times["Mamba S6"]
    print(f"\n  Speed ratios (vs Mamba S6):")
    for name in models:
        tr = train_times[name] / s6_train if s6_train > 0 else float("inf")
        ir = infer_times[name] / s6_infer if s6_infer > 0 else float("inf")
        print(f"    {name:<20s}  train {tr:.2f}×   infer {ir:.2f}×")

    print(f"\n{SEP}")
    print("  Done.")


if __name__ == "__main__":
    main()
