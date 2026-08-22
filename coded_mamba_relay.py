#!/usr/bin/env python3
"""Coded-aware learned relay: Mamba-S6 variant.

Companion to coded_learned_relay.py (the MLP variant). Chapter 5 found
the sequence models gain nothing over the MLP on the canonical
memoryless channel because it "offers no temporal structure for their
inductive bias to exploit." A convolutional code is real temporal
structure -- this is the first setting in the thesis where Mamba-S6 has
a genuine reason to separate from the MLP.

Reuses the actual Mamba-S6 architecture (S6Layer/MambaBlock/MambaRelay
in checkpoints/checkpoint_20_mamba_s6_relay.py) at the same d_model=32,
d_state=16, num_layers=2 configuration used everywhere else in the
thesis (~24k parameters). MambaRelayWrapper's own .train()/.process()
are NOT reused: they are hardcoded to per-axis real classification or
the 16-QAM-only 2-D classifier, neither of which is a QPSK joint 4-class
classifier over coded windows. Training/inference here is written
directly against the underlying MambaRelay module instead, which is
otherwise unmodified.

Window is 21 (same as the MLP variant, ~5xK for the K=3 code), input is
2-channel (I, Q) per window position, output is a 4-class softmax over
the same Gray-coded QPSK alphabet/index convention as
MLPQPSKClassifierRelay so results are directly comparable.
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "checkpoints")
from checkpoint_20_mamba_s6_relay import MambaRelay  # noqa: E402

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.relays.base import Relay

from coded_df_experiment import (
    SNRS, N_TRIALS, FRAME_INFO_BITS, N_FRAMES, decode_all_frames,
)

WINDOW = 21
TRAIN_SNRS = [5, 10, 15]
TRAIN_FRAMES_PER_SNR = 300  # 300 * 202 ~= 60,600 windows/SNR, 3 SNRs ~= 182k total
ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)  # matches MLPQPSKClassifierRelay


def extract_iq_windows(y, window_size):
    pad = window_size // 2
    rx_I = np.pad(y.real, pad, mode="edge")
    rx_Q = np.pad(y.imag, pad, mode="edge")
    n = len(y)
    windows = np.zeros((n, window_size, 2), dtype=np.float32)
    for i in range(n):
        windows[i, :, 0] = rx_I[i:i + window_size]
        windows[i, :, 1] = rx_Q[i:i + window_size]
    return windows


class CodedMambaRelay(Relay):
    """Trained Mamba-S6 4-class QPSK classifier over coded windows."""

    def __init__(self, model, window_size, device, target_power=1.0):
        self.model = model
        self.window_size = window_size
        self.device = device
        self.target_power = target_power

    def process(self, received_signal):
        self.model.eval()
        windows = extract_iq_windows(received_signal, self.window_size)
        with torch.no_grad():
            inp = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
            out = self.model(inp)
            indices = out.argmax(dim=-1).cpu().numpy()
        processed = ALPHABET[indices]
        power = np.mean(np.abs(processed) ** 2)
        if power > 0:
            processed = processed * np.sqrt(self.target_power / power)
        return processed


def generate_coded_training_data(encoder, n_frames_per_snr, window_size, seed=0):
    rng = np.random.default_rng(seed)
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        info = rng.integers(0, 2, n_frames_per_snr * FRAME_INFO_BITS)
        coded = np.concatenate([
            encoder.encode(info[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
            for f in range(n_frames_per_snr)
        ])
        tx = qpsk_modulate(coded)
        rx = rayleigh_fading_channel(tx, snr_db)

        windows = extract_iq_windows(rx, window_size)
        bit_pairs = coded.reshape(-1, 2)
        target_idx = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]

        X_list.append(windows)
        T_list.append(target_idx)
    return np.vstack(X_list), np.concatenate(T_list)


def train_mamba(X, T, epochs=15, batch_size=4096, lr=1e-3, device="cpu"):
    model = MambaRelay(window_size=WINDOW, d_model=32, d_state=16, num_layers=2,
                        in_channels=2, num_classes=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba-S6-coded relay: window={WINDOW}, params={n_params}")

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(X_t[idx])
            loss = crit(out, T_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}")
    return model


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    print("Generating coded training data...")
    X, T = generate_coded_training_data(encoder, TRAIN_FRAMES_PER_SNR, WINDOW)
    print(f"Training set: {X.shape[0]} windows (device={device})")

    print("Training...")
    model = train_mamba(X, T, device=device)
    relay = CodedMambaRelay(model, WINDOW, device)

    print(f"\n{'SNR':>5} {'Mamba-coded':>13} {'Mamba-coded-FER':>16}")
    results = {"mamba_coded": [], "mamba_coded_fer": []}
    for snr_db in SNRS:
        trial_bers, trial_fers = [], []
        for t in range(N_TRIALS):
            rng = np.random.default_rng(6000 * int(snr_db) + t)
            info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
            coded = np.concatenate([
                encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
                for f in range(N_FRAMES)
            ])
            tx = qpsk_modulate(coded)
            rx_relay = rayleigh_fading_channel(tx, snr_db)
            relay_out = relay.process(rx_relay)
            rx_dest = rayleigh_fading_channel(relay_out, snr_db)

            info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)
            n = min(len(info_hat), len(info_bits))
            ber = np.mean(info_hat[:n] != info_bits[:n])

            frame_errors = 0
            for f in range(N_FRAMES):
                a = info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
                b = info_hat[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
                if not np.array_equal(a, b):
                    frame_errors += 1
            trial_bers.append(ber)
            trial_fers.append(frame_errors / N_FRAMES)

        mean_ber = float(np.mean(trial_bers))
        mean_fer = float(np.mean(trial_fers))
        results["mamba_coded"].append(mean_ber)
        results["mamba_coded_fer"].append(mean_fer)
        print(f"{snr_db:>5} {mean_ber:>13.5f} {mean_fer:>16.4f}")

    with open("results/coded_df_experiment.json") as fh:
        merged = json.load(fh)
    merged.update(results)
    with open("results/coded_df_experiment.json", "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nMerged into results/coded_df_experiment.json  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
