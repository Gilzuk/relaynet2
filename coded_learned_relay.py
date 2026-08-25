#!/usr/bin/env python3
"""Coded-aware learned relay: does a learned relay do better with real
temporal structure (a convolutional code) to exploit, than it does on
the canonical memoryless channel where Chapter 5 found the sequence
models gain nothing ("the memoryless channel simply offers no temporal
structure for their inductive bias to exploit")?

Reuses ``MLPQPSKClassifierRelay`` unchanged (it already predicts a
per-window QPSK class, matching the coded pipeline's alphabet exactly),
trained on windows drawn from CODED transmissions rather than i.i.d.
QPSK symbols, with a window wide enough (21 symbols, roughly 5xK for the
K=3 code) to span multiple trellis steps. This keeps it plug-compatible
with the same evaluation harness used for AF / uncoded-DF / coded-DF in
coded_df_experiment.py -- the relay is not told about the code, it is
only ever shown noisy windows during training, exactly like the existing
relays are only ever shown noisy windows on the canonical channel.
"""

import json
import numpy as np

from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import (
    SNRS, N_TRIALS, FRAME_INFO_BITS, N_FRAMES, decode_all_frames,
)

WINDOW = 21
TRAIN_SNRS = [5, 10, 15]
TRAIN_FRAMES_PER_SNR = 2000  # 2000 * 202 ~= 404,000 symbols/SNR, 3 SNRs


def generate_coded_training_data(encoder, relay, n_frames_per_snr, seed=0):
    rng = np.random.default_rng(seed)
    # rayleigh_fading_channel draws from the global RNG, so seeding only the
    # bit generator above would leave the fading and noise irreproducible.
    # Same convention as coded_reliable_regime.py / coded_error_mechanism.py.
    np.random.seed(seed % (2 ** 31))
    frame_symbols = FRAME_INFO_BITS + encoder.num_tail
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        info = rng.integers(0, 2, n_frames_per_snr * FRAME_INFO_BITS)
        coded = np.concatenate([
            encoder.encode(info[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
            for f in range(n_frames_per_snr)
        ])
        tx = qpsk_modulate(coded)
        rx = rayleigh_fading_channel(tx, snr_db)

        windows = relay._extract_windows(rx)
        bit_pairs = coded.reshape(-1, 2)
        target_idx = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]
        assert len(target_idx) == frame_symbols * n_frames_per_snr

        X_list.append(windows)
        T_list.append(target_idx)
    return np.vstack(X_list), np.concatenate(T_list)


def main():
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    relay = MLPQPSKClassifierRelay(window_size=WINDOW, hidden_size=16, seed=0)
    print(f"MLP-coded relay: window={WINDOW}, params={relay.W1.size + relay.b1.size + relay.W2.size + relay.b2.size}")

    print("Generating coded training data...")
    X, T = generate_coded_training_data(encoder, relay, TRAIN_FRAMES_PER_SNR)
    print(f"Training set: {X.shape[0]} windows")

    print("Training...")
    relay.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)

    print(f"\n{'SNR':>5} {'MLP-coded':>12} {'MLP-coded-FER':>15}")
    results = {"snr_db": SNRS.tolist(), "mlp_coded": [], "mlp_coded_fer": []}
    for snr_db in SNRS:
        trial_bers, trial_fers = [], []
        for t in range(N_TRIALS):
            seed = 5000 * int(snr_db) + t
            rng = np.random.default_rng(seed)
            # rayleigh_fading_channel draws from the global RNG; see the note
            # in generate_coded_training_data above.
            np.random.seed(seed % (2 ** 31))
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
        results["mlp_coded"].append(mean_ber)
        results["mlp_coded_fer"].append(mean_fer)
        print(f"{snr_db:>5} {mean_ber:>12.5f} {mean_fer:>15.4f}")

    with open("results/coded_df_experiment.json") as fh:
        merged = json.load(fh)
    merged.update(results)
    with open("results/coded_df_experiment.json", "w") as fh:
        json.dump(merged, fh, indent=2)
    print("\nMerged into results/coded_df_experiment.json")


if __name__ == "__main__":
    main()
