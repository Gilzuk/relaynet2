#!/usr/bin/env python3
"""Constraint-length sweep (K=3,5,7) of the classical coded-DF baseline
on canonical QPSK/Rayleigh -- characterizes how code strength shifts
the low-SNR threshold effect found with the original K=3 code
(coded_df_experiment.py): coding helps decisively above the threshold
but can actively hurt below it, and a stronger code (larger d_free)
sharpens that cliff rather than removing it.

Rate stays 1/2 throughout (only K varies); reuses the same
ConvolutionalEncoder/ViterbiCodeDecoder/CodedDecodeAndForwardRelay as
the K=3 baseline, now generalized to K in {3, 5, 7} (standard
maximal-free-distance generators -- see relaynet/coding/convolutional.py).
The learned relays (MLP-coded, Mamba-coded) are NOT retrained for every
K: each training run is too expensive to repeat six-fold, so they stay
anchored to the original K=3 config as the representative learned-vs-
classical comparison point; this script only extends the *classical*
characterization.
"""

import json
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import SNRS, N_TRIALS, FRAME_INFO_BITS, N_FRAMES, decode_all_frames

K_VALUES = [3, 5, 7]


def run_coded_trial(relay, encoder, decoder, snr_db, seed, frame_symbols):
    rng = np.random.default_rng(seed)
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
    return ber, frame_errors / N_FRAMES


def main():
    results = {"snr_db": SNRS.tolist(), "k_sweep": {}}

    for K in K_VALUES:
        encoder = ConvolutionalEncoder(constraint_length=K)
        decoder = ViterbiCodeDecoder(constraint_length=K)
        relay = CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS, constraint_length=K)
        frame_symbols = FRAME_INFO_BITS + decoder.num_tail

        print(f"\n=== K={K} (states={decoder.num_states}) ===")
        print(f"{'SNR':>5} {'coded-DF':>12} {'FER':>10}")
        bers, fers = [], []
        for snr_db in SNRS:
            trial_bers, trial_fers = [], []
            for t in range(N_TRIALS):
                ber, fer = run_coded_trial(relay, encoder, decoder, snr_db,
                                            10000 * K + 1000 * int(snr_db) + t, frame_symbols)
                trial_bers.append(ber)
                trial_fers.append(fer)
            mean_ber = float(np.mean(trial_bers))
            mean_fer = float(np.mean(trial_fers))
            bers.append(mean_ber)
            fers.append(mean_fer)
            print(f"{snr_db:>5} {mean_ber:>12.5f} {mean_fer:>10.4f}")

        results["k_sweep"][f"K{K}"] = {"ber": bers, "fer": fers, "num_states": decoder.num_states}

    with open("results/coded_df_experiment.json") as fh:
        merged = json.load(fh)
    merged.update(results)
    with open("results/coded_df_experiment.json", "w") as fh:
        json.dump(merged, fh, indent=2)
    print("\nMerged into results/coded_df_experiment.json")


if __name__ == "__main__":
    main()
