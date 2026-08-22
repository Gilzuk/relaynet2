#!/usr/bin/env python3
"""Constraint-length sweep (K=3,5,7) of the classical coded-DF baseline
on canonical 16-QAM/Rayleigh -- the 16-QAM counterpart of
coded_k_sweep_qpsk.py, using the PAM-4 branch-metric decoder
(relaynet.coding.convolutional_qam16.QAM16CodeDecoder) since 16-QAM's
Gray mapping is not decomposable into independent per-bit soft
observations the way QPSK's is.

Rate 1/2 throughout; uncoded reference reuses the existing
modulation-aware DFHardRelay('qam16'). No learned-relay comparison here
(the MLP/Mamba coded-aware relays were only trained for QPSK; retraining
for every (K, modulation) combination is not attempted -- see
coded_k_sweep_qpsk.py's docstring for the same scoping note).
"""

import json
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder
from relaynet.coding.convolutional_qam16 import QAM16CodeDecoder
from relaynet.relays.coded_df_qam16 import CodedDecodeAndForwardRelayQAM16
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qam import qam16_modulate
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay

from coded_df_experiment import SNRS, N_TRIALS, N_FRAMES

K_VALUES = [3, 5, 7]
FRAME_INFO_BITS = 200  # + tail (2,4,6) is even for K in {3,5,7} -> whole QAM16 symbols


def decode_all_frames_qam16(rx_symbols, decoder, frame_symbols):
    n_frames = len(rx_symbols) // frame_symbols
    info_hats = []
    for f in range(n_frames):
        seg = rx_symbols[f * frame_symbols:(f + 1) * frame_symbols]
        axis_vals = np.empty(2 * frame_symbols, dtype=float)
        axis_vals[0::2] = seg.real
        axis_vals[1::2] = seg.imag
        info_hats.append(decoder.decode(axis_vals))
    return np.concatenate(info_hats)


def run_coded_trial(relay, encoder, decoder, snr_db, seed, frame_symbols):
    rng = np.random.default_rng(seed)
    info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
    coded = np.concatenate([
        encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(N_FRAMES)
    ])
    tx = qam16_modulate(coded)

    rx_relay = rayleigh_fading_channel(tx, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = rayleigh_fading_channel(relay_out, snr_db)

    info_hat = decode_all_frames_qam16(rx_dest, decoder, frame_symbols)
    n = min(len(info_hat), len(info_bits))
    ber = np.mean(info_hat[:n] != info_bits[:n])

    frame_errors = 0
    for f in range(N_FRAMES):
        a = info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
        b = info_hat[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
        if not np.array_equal(a, b):
            frame_errors += 1
    return ber, frame_errors / N_FRAMES


def run_uncoded_trial(snr_db, seed):
    source = Source(seed=seed, modulation="qam16")
    dest = Destination(modulation="qam16")
    relay = DFHardRelay("qam16")

    tx_bits, tx_symbols = source.transmit(N_FRAMES * FRAME_INFO_BITS)
    rx_relay = rayleigh_fading_channel(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = rayleigh_fading_channel(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)

    n = min(len(rx_bits), len(tx_bits))
    return np.mean(rx_bits[:n] != tx_bits[:n])


def main():
    results = {"snr_db": SNRS.tolist(), "qam16_uncoded_df": [], "qam16_k_sweep": {}}

    print("Uncoded 16-QAM DF reference...")
    unc = []
    for snr_db in SNRS:
        ber = np.mean([run_uncoded_trial(snr_db, 7000 * int(snr_db) + t) for t in range(N_TRIALS)])
        unc.append(float(ber))
        print(f"  {snr_db:>5} {ber:.5f}")
    results["qam16_uncoded_df"] = unc

    for K in K_VALUES:
        encoder = ConvolutionalEncoder(constraint_length=K)
        decoder = QAM16CodeDecoder(constraint_length=K)
        relay = CodedDecodeAndForwardRelayQAM16(frame_info_bits=FRAME_INFO_BITS, constraint_length=K)
        frame_symbols = relay.frame_symbols

        print(f"\n=== 16-QAM K={K} (states={decoder.num_states}) ===")
        print(f"{'SNR':>5} {'coded-DF':>12} {'FER':>10}")
        bers, fers = [], []
        for snr_db in SNRS:
            trial_bers, trial_fers = [], []
            for t in range(N_TRIALS):
                ber, fer = run_coded_trial(relay, encoder, decoder, snr_db,
                                            20000 * K + 1000 * int(snr_db) + t, frame_symbols)
                trial_bers.append(ber)
                trial_fers.append(fer)
            mean_ber = float(np.mean(trial_bers))
            mean_fer = float(np.mean(trial_fers))
            bers.append(mean_ber)
            fers.append(mean_fer)
            print(f"{snr_db:>5} {mean_ber:>12.5f} {mean_fer:>10.4f}")

        results["qam16_k_sweep"][f"K{K}"] = {"ber": bers, "fer": fers, "num_states": decoder.num_states}

    with open("results/coded_df_experiment.json") as fh:
        merged = json.load(fh)
    merged.update(results)
    with open("results/coded_df_experiment.json", "w") as fh:
        json.dump(merged, fh, indent=2)
    print("\nMerged into results/coded_df_experiment.json")


if __name__ == "__main__":
    main()
