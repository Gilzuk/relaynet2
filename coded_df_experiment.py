#!/usr/bin/env python3
"""Coded block-DF baseline on the canonical QPSK/Rayleigh channel.

Closes the caveat stated in Chapter 1's DF-terminology remark and Chapter
8's future-work list ("the reported DF results should not be read as
bounds on coded block-DF performance") by actually measuring it: a
rate-1/2, K=3 convolutional code, soft-decision Viterbi-decoded, used as
a genuine information-theoretic block-DF relay
(``CodedDecodeAndForwardRelay``), compared against the existing
uncoded symbol-wise DF baseline and coded AF (relay does not decode, so
the code's gain is visible without any relay intelligence).

Channel: canonical i.i.d. Rayleigh fast fading, both hops, no ISI --
the same channel as Table~tbl:table2. SNR axis is per-channel-use
Es/N0 (the physical channel does not know about the outer code); the
rate-1/2 code's ~3 dB Eb/N0 penalty is reported separately, the same
convention already used for the QPSK-vs-BPSK Eb/N0 note in Chapter 5.
"""

import json
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.nodes import Source, Destination
from relaynet.modulation.qpsk import qpsk_modulate
from e6_sim_enhanced_multimod import DFHardRelay

SNRS = np.arange(0, 21, 4)
N_TRIALS = 10
FRAME_INFO_BITS = 200
N_FRAMES = 500  # 500 * 200 = 100,000 info bits/trial, matching the thesis-standard scale
MODULATION = "qpsk"


def decode_all_frames(rx_symbols, decoder, frame_symbols):
    n_frames = len(rx_symbols) // frame_symbols
    info_hats = []
    for f in range(n_frames):
        seg = rx_symbols[f * frame_symbols:(f + 1) * frame_symbols]
        soft = np.empty(2 * frame_symbols, dtype=float)
        soft[0::2] = seg.real
        soft[1::2] = seg.imag
        info_hats.append(decoder.decode(soft))
    return np.concatenate(info_hats)


def run_coded_trial(relay, snr_db, seed, encoder, decoder, frame_symbols):
    """One trial of coded transmission: encode -> hop1 -> relay -> hop2 -> decode."""
    rng = np.random.default_rng(seed)
    info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)

    coded_list = [encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
                  for f in range(N_FRAMES)]
    coded_bits = np.concatenate(coded_list)
    tx_symbols = qpsk_modulate(coded_bits)

    rx_relay = rayleigh_fading_channel(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = rayleigh_fading_channel(relay_out, snr_db)

    info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)
    n = min(len(info_hat), len(info_bits))
    errors = np.sum(info_hat[:n] != info_bits[:n])
    frame_errors = 0
    for f in range(N_FRAMES):
        a = info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
        b = info_hat[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
        if not np.array_equal(a, b):
            frame_errors += 1
    return errors / n, frame_errors / N_FRAMES


def run_uncoded_df_trial(snr_db, seed):
    """Reference: no code, existing modulation-aware symbol-wise DF."""
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    relay = DFHardRelay(MODULATION)

    tx_bits, tx_symbols = source.transmit(N_FRAMES * FRAME_INFO_BITS)
    rx_relay = rayleigh_fading_channel(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = rayleigh_fading_channel(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)

    n = min(len(rx_bits), len(tx_bits))
    return np.mean(rx_bits[:n] != tx_bits[:n])


def main():
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    coded_df_relay = CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    af_relay = AmplifyAndForwardRelay()

    results = {"snr_db": SNRS.tolist(), "uncoded_df": [], "coded_af": [], "coded_df": [],
               "coded_af_fer": [], "coded_df_fer": []}

    print(f"{'SNR':>5} {'uncoded-DF':>12} {'coded-AF':>12} {'coded-DF':>12} {'AF-FER':>10} {'DF-FER':>10}")
    for snr_db in SNRS:
        unc = [run_uncoded_df_trial(snr_db, seed=1000 * int(snr_db) + t) for t in range(N_TRIALS)]
        af_trials = [run_coded_trial(af_relay, snr_db, 2000 * int(snr_db) + t, encoder, decoder, frame_symbols)
                     for t in range(N_TRIALS)]
        df_trials = [run_coded_trial(coded_df_relay, snr_db, 3000 * int(snr_db) + t, encoder, decoder, frame_symbols)
                     for t in range(N_TRIALS)]

        af_ber = np.mean([x[0] for x in af_trials])
        af_fer = np.mean([x[1] for x in af_trials])
        df_ber = np.mean([x[0] for x in df_trials])
        df_fer = np.mean([x[1] for x in df_trials])
        unc_ber = np.mean(unc)

        results["uncoded_df"].append(float(unc_ber))
        results["coded_af"].append(float(af_ber))
        results["coded_df"].append(float(df_ber))
        results["coded_af_fer"].append(float(af_fer))
        results["coded_df_fer"].append(float(df_fer))

        print(f"{snr_db:>5} {unc_ber:>12.5f} {af_ber:>12.5f} {df_ber:>12.5f} {af_fer:>10.4f} {df_fer:>10.4f}")

    with open("results/coded_df_experiment.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved results/coded_df_experiment.json")


if __name__ == "__main__":
    main()
