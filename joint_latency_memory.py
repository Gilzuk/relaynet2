#!/usr/bin/env python3
"""Memory channel *under* a latency constraint: the joint measurement.

The thesis answers its two research objectives separately. Objective 1
asks, with no budget at all, in which regime a learned relay can replace
the classical method on a channel with memory; Objective 2 asks the same
question when the relay must also fit a latency and complexity budget.
Neither existing experiment puts both constraints on the same axis, so
the distance between the two answers has never been measured.

This script measures it. Every scheme is placed inside one identical
coded system -- rate-1/2, K=3 convolutional code, QPSK, 200 information
bits per frame, soft Viterbi decoding at the *destination* -- so all of
them deliver an information BER at the same information rate and differ
only in what the relay does. Each relay is then labelled with the
structural decision delay it imposes, in symbols:

    amplify-and-forward                       0
    symbol-wise hard decode-and-forward       0
    MLP-QPSK classifier, window w             w // 2
    MLSE with traceback depth D               D
    block DF (decode + re-encode a frame)     frame_symbols (202)

Part A sweeps the latency axis on a 3-tap channel. Part B holds the
latency budget fixed and grows the channel memory, so the M^(L-1) state
count of MLSE moves while the relay network stays the same size -- the
compute boundary rather than the latency boundary.

Channel: hop 1 carries the unknown L-tap ISI response (geometric decay
h_k = 0.7^k, unit-energy normalized) plus complex AWGN; hop 2 is AWGN.
SNR is the thesis convention throughout, gamma = 10^(SNR_dB/10).
"""

import json
import time
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays import MLPQPSKClassifierRelay, TruncatedViterbiQPSKRelay
from relaynet.channels.e6_channels import ComplexISIChannel, ComplexAWGNChannel
from relaynet.modulation.qpsk import qpsk_modulate
from e6_sim_enhanced_multimod import DFHardRelay

FRAME_INFO_BITS = 200
N_FRAMES = 150                      # 30,000 info bits per trial
SEEDS = (0, 1, 2)
SNRS_A = (0, 4, 8, 12, 16, 20)
SNR_B = 12
TRAIN_SNRS = (5, 10, 15)
N_TRAIN = 120_000
MLP_HIDDEN = 8

MLP_WINDOWS_A = (1, 3, 5, 11, 21)
TRACEBACKS_A = (0, 1, 2, 3, 5, 15)


def taps_for(L):
    h = np.array([0.7 ** k for k in range(L)])
    return h / np.linalg.norm(h)


class ComposedRelay:
    """Equalize first, then run a second relay on the equalized symbols."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def process(self, y):
        return self.second.process(self.first.process(y))


def train_mlp(L, window, seed):
    """Train the 4-class QPSK MLP on hop 1 of this channel, at three SNRs."""
    mlp = MLPQPSKClassifierRelay(window_size=window, hidden_size=MLP_HIDDEN, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    per_snr = N_TRAIN // len(TRAIN_SNRS)
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        idx = rng.integers(0, 4, per_snr)
        x = MLPQPSKClassifierRelay.ALPHABET[idx]
        y = ComplexISIChannel(taps_for(L), seed=2000 + seed)(x, snr_db)
        X_list.append(mlp._extract_windows(y))
        T_list.append(idx)
    mlp.train_on_data(np.vstack(X_list), np.concatenate(T_list),
                      epochs=25, batch_size=256, lr=3e-3)
    return mlp


def run_trial(relay, snr_db, seed, L, encoder, decoder, frame_symbols):
    """Encode -> ISI hop 1 -> relay -> AWGN hop 2 -> destination decode."""
    rng = np.random.default_rng(seed)
    info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)

    coded = np.concatenate([
        encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(N_FRAMES)])
    tx = qpsk_modulate(coded)

    hop1 = ComplexISIChannel(taps_for(L), seed=10_000 + seed)
    hop2 = ComplexAWGNChannel(seed=20_000 + seed)

    rx = hop2(relay.process(hop1(tx, snr_db)), snr_db)

    n_frames = len(rx) // frame_symbols
    info_hat = []
    for f in range(n_frames):
        seg = rx[f * frame_symbols:(f + 1) * frame_symbols]
        soft = np.empty(2 * frame_symbols, dtype=float)
        soft[0::2] = seg.real
        soft[1::2] = seg.imag
        info_hat.append(decoder.decode(soft))
    info_hat = np.concatenate(info_hat)

    n = min(len(info_hat), len(info_bits))
    ber = float(np.mean(info_hat[:n] != info_bits[:n]))
    frame_errors = sum(
        not np.array_equal(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS],
                           info_hat[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(n_frames))
    return ber, frame_errors / n_frames


def build_schemes(L, frame_symbols, mlp_cache):
    """Return [(label, latency_symbols, n_states_or_None, relay_factory)]."""
    taps = taps_for(L)
    schemes = [
        ("AF", 0, None, lambda seed: AmplifyAndForwardRelay()),
        ("DF-hard (symbol-wise)", 0, None, lambda seed: DFHardRelay("qpsk")),
    ]
    for w in MLP_WINDOWS_A:
        schemes.append((f"MLP w={w}", w // 2, None,
                        (lambda w: lambda seed: mlp_cache[(L, w, seed)])(w)))
    for D in TRACEBACKS_A:
        schemes.append((f"MLSE D={D}", D, 4 ** (L - 1),
                        (lambda D: lambda seed: TruncatedViterbiQPSKRelay(
                            channel_taps=taps, traceback=D))(D)))
    schemes.append(("block DF", frame_symbols, None,
                    lambda seed: CodedDecodeAndForwardRelay(
                        frame_info_bits=FRAME_INFO_BITS)))
    schemes.append(("block DF + MLSE", frame_symbols, 4 ** (L - 1),
                    lambda seed: ComposedRelay(
                        TruncatedViterbiQPSKRelay(channel_taps=taps, traceback=5 * L),
                        CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS))))
    return schemes


def measure_compute(relay, n=20_000, repeats=3):
    """Median wall-clock microseconds per symbol for one relay."""
    rng = np.random.default_rng(7)
    y = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        relay.process(y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / n * 1e6


def main():
    encoder = ConvolutionalEncoder(constraint_length=3)
    decoder = ViterbiCodeDecoder(constraint_length=3)
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    out = {"frame_info_bits": FRAME_INFO_BITS, "n_frames": N_FRAMES,
           "frame_symbols": frame_symbols, "seeds": list(SEEDS),
           "mlp_hidden": MLP_HIDDEN, "train_snrs": list(TRAIN_SNRS)}

    mlp_cache = {}
    for L, windows in ((3, MLP_WINDOWS_A), (5, (11,)), (7, (11,))):
        for w in windows:
            for seed in SEEDS:
                print(f"training MLP  L={L} w={w} seed={seed}", flush=True)
                mlp_cache[(L, w, seed)] = train_mlp(L, w, seed)
    out["mlp_params"] = {f"w={w}": mlp_cache[(3, w, 0)].n_params()
                         for w in MLP_WINDOWS_A}

    # ---- Part A: the latency axis at L = 3 ----
    L = 3
    part_a = []
    for label, latency, states, factory in build_schemes(L, frame_symbols, mlp_cache):
        row = {"scheme": label, "latency_symbols": latency, "states": states,
               "snr_db": list(SNRS_A), "ber": [], "ber_std": [], "fer": []}
        for snr in SNRS_A:
            bers, fers = [], []
            for seed in SEEDS:
                b, f = run_trial(factory(seed), snr, seed, L,
                                 encoder, decoder, frame_symbols)
                bers.append(b)
                fers.append(f)
            row["ber"].append(float(np.mean(bers)))
            row["ber_std"].append(float(np.std(bers)))
            row["fer"].append(float(np.mean(fers)))
            print(f"A  {label:24s} lat={latency:4d}  SNR={snr:2d}  "
                  f"BER={np.mean(bers):.6f} (sd {np.std(bers):.6f})  "
                  f"FER={np.mean(fers):.4f}", flush=True)
        part_a.append(row)
    out["part_a"] = {"channel_taps_L": L, "rows": part_a}

    # ---- Part B: the compute axis, latency held near-fixed ----
    part_b = []
    for Lm in (3, 5, 7):
        taps = taps_for(Lm)
        cases = [
            ("MLP w=11", 5, None, lambda seed, Lm=Lm: mlp_cache[(Lm, 11, seed)]),
            (f"MLSE D={5*Lm}", 5 * Lm, 4 ** (Lm - 1),
             lambda seed, taps=taps, Lm=Lm: TruncatedViterbiQPSKRelay(
                 channel_taps=taps, traceback=5 * Lm)),
            ("block DF", frame_symbols, None,
             lambda seed: CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)),
        ]
        for label, latency, states, factory in cases:
            bers = [run_trial(factory(seed), SNR_B, seed, Lm,
                              encoder, decoder, frame_symbols)[0] for seed in SEEDS]
            us = measure_compute(factory(SEEDS[0]))
            part_b.append({"channel_taps_L": Lm, "scheme": label,
                           "latency_symbols": latency, "states": states,
                           "snr_db": SNR_B, "ber": float(np.mean(bers)),
                           "ber_std": float(np.std(bers)),
                           "us_per_symbol": us})
            print(f"B  L={Lm}  {label:14s} lat={latency:4d} states={states}  "
                  f"BER={np.mean(bers):.6f}  {us:.3f} us/sym", flush=True)
    out["part_b"] = {"snr_db": SNR_B, "rows": part_b}

    with open("results/joint_latency_memory.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/joint_latency_memory.json")


if __name__ == "__main__":
    main()
