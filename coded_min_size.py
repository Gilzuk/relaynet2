"""Minimum learned relay that approaches block (coded) DF.

Different scenario from the rest of the minimum-size study, and the question
is posed differently because the incumbent already loses.

Everywhere else the learned relay matches its classical comparator and the
question is how far it can be shrunk. Here it does not: the published
756-parameter relay (MLPQPSKClassifierRelay, window 21, hidden 16) trails
block DF by +33% to +58% across 4-12 dB and only wins at 20 dB
(results/coded_df_experiment.json). So "the minimum size that matches" has no
answer, and the useful question is the one actually asked: how close can a
relay of a given size get, and at what size does adding parameters stop
buying anything.

The deliverable is therefore an approach curve, not a threshold.

WHAT BLOCK DF IS. Not the symbol-wise slicer used elsewhere in this study.
CodedDecodeAndForwardRelay Viterbi-decodes an entire rate-1/2 convolutional
frame, re-encodes and re-modulates, so it forwards a clean codeword. It is a
far stronger baseline than hard slicing, which is why the learned relay loses
to it.

PARAMETER COUNT differs from the regression relays, because this relay is a
4-class classifier over a complex window (I and Q concatenated):

    params = 2*W*H + H + 4*H + 4

so window 21, hidden 16 gives 42*16 + 16 + 64 + 4 = 756.

TWO SNR REGIONS, reported separately. Below the code threshold the frame
error rate is 1.0 for every relay and coded DF is actually *worse* than
uncoded DF (0.4204 vs 0.3337 at 0 dB): decoding failures scramble bits that
an uncoded slicer would have got right. BER there says little about relay
quality. The headline figure is therefore the worst-case penalty over the
operational region, where coded DF's FER is below 0.99, and the all-SNR
figure is printed next to it rather than replaced by it.

FER IS REPORTED ALONGSIDE BER because the two disagree in an interesting
way at high SNR: the published relay has the *better* FER at 16 and 20 dB
(0.419 vs 0.453, 0.145 vs 0.200) while having the worse BER at 16 dB. It
breaks fewer frames but scrambles the ones it breaks more thoroughly. A BER
only view hides that.

BUDGET. Reduced from the published experiment: 10 trials x 100 frames x 200
info bits = 200k info bits per SNR point, against 100 x 500 x 200 = 10M
there. At 20 dB, where coded DF sits near 0.00137, that is still ~270 errors
per point. Coarser than the published run and stated as such; the incumbent
(21, 16) is re-measured here at the same reduced budget so the comparison is
like for like rather than against the published number.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coded_df_experiment as CE
import coded_learned_relay as CL
from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 10
N_FRAMES = 100
TRAIN_FRAMES_PER_SNR = 800
EPOCHS = 25
BATCH = 512
LR = 3e-3
TRAIN_SEEDS = [0, 1, 2]
OPERATIONAL_FER = 0.99      # below this, the code is actually working

# (window, hidden). (21, 16) is the published relay, the control.
GRID = [
    (1, 2), (1, 8), (1, 16),
    (3, 2), (3, 8), (3, 16),
    (5, 2), (5, 8), (5, 16),
    (9, 2), (9, 8), (9, 16),
    (15, 2), (15, 8), (15, 16),
    (21, 2), (21, 8), (21, 16),        # (21,16) = MLP-756, the control
    (31, 8), (31, 16),
]


def n_params(window, hidden):
    return 2 * window * hidden + hidden + 4 * hidden + 4


def evaluate(relay, encoder, decoder, frame_symbols, tag=None):
    """BER and FER per SNR, from a fixed seed sequence so every relay sees
    the same fading, noise and payload."""
    ber, fer = [], []
    for snr in SNRS:
        b = [CE.run_coded_trial(relay, snr, 5000 + i, encoder, decoder, frame_symbols)
             for i in range(N_TRIALS)]
        ber.append(float(np.mean([x[0] for x in b])))
        fer.append(float(np.mean([x[1] for x in b])))
    if tag:
        print(f"    {tag:<26} BER " + "  ".join(f"{x:.5f}" for x in ber))
        print(f"    {'':<26} FER " + "  ".join(f"{x:.5f}" for x in fer))
    return np.asarray(ber), np.asarray(fer)


def main():
    CE.N_FRAMES = N_FRAMES
    encoder, decoder = ConvolutionalEncoder(), ViterbiCodeDecoder()
    frame_symbols = CE.FRAME_INFO_BITS + decoder.num_tail

    print("Minimum learned relay approaching block (coded) DF")
    print(f"SNRs {SNRS} | {N_TRIALS} trials x {N_FRAMES} frames x "
          f"{CE.FRAME_INFO_BITS} info bits | {len(GRID)} configs x "
          f"{len(TRAIN_SEEDS)} inits")
    print(f"params = 2*W*H + H + 4*H + 4   (4-class classifier on I/Q windows)\n")

    print("  classical baseline")
    df = CodedDecodeAndForwardRelay(frame_info_bits=CE.FRAME_INFO_BITS)
    df_ber, df_fer = evaluate(df, encoder, decoder, frame_symbols,
                              "block DF (0 params)")
    usable = [i for i, f in enumerate(df_fer) if f < OPERATIONAL_FER]
    print(f"    operational region (coded-DF FER < {OPERATIONAL_FER}): "
          f"{[SNRS[i] for i in usable] or 'none'} dB")

    print("\n  sweep")
    rows = []
    for window, hidden in GRID:
        p = n_params(window, hidden)
        t0 = time.time()
        runs = []
        for ts in TRAIN_SEEDS:
            relay = MLPQPSKClassifierRelay(window_size=window,
                                           hidden_size=hidden, seed=ts)
            X, T = CL.generate_coded_training_data(
                encoder, relay, TRAIN_FRAMES_PER_SNR, seed=ts)
            relay.train_on_data(X, T, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            ber, fer = evaluate(relay, encoder, decoder, frame_symbols)
            rel_all = [(ber[i] - df_ber[i]) / df_ber[i] for i in range(len(SNRS))]
            rel_op = [rel_all[i] for i in usable]
            runs.append({
                "train_seed": ts,
                "ber": [float(x) for x in ber], "fer": [float(x) for x in fer],
                "worst_rel_all": float(max(rel_all)),
                "worst_rel_operational": float(max(rel_op)) if rel_op else float("nan"),
                "beats_df_fer_at": [SNRS[i] for i in range(len(SNRS))
                                    if fer[i] < df_fer[i]],
            })
        w_op = max(r["worst_rel_operational"] for r in runs)
        b_op = min(r["worst_rel_operational"] for r in runs)
        w_all = max(r["worst_rel_all"] for r in runs)
        rows.append({
            "window": window, "hidden": hidden, "params": p,
            "worst_rel_operational": float(w_op),
            "best_rel_operational": float(b_op),
            "worst_rel_all": float(w_all),
            "seed_runs": runs,
        })
        ctrl = "  <- published MLP-756" if (window, hidden) == (21, 16) else ""
        print(f"    w={window:<3} h={hidden:<3} {p:>5}p   "
              f"vs block DF, operational {100*b_op:+7.1f}% .. {100*w_op:+7.1f}%"
              f"   all-SNR {100*w_all:+7.1f}%   [{time.time()-t0:.0f}s]{ctrl}",
              flush=True)

        out = {
            "snr_db": SNRS, "n_trials": N_TRIALS, "n_frames": N_FRAMES,
            "frame_info_bits": CE.FRAME_INFO_BITS,
            "train_seeds": TRAIN_SEEDS, "operational_fer": OPERATIONAL_FER,
            "operational_snrs": [SNRS[i] for i in usable],
            "df_ber": [float(x) for x in df_ber],
            "df_fer": [float(x) for x in df_fer],
            "sweep": rows,
        }
        with open("results/coded_min_size.json", "w") as fh:
            json.dump(out, fh, indent=2)

    print("\n" + "=" * 78)
    ctrl = [r for r in rows if (r["window"], r["hidden"]) == (21, 16)]
    if ctrl:
        c = ctrl[0]
        print(f"  published MLP-756 at this budget: "
              f"{100*c['worst_rel_operational']:+.1f}% vs block DF (operational)")
        within = [r for r in rows
                  if r["worst_rel_operational"] <= c["worst_rel_operational"]]
        if within:
            b = min(within, key=lambda r: r["params"])
            print(f"  smallest relay at least as close as MLP-756: {b['params']} "
                  f"params (w={b['window']} h={b['hidden']}) -- "
                  f"{756 / b['params']:.1f}x smaller")
    print(f"  saved results/coded_min_size.json")


if __name__ == "__main__":
    main()
