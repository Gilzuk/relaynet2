#!/usr/bin/env python3
"""Is the learned relay's apparent high-SNR edge over Viterbi real, or noise?

At 16 and 20 dB the 10-trial sweep in coded_df_experiment.py showed the
coded-aware learned relays a hair *below* classical coded-DF (Viterbi):
0.0046/0.0042 vs 0.0043 at 16 dB, 0.0011/0.0009 vs 0.0013 at 20 dB.
Those gaps are 1e-4-ish at low absolute BER on only 10 trials, and the
thesis text calls them "within likely trial-to-trial noise" rather than
a win. This script tests that call properly instead of leaving it as a
judgement.

Viterbi is the maximum-likelihood *sequence* decoder for a known code on
a known channel, so the prior is strongly that it cannot be beaten here
-- the one documented mechanism in this thesis by which a learned relay
*can* edge it out (Section~sec:qpsk-unknown-channel: sequence-ML is not
bit-ML for a Gray-coded multi-bit branch) is a real effect but was
measured on an unknown-ISI channel, not this one. So this is a genuine
two-sided question, not a hunt for a win.

Design: **paired** trials. Every relay sees the identical information
bits and the identical channel realizations within a trial (seeded per
trial, channel noise re-drawn identically via a per-trial seed reset),
so the per-trial difference isolates the relay and removes the
channel-draw variance that dominates an unpaired comparison. Wilcoxon
signed-rank on the paired per-trial differences, the same test used
elsewhere in this thesis. Per-trial arrays are persisted this time
(the first sweep saved only trial means).
"""

import json
import time

import numpy as np
import torch
from scipy import stats

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import FRAME_INFO_BITS, N_FRAMES, decode_all_frames
from coded_learned_relay import (
    WINDOW as MLP_WINDOW,
    generate_coded_training_data as gen_mlp_data,
    TRAIN_FRAMES_PER_SNR as MLP_TRAIN_FRAMES,
)
from coded_mamba_relay import (
    WINDOW as MAMBA_WINDOW,
    CodedMambaRelay,
    generate_coded_training_data as gen_mamba_data,
    train_mamba,
    TRAIN_FRAMES_PER_SNR as MAMBA_TRAIN_FRAMES,
)

SNRS = [16, 20]
N_TRIALS = 100  # 10x the original budget at these two points


def run_one_trial(relays, encoder, decoder, frame_symbols, snr_db, trial_seed):
    """Evaluate every relay on the SAME bits and SAME channel draws."""
    rng = np.random.default_rng(trial_seed)
    info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
    coded = np.concatenate([
        encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(N_FRAMES)
    ])
    tx = qpsk_modulate(coded)

    out = {}
    for name, relay in relays.items():
        # Reset the legacy global RNG that rayleigh_fading_channel draws from,
        # identically for every relay, so all relays face the same hop-1 and
        # hop-2 fading/noise realizations within this trial.
        np.random.seed(trial_seed % (2 ** 31))
        rx_relay = rayleigh_fading_channel(tx, snr_db)
        relay_out = relay.process(rx_relay)
        rx_dest = rayleigh_fading_channel(relay_out, snr_db)

        info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)
        n = min(len(info_hat), len(info_bits))
        out[name] = float(np.mean(info_hat[:n] != info_bits[:n]))
    return out


def main():
    t0 = time.time()
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    print("Training MLP-coded ...", flush=True)
    mlp = MLPQPSKClassifierRelay(window_size=MLP_WINDOW, hidden_size=16, seed=0)
    X, T = gen_mlp_data(encoder, mlp, MLP_TRAIN_FRAMES)
    mlp.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)

    print("Training Mamba-coded ...", flush=True)
    torch.manual_seed(0)
    Xm, Tm = gen_mamba_data(encoder, MAMBA_TRAIN_FRAMES, MAMBA_WINDOW)
    model = train_mamba(Xm, Tm, device="cpu")
    mamba = CodedMambaRelay(model, MAMBA_WINDOW, "cpu")

    relays = {
        "coded_df": CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS),
        "mlp_coded": mlp,
        "mamba_coded": mamba,
    }

    results = {"snr_db": SNRS, "n_trials": N_TRIALS, "per_trial": {}, "summary": {}}

    for snr_db in SNRS:
        per_trial = {k: [] for k in relays}
        print(f"\n=== {snr_db} dB, {N_TRIALS} paired trials ===", flush=True)
        for t in range(N_TRIALS):
            got = run_one_trial(relays, encoder, decoder, frame_symbols,
                                snr_db, 900000 + 1000 * snr_db + t)
            for k, v in got.items():
                per_trial[k].append(v)
            if (t + 1) % 10 == 0:
                print(f"  {t+1}/{N_TRIALS} trials  "
                      + "  ".join(f"{k}={np.mean(per_trial[k]):.5f}" for k in relays),
                      flush=True)

        results["per_trial"][str(snr_db)] = {k: v for k, v in per_trial.items()}

        summ = {}
        for k, v in per_trial.items():
            a = np.asarray(v)
            summ[k] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)),
                "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
            }

        # Paired comparisons against the classical decoder.
        vit = np.asarray(per_trial["coded_df"])
        for k in ("mlp_coded", "mamba_coded"):
            arr = np.asarray(per_trial[k])
            diff = arr - vit  # negative => learned relay better
            try:
                w_stat, p_val = stats.wilcoxon(arr, vit)
            except ValueError:  # all-zero differences
                w_stat, p_val = float("nan"), 1.0
            summ[k]["vs_viterbi"] = {
                "mean_diff": float(diff.mean()),
                "ci95_diff": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                "wins": int((diff < 0).sum()),
                "losses": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "wilcoxon_p": float(p_val),
            }
        results["summary"][str(snr_db)] = summ

        print(f"\n  {snr_db} dB summary ({N_TRIALS} paired trials):")
        for k in relays:
            s = summ[k]
            print(f"    {k:14s} {s['mean']:.6f} +/- {s['ci95']:.6f}")
        for k in ("mlp_coded", "mamba_coded"):
            c = summ[k]["vs_viterbi"]
            verdict = ("learned better" if c["mean_diff"] < 0 else "Viterbi better")
            sig = "SIGNIFICANT" if c["wilcoxon_p"] < 0.05 else "not significant"
            print(f"    {k} vs Viterbi: diff={c['mean_diff']:+.6f} "
                  f"+/-{c['ci95_diff']:.6f}  W/L/T={c['wins']}/{c['losses']}/{c['ties']}  "
                  f"p={c['wilcoxon_p']:.4f}  -> {verdict}, {sig}")

    with open("results/coded_high_budget_test.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved results/coded_high_budget_test.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
