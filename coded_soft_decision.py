"""Soft-decision relaying on the coded two-hop link.

The hard relays each waste something. Hard block-DF decodes with the code's
redundancy but then re-encodes, so a wrong decode leaves a valid-but-wrong
codeword the destination cannot repair. The hard learned relay keeps errors
repairable but throws away its own confidence. This script adds the two
soft relays that keep what each was discarding, and measures all of them
against each other on the same channel draws:

  coded-DF (hard)   Viterbi at the relay, re-encode, forward       [existing]
  soft-DF (BCJR)    BCJR at the relay, forward posterior means     [new]
  MLP hard          per-symbol argmax over the trained classifier  [existing]
  MLP soft          posterior mean over the SAME trained weights   [new]
  oracle            forwards the clean codeword (hop-2-only floor) [control]

The MLP soft/hard pair shares one set of weights and differs only in the
read-out rule, so any gap between them isolates the decision rule itself
rather than training.

Paired trials throughout: within a trial every relay sees identical
information bits and identical hop-1/hop-2 realizations, so per-trial
differences remove channel-draw variance. Wilcoxon signed-rank on those
differences, matching the test used elsewhere in this thesis.
"""

import json
import time

import numpy as np
from scipy import stats

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay, SoftLearnedRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import FRAME_INFO_BITS, decode_all_frames
from coded_learned_relay import (
    WINDOW as MLP_WINDOW,
    generate_coded_training_data as gen_mlp_data,
    TRAIN_FRAMES_PER_SNR as MLP_TRAIN_FRAMES,
)

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 100
N_FRAMES = 500
ORDER = ["coded_df", "soft_df", "mlp_hard", "mlp_soft", "oracle"]


class OracleRelay:
    def __init__(self):
        self.truth = None

    def process(self, received_signal):
        return self.truth


def main():
    t0 = time.time()
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    print("Training the QPSK classifier (shared by MLP hard and MLP soft) ...", flush=True)
    mlp = MLPQPSKClassifierRelay(window_size=MLP_WINDOW, hidden_size=16, seed=0)
    X, T = gen_mlp_data(encoder, mlp, MLP_TRAIN_FRAMES)
    mlp.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)

    soft_df = SoftCodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    oracle = OracleRelay()
    relays = {
        "coded_df": CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS),
        "soft_df": soft_df,
        "mlp_hard": mlp,
        "mlp_soft": SoftLearnedRelay(mlp),
        "oracle": oracle,
    }

    results = {"snr_db": SNRS, "n_trials": N_TRIALS, "n_frames": N_FRAMES,
               "per_trial": {}, "summary": {}}

    for snr_db in SNRS:
        soft_df.set_snr_db(snr_db)  # relay knows the nominal SNR only
        per_trial = {k: [] for k in relays}
        print(f"\n=== {snr_db} dB, {N_TRIALS} paired trials ===", flush=True)

        for t in range(N_TRIALS):
            seed = 800000 + 1000 * snr_db + t
            rng = np.random.default_rng(seed)
            info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
            coded = np.concatenate([
                encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
                for f in range(N_FRAMES)
            ])
            tx = qpsk_modulate(coded)
            oracle.truth = tx

            for name, relay in relays.items():
                np.random.seed(seed % (2 ** 31))
                rx_relay = rayleigh_fading_channel(tx, snr_db)
                relay_out = relay.process(rx_relay)
                rx_dest = rayleigh_fading_channel(relay_out, snr_db)
                info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)
                n = min(len(info_hat), len(info_bits))
                per_trial[name].append(float(np.mean(info_hat[:n] != info_bits[:n])))

            if (t + 1) % 5 == 0:
                print("  " + f"{t+1}/{N_TRIALS}  "
                      + "  ".join(f"{k}={np.mean(per_trial[k]):.5f}" for k in ORDER),
                      flush=True)

        results["per_trial"][str(snr_db)] = per_trial
        summ = {}
        for k, v in per_trial.items():
            a = np.asarray(v)
            summ[k] = {"mean": float(a.mean()),
                       "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))}

        base = np.asarray(per_trial["coded_df"])
        for k in ("soft_df", "mlp_hard", "mlp_soft"):
            arr = np.asarray(per_trial[k])
            diff = arr - base
            try:
                _, p = stats.wilcoxon(arr, base)
            except ValueError:
                p = 1.0
            summ[k]["vs_hard_coded_df"] = {
                "mean_diff": float(diff.mean()),
                "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()),
                "wilcoxon_p": float(p),
            }
        results["summary"][str(snr_db)] = summ

        print(f"\n  {snr_db} dB:")
        for k in ORDER:
            s = summ[k]
            extra = ""
            if "vs_hard_coded_df" in s:
                c = s["vs_hard_coded_df"]
                sig = "sig" if c["wilcoxon_p"] < 0.05 else "n.s."
                extra = (f"   vs hard coded-DF: {c['mean_diff']:+.6f}  "
                         f"W/L={c['wins']}/{c['losses']}  p={c['wilcoxon_p']:.4f} ({sig})")
            print(f"    {k:10s} {s['mean']:.6f} +/- {s['ci95']:.6f}{extra}")

    with open("results/coded_soft_decision.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved results/coded_soft_decision.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
