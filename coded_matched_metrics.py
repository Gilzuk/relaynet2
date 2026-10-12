#!/usr/bin/env python3
"""Coded two-hop comparison with reliability-weighted (matched) decoder metrics.

The coded classical baselines of Chapter 5 decode the zero-forced observation
y/h = x + n/h with an unweighted Euclidean metric (Viterbi) or one nominal noise
variance (BCJR). After zero forcing the noise variance is sigma^2/|h_i|^2 per
symbol, so the matched metric weights each squared residual by |h_i|^2, which is
the same as decoding |y - h x|^2 on the unequalized observation. This script
reruns the principal coded comparisons (Tables tbl:table34, tbl:table37,
tbl:table39) with that metric at the relay and at the destination, alongside the
published unweighted metric, on identical channel draws.

Protocol: identical to coded_soft_decision.py -- canonical QPSK/Rayleigh, both
hops; rate-1/2 K=3 code; 200 information bits per frame, 500 frames per trial;
30 paired trials per SNR (the first 30 of the published 100; a 100-trial run
was stopped once its running means had settled), seed 800000 + 1000*SNR + t; the same MLP classifier,
trained with the same data and seed. Within a trial every scheme sees the same
information bits and the same hop-1/hop-2 fading and noise. The unweighted rows
therefore reproduce results/coded_soft_decision.json trial by trial, which is
checked and recorded.

Relays (what the relay does)        destination metrics (both are run)
  coded_df     Viterbi, unweighted   plain   : unweighted Euclidean (published)
  coded_df_m   Viterbi, |h1|^2       matched : |h2|^2 per symbol; for AF the
  soft_df      BCJR, nominal sigma^2           end-to-end variance
  soft_df_m    BCJR, sigma^2/|h1|^2            g^2/|h1|^2 + 1/|h2|^2 with
  mlp_hard     learned, no CSI                 reference points scaled by g
  mlp_soft     learned, no CSI
  oracle       clean codeword
  coded_af     block-normalized AF

The learned relays are not given |h1|; only their destination metric changes.
SNR convention: gamma = 10^(SNR/10), unit symbol energy, noise N0/2 = 1/(2 gamma)
per real dimension (relaynet.channels.fading.rayleigh_fading_channel).
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy import stats

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.coding.bcjr import BCJRCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay, SoftLearnedRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import FRAME_INFO_BITS
from coded_learned_relay import (
    WINDOW as MLP_WINDOW,
    generate_coded_training_data as gen_mlp_data,
    TRAIN_FRAMES_PER_SNR as MLP_TRAIN_FRAMES,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "coded_matched_metrics.json")
PUBLISHED = os.path.join(ROOT, "results", "coded_soft_decision.json")
SNRS = [int(s) for s in os.environ.get("CMM_SNRS", "0,4,8,12,16,20").split(",")]
N_TRIALS = int(os.environ.get("CMM_TRIALS", "30"))
N_FRAMES = 500
RELAYS = ["coded_df", "coded_df_m", "soft_df", "soft_df_m",
          "mlp_hard", "mlp_soft", "oracle", "coded_af"]
DESTS = ["plain", "matched"]

ENC = ConvolutionalEncoder()
DEC = ViterbiCodeDecoder()
FRAME_SYMBOLS = FRAME_INFO_BITS + DEC.num_tail
MLP = None  # trained in main(), inherited by the worker processes


def interleave(seg):
    soft = np.empty(2 * len(seg), dtype=float)
    soft[0::2] = seg.real
    soft[1::2] = seg.imag
    return soft


def decode_frames(rx, weights=None):
    """Viterbi-decode every frame; weights is one value per QPSK symbol or None."""
    out = []
    for f in range(len(rx) // FRAME_SYMBOLS):
        sl = slice(f * FRAME_SYMBOLS, (f + 1) * FRAME_SYMBOLS)
        out.append(DEC.decode(interleave(rx[sl]),
                              weights=None if weights is None else weights[sl]))
    return np.concatenate(out)


def matched_block_df(y, h):
    """Hard block DF with the |h|^2-weighted Viterbi metric at the relay."""
    out = np.empty(len(y), dtype=complex)
    for f in range(len(y) // FRAME_SYMBOLS):
        sl = slice(f * FRAME_SYMBOLS, (f + 1) * FRAME_SYMBOLS)
        info = DEC.decode(interleave(y[sl]), weights=np.abs(h[sl]) ** 2)
        out[sl] = qpsk_modulate(ENC.encode(info))
    return out


def matched_soft_df(y, h, snr_db, bcjr):
    """Soft block DF: BCJR posterior means with per-symbol variance sigma^2/|h|^2.

    The observation is rescaled by sqrt(2) so the BCJR's +/-1 reference points
    match the QPSK axis amplitude; the per-real-dimension variance becomes
    2 * 1/(2 gamma |h|^2) = 1/(gamma |h|^2).
    """
    gamma = 10 ** (snr_db / 10)
    out = np.empty(len(y), dtype=complex)
    for f in range(len(y) // FRAME_SYMBOLS):
        sl = slice(f * FRAME_SYMBOLS, (f + 1) * FRAME_SYMBOLS)
        p1 = bcjr.coded_bit_posteriors(np.sqrt(2) * interleave(y[sl]),
                                       noise_var=1.0 / (gamma * np.abs(h[sl]) ** 2))
        out[sl] = ((1 - 2 * p1[:, 0]) + 1j * (1 - 2 * p1[:, 1])) / np.sqrt(2)
    return out / np.sqrt(np.mean(np.abs(out) ** 2))


def run_snr(snr_db):
    soft_df = SoftCodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    soft_df.set_snr_db(snr_db)
    coded_df = CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    af = AmplifyAndForwardRelay()
    mlp_soft = SoftLearnedRelay(MLP)
    bcjr = BCJRCodeDecoder()
    per = {f"{r}|{d}": [] for r in RELAYS for d in DESTS}
    t0 = time.time()
    for t in range(N_TRIALS):
        seed = 800000 + 1000 * snr_db + t
        rng = np.random.default_rng(seed)
        info = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
        tx = qpsk_modulate(np.concatenate([
            ENC.encode(info[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
            for f in range(N_FRAMES)]))
        for r in RELAYS:
            # Same global-RNG draws as coded_soft_decision.py: return_channel only
            # returns h, it does not change what is drawn.
            np.random.seed(seed % (2 ** 31))
            y1, h1 = rayleigh_fading_channel(tx, snr_db, return_channel=True)
            g = None
            if r == "coded_df":
                out = coded_df.process(y1)
            elif r == "coded_df_m":
                out = matched_block_df(y1, h1)
            elif r == "soft_df":
                out = soft_df.process(y1)
            elif r == "soft_df_m":
                out = matched_soft_df(y1, h1, snr_db, bcjr)
            elif r == "mlp_hard":
                out = MLP.process(y1)
            elif r == "mlp_soft":
                out = mlp_soft.process(y1)
            elif r == "oracle":
                out = tx
            else:
                out = af.process(y1)
                g = float(np.real(np.mean(out / y1)))
            y2, h2 = rayleigh_fading_channel(out, snr_db, return_channel=True)
            n = len(info)
            hat = decode_frames(y2)
            per[f"{r}|plain"].append(float(np.mean(hat[:n] != info[:n])))
            if g is None:
                hat = decode_frames(y2, weights=np.abs(h2) ** 2)
            else:
                # AF: y2 = g x + g n1/h1 + n2/h2; per-symbol variance is
                # proportional to g^2/|h1|^2 + 1/|h2|^2 (hop-2 power is 1).
                w = 1.0 / (g ** 2 / np.abs(h1) ** 2 + 1.0 / np.abs(h2) ** 2)
                hat = decode_frames(y2 / g, weights=w * g ** 2)
            per[f"{r}|matched"].append(float(np.mean(hat[:n] != info[:n])))
        if (t + 1) % 10 == 0:
            print(f"[{snr_db:2d} dB] {t+1}/{N_TRIALS}  {time.time()-t0:.0f}s  "
                  f"df={np.mean(per['coded_df|plain']):.5f}/"
                  f"{np.mean(per['coded_df_m|matched']):.5f}  "
                  f"mlp={np.mean(per['mlp_hard|plain']):.5f}/"
                  f"{np.mean(per['mlp_hard|matched']):.5f}", flush=True)
    return snr_db, per


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_diff": float(d.mean()), "wins": int((d < 0).sum()),
            "losses": int((d > 0).sum()), "wilcoxon_p": p}


def main():
    global MLP
    t0 = time.time()
    print("Training the QPSK classifier (as coded_soft_decision.py) ...", flush=True)
    MLP = MLPQPSKClassifierRelay(window_size=MLP_WINDOW, hidden_size=16, seed=0)
    X, T = gen_mlp_data(ENC, MLP, MLP_TRAIN_FRAMES)
    MLP.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)

    with mp.get_context("fork").Pool(min(len(SNRS), os.cpu_count() or 1)) as pool:
        runs = dict(pool.map(run_snr, SNRS))

    published = json.load(open(PUBLISHED))["per_trial"] if os.path.exists(PUBLISHED) else {}
    out = {"description": __doc__.strip().splitlines()[0],
           "snr_db": SNRS, "n_trials": N_TRIALS, "n_frames": N_FRAMES,
           "frame_info_bits": FRAME_INFO_BITS, "relays": RELAYS, "dests": DESTS,
           "per_trial": {str(s): runs[s] for s in SNRS}, "summary": {},
           "reproduction": {}}
    for s in SNRS:
        per = runs[s]
        summ = {k: {"mean": float(np.mean(v)),
                    "ci95": float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))}
                for k, v in per.items()}
        ref = per["coded_df_m|matched"]
        for k in ("mlp_hard|matched", "mlp_soft|matched", "soft_df_m|matched",
                  "coded_df|plain"):
            summ[k]["vs_matched_block_df"] = compare(per[k], ref)
        for k in ("mlp_hard|matched", "mlp_soft|matched"):
            summ[k]["vs_matched_soft_df"] = compare(per[k], per["soft_df_m|matched"])
        out["summary"][str(s)] = summ
        pub = published.get(str(s), {})
        rep = {}
        for r in ("coded_df", "soft_df", "mlp_hard", "mlp_soft", "oracle"):
            if r in pub and len(pub[r]) >= len(per[f"{r}|plain"]):
                mine = np.asarray(per[f"{r}|plain"])
                rep[r] = float(np.max(np.abs(np.asarray(pub[r][:len(mine)]) - mine)))
        out["reproduction"][str(s)] = rep
        print(f"\n{s} dB  (max |plain - published| per trial: {rep})")
        for k in sorted(summ):
            extra = summ[k].get("vs_matched_block_df")
            line = f"  {k:20s} {summ[k]['mean']:.6f} +/- {summ[k]['ci95']:.6f}"
            if extra:
                line += (f"   vs matched block-DF {extra['mean_diff']:+.6f} "
                         f"W/L {extra['wins']}/{extra['losses']} p={extra['wilcoxon_p']:.2g}")
            print(line)
    out["seconds"] = round(time.time() - t0)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
