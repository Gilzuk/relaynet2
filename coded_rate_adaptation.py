"""Adaptive modulation and coding: pick the rate per SNR to maximize goodput.

Every comparison so far has held the modulation-and-coding scheme (MCS)
fixed and asked which relay gets the lowest BER. That is the wrong
objective for a link, and Section~sec:coded-latency-throughput showed why:
a rate-1/2 code buys reliability with bandwidth, so at equal spectral
efficiency it does not pay for itself below 20 dB. A real link does not
hold the MCS fixed -- it adapts. The right question is therefore not
"which relay has the lowest BER at this rate" but "what is the highest
useful data rate this relay can sustain at this SNR".

Objective: **goodput**, the information rate actually delivered,

    G = R * (1 - FER),   R = information bits per channel symbol,

which counts only error-free frames, since a frame with any residual
error is discarded by any real protocol. Maximizing G over the MCS grid
at each SNR traces the link-adaptation envelope.

MCS grid: {QPSK, 16-QAM} x {uncoded, rate 1/2, 2/3, 3/4}, the punctured
rates coming from the same K=3 mother code (relaynet.coding.puncturing).
Everything runs through a BICM pipeline (relaynet.coding.bicm) so that
puncturing, modulation and decoding stay separable.

Two relay strategies are compared across the whole grid, which is the
point of the exercise -- they embody the two sides of the mechanism found
earlier:

  block-DF      decodes the code at the relay, re-encodes, forwards.
                Spends the code's redundancy before transmitting.
  denoise-only  hard-decides each bit and re-modulates, never invoking
                the code. Leaves the redundancy intact for the
                destination. Needs no training and no code knowledge.

On uncoded MCS the two coincide (there is no code to decode), so those
rows are run once and shared.
"""

import json
import time

import numpy as np

from relaynet.coding.convolutional import ViterbiCodeDecoder
from relaynet.coding.puncturing import PuncturedCode
from relaynet.coding.bicm import modulate_bits, soft_demap, BITS_PER_SYMBOL
from relaynet.channels.fading import rayleigh_fading_channel

SNRS = [8, 12, 16, 20, 24, 28, 32, 36, 40]
RATES = ["uncoded", "1/2", "2/3", "3/4"]
MODS = ["qpsk", "qam16"]
FRAME_INFO_BITS = 200
N_FRAMES = 200
N_TRIALS = 100


def build(mod, rate, info_bits):
    """Encode+puncture a whole trial's frames; return (coded bits, n_steps)."""
    if rate == "uncoded":
        return info_bits, None
    pc = PuncturedCode(rate=rate)
    n_steps = pc.n_steps(FRAME_INFO_BITS)
    coded = np.concatenate([
        pc.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(N_FRAMES)
    ])
    return coded, n_steps


def decode_frames(soft, rate, n_steps, dec):
    """Depuncture + Viterbi each frame; return decoded information bits."""
    pc = PuncturedCode(rate=rate)
    per = len(soft) // N_FRAMES
    out = []
    for f in range(N_FRAMES):
        seg = soft[f * per:(f + 1) * per]
        out.append(dec.decode(pc.depuncture(seg, n_steps)))
    return np.concatenate(out)


def run_trial(mod, rate, relay, snr_db, seed, dec):
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
    coded, n_steps = build(mod, rate, info)
    tx, _ = modulate_bits(coded, mod)

    np.random.seed(seed % (2 ** 31))
    rx1 = rayleigh_fading_channel(tx, snr_db)

    # ---- relay ----
    soft1 = soft_demap(rx1, mod, len(coded))
    if relay == "denoise" or rate == "uncoded":
        fwd_bits = (soft1 < 0).astype(int)
    else:  # block-DF: decode the code, re-encode, forward
        info_hat = decode_frames(soft1, rate, n_steps, dec)
        pc = PuncturedCode(rate=rate)
        fwd_bits = np.concatenate([
            pc.encode(info_hat[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
            for f in range(N_FRAMES)
        ])
    relay_out, _ = modulate_bits(fwd_bits, mod)

    rx2 = rayleigh_fading_channel(relay_out, snr_db)

    # ---- destination ----
    soft2 = soft_demap(rx2, mod, len(coded))
    if rate == "uncoded":
        info_hat = (soft2 < 0).astype(int)
    else:
        info_hat = decode_frames(soft2, rate, n_steps, dec)

    n = min(len(info_hat), len(info))
    err = info_hat[:n] != info[:n]
    ber = float(err.mean())
    fer = float(np.mean([err[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS].any()
                         for f in range(N_FRAMES)]))
    rate_info = len(info) / len(tx)          # information bits per channel symbol
    return ber, fer, rate_info


def main():
    t0 = time.time()
    dec = ViterbiCodeDecoder()
    results = {"snr_db": SNRS, "n_trials": N_TRIALS, "n_frames": N_FRAMES, "mcs": {}}

    for mod in MODS:
        for rate in RATES:
            for relay in ("blockdf", "denoise"):
                if rate == "uncoded" and relay == "blockdf":
                    continue  # identical to denoise; no code to decode
                key = f"{mod}|{rate}|{relay}"
                bers, fers, rinfo = [], [], None
                for snr in SNRS:
                    tb, tf = [], []
                    for t in range(N_TRIALS):
                        b, f, r = run_trial(mod, rate, relay, snr,
                                            310000 + 977 * snr + t, dec)
                        tb.append(b); tf.append(f); rinfo = r
                    bers.append(float(np.mean(tb)))
                    fers.append(float(np.mean(tf)))
                goodput = [rinfo * (1 - f) for f in fers]
                results["mcs"][key] = {"ber": bers, "fer": fers,
                                       "rate_info_bits_per_symbol": rinfo,
                                       "goodput": goodput}
                print(f"{key:26s} R={rinfo:.3f}  "
                      + " ".join(f"{g:.3f}" for g in goodput), flush=True)

    # ---- envelope: best MCS per SNR, per relay strategy ----
    print("\n" + "=" * 86)
    print("LINK-ADAPTATION ENVELOPE  (best MCS at each SNR, by goodput)")
    print("=" * 86)
    envelope = {}
    for relay in ("blockdf", "denoise"):
        rows = []
        print(f"\n-- relay strategy: {relay} --")
        print(f"{'SNR':>5} {'best MCS':>22} {'R':>7} {'FER':>8} {'goodput':>9}")
        for i, snr in enumerate(SNRS):
            best, bestg = None, -1.0
            for key, v in results["mcs"].items():
                m, r, rl = key.split("|")
                # uncoded rows are shared by both strategies
                if rl != relay and not (r == "uncoded" and rl == "denoise"):
                    continue
                if v["goodput"][i] > bestg:
                    bestg, best = v["goodput"][i], (key, v, i)
            key, v, i = best
            m, r, _ = key.split("|")
            print(f"{snr:>4} {m + ' ' + r:>22} {v['rate_info_bits_per_symbol']:>7.3f} "
                  f"{v['fer'][i]:>8.4f} {bestg:>9.4f}")
            rows.append({"snr_db": snr, "mcs": f"{m} {r}",
                         "rate": v["rate_info_bits_per_symbol"],
                         "fer": v["fer"][i], "goodput": bestg})
        envelope[relay] = rows
    results["envelope"] = envelope

    with open("results/coded_rate_adaptation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/coded_rate_adaptation.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
