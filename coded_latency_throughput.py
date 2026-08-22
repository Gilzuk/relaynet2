"""Latency and throughput accounting for the coded relays.

The BER tables in Section~sec:coded-block-df compare relays at equal
Es/N0, which is the right axis for a detection question but hides two
costs a link designer cannot ignore.

1. THROUGHPUT. A rate-1/2 code halves the information carried per channel
   use, so "coded block-DF beats uncoded DF" at equal Es/N0 is not a
   like-for-like statement: the coded link delivers half the data. The
   equal-spectral-efficiency comparison is rate-1/2 16-QAM (4 x 1/2 = 2
   info bits/symbol) against uncoded QPSK (2 info bits/symbol), which is
   computed here from the already-measured data -- no new simulation.

2. LATENCY. Relays differ enormously in how much they must buffer before
   they can emit anything. Symbol-wise DF emits immediately; a windowed
   learned relay needs w symbols of look-ahead; block-DF cannot emit until
   it has decoded a whole frame, and BCJR additionally needs a backward
   recursion over that frame, so it is structurally incapable of starting
   early. That buffering latency scales with frame length for the block
   relays and is constant for the learned ones.

Both are measured/derived here rather than argued.
"""

import json
import time

import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay, SoftLearnedRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.modulation.qpsk import qpsk_modulate

FRAME_INFO_BITS = 200
N_SYMBOLS = 50_000
REPEATS = 7


def bench(relay, x, repeats=REPEATS):
    """Median wall-clock over `repeats` runs, after a discarded warm-up.

    The warm-up matters: without it the first relay benchmarked pays for
    cold BLAS/allocator state and the last looks artificially fast, which
    is exactly the kind of artefact that would misreport a 756-parameter
    forward pass as several times cheaper than an identical one.
    """
    relay.process(x[: min(len(x), 5000)])  # warm-up, discarded
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        relay.process(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    enc = ConvolutionalEncoder()
    tail = enc.num_tail
    frame_symbols = FRAME_INFO_BITS + tail  # QPSK: 1 trellis step == 1 symbol

    # ---- 1. throughput / spectral efficiency -------------------------------
    with open("results/coded_df_experiment.json") as f:
        d = json.load(f)
    snrs = d["snr_db"]
    eff_uncoded_qpsk = 2.0
    eff_coded_qpsk = 2.0 * FRAME_INFO_BITS / (2 * frame_symbols)
    eff_coded_qam16 = 4.0 * FRAME_INFO_BITS / (2 * frame_symbols)

    print("=" * 78)
    print("SPECTRAL EFFICIENCY (information bits per channel symbol)")
    print("=" * 78)
    print(f"  uncoded QPSK          {eff_uncoded_qpsk:.3f}")
    print(f"  rate-1/2 QPSK         {eff_coded_qpsk:.3f}   <- half the data rate")
    print(f"  rate-1/2 16-QAM       {eff_coded_qam16:.3f}   <- matches uncoded QPSK")

    print()
    print("=" * 78)
    print("EQUAL-THROUGHPUT COMPARISON (both carry ~2 info bits/symbol)")
    print("=" * 78)
    print(f"{'SNR':>6} {'uncoded QPSK':>14} {'rate-1/2 16-QAM':>17}   verdict")
    tput_rows = []
    for i, s in enumerate(snrs):
        u = d["uncoded_df"][i]
        c = d["qam16_k_sweep"]["K3"]["ber"][i]
        verdict = (f"coding wins {u/c:.2f}x" if c < u else f"coding LOSES {c/u:.2f}x")
        print(f"{s:>4} dB {u:>14.5f} {c:>17.5f}   {verdict}")
        tput_rows.append({"snr_db": s, "uncoded_qpsk": u, "coded_qam16": c,
                          "ratio_uncoded_over_coded": u / c})

    # ---- 2. structural (buffering) latency ---------------------------------
    print()
    print("=" * 78)
    print("STRUCTURAL LATENCY (symbols the relay must buffer before it can emit)")
    print("=" * 78)
    mlp_window = 21
    structural = [
        ("AF / symbol-wise DF", 0, "per-symbol, emits immediately"),
        ("MLP / Mamba, window 21", mlp_window // 2, "w symbols of look-ahead, constant in frame length"),
        ("hard block-DF (Viterbi)", frame_symbols, "whole frame before decode; grows with frame length"),
        ("soft block-DF (BCJR)", frame_symbols, "whole frame, and the backward pass forbids starting early"),
    ]
    print(f"{'relay':30} {'buffer (symbols)':>18}   note")
    for name, lat, note in structural:
        print(f"{name:30} {lat:>18}   {note}")

    # ---- 3. measured compute cost ------------------------------------------
    print()
    print("=" * 78)
    print(f"COMPUTE COST (median of {REPEATS} runs over {N_SYMBOLS:,} symbols)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    n = (N_SYMBOLS // frame_symbols) * frame_symbols
    info = rng.integers(0, 2, (n // frame_symbols) * FRAME_INFO_BITS)
    coded = np.concatenate([enc.encode(info[i * FRAME_INFO_BITS:(i + 1) * FRAME_INFO_BITS])
                            for i in range(n // frame_symbols)])
    x = qpsk_modulate(coded) + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    mlp = MLPQPSKClassifierRelay(window_size=mlp_window, hidden_size=16, seed=0)
    soft_df = SoftCodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    soft_df.set_snr_db(16)
    cands = {
        "hard block-DF (Viterbi)": CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS),
        "soft block-DF (BCJR)": soft_df,
        "MLP hard (756p)": mlp,
        "MLP soft (756p)": SoftLearnedRelay(mlp),
    }
    print(f"{'relay':30} {'total (s)':>10} {'us/symbol':>12} {'Msym/s':>9}")
    compute = {}
    for name, r in cands.items():
        t = bench(r, x)
        us = t / n * 1e6
        print(f"{name:30} {t:>10.3f} {us:>12.2f} {n/t/1e6:>9.3f}")
        compute[name] = {"total_s": t, "us_per_symbol": us, "msym_per_s": n / t / 1e6}

    with open("results/coded_latency_throughput.json", "w") as f:
        json.dump({"spectral_efficiency": {"uncoded_qpsk": eff_uncoded_qpsk,
                                           "coded_qpsk": eff_coded_qpsk,
                                           "coded_qam16": eff_coded_qam16},
                   "equal_throughput": tput_rows,
                   "structural_latency_symbols": {n_: l for n_, l, _ in structural},
                   "compute": compute,
                   "n_symbols": n, "frame_symbols": frame_symbols}, f, indent=2)
    print("\nSaved results/coded_latency_throughput.json")


if __name__ == "__main__":
    main()
