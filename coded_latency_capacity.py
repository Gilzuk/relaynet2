"""Capacity under a strict latency constraint -- the finite-blocklength framing.

Section~sec:coded-rate-adaptation optimizes goodput per SNR with no bound
on how long the relay may take to decide. That is the wrong question for
a link with a deadline. The right one is standard in the coding-theory
literature under two names:

  finite-blocklength capacity   Polyanskiy, Poor & Verdu (2010): constrain
                                 the blocklength n and the achievable rate
                                 drops by O(1/sqrt(n)) relative to Shannon
                                 capacity -- n *is* a latency budget in
                                 channel uses.
  delay-limited capacity        Hanly & Tse (1998): the rate achievable
                                 under a hard per-block deadline in a
                                 fading channel, as opposed to the ergodic
                                 (infinite-delay-tolerance) capacity.

This script re-derives the AMC envelope of coded_rate_adaptation.py under
a round-trip latency budget L_max, using the same already-measured FER
data -- no new simulation.

Latency accounting (consistent with the structural-latency figures already
published in Section~sec:coded-latency-throughput):

  destination     ALWAYS buffers a full coded frame before it can decode,
                  for every relay strategy -- this is unavoidable for any
                  coded MCS.
  block-DF relay  ALSO buffers a full frame (decode, re-encode, forward),
                  so its round-trip cost is 2x the frame length.
  denoise relay   only needs its window (10 symbols, constant), so its
                  round-trip cost is approximately 1x the frame length.
  uncoded MCS     no block decode anywhere; round-trip cost is 0 for both
                  strategies.

Sweeping L_max traces a rate-vs-blocklength curve for each relay strategy,
directly analogous to the textbook finite-blocklength rate-vs-n plot.
"""

import json
import math

from relaynet.coding.puncturing import PuncturedCode
from relaynet.coding.bicm import BITS_PER_SYMBOL

FRAME_INFO_BITS = 200
WINDOW_HALF = 10  # relaynet learned-relay window/2 (Table~tbl:table41)
TABLE_BUDGET = 150  # illustrative round-trip budget for the snapshot table


def frame_symbols(mod, rate):
    pc = PuncturedCode(rate=rate)
    n_steps = pc.n_steps(FRAME_INFO_BITS)
    coded_bits = pc.n_coded_bits(n_steps)
    return math.ceil(coded_bits / BITS_PER_SYMBOL[mod])


def total_latency(mod, rate, relay):
    if rate == "uncoded":
        return 0
    fs = frame_symbols(mod, rate)
    relay_buf = fs if relay == "blockdf" else WINDOW_HALF
    return relay_buf + fs  # relay hop, then the destination's own mandatory decode


def best_at_budget(entries, relay, snr_idx, l_max):
    best = None
    for (mod, rate, rl), v in entries.items():
        if rl != relay and not (rate == "uncoded" and rl == "denoise"):
            continue
        lat = total_latency(mod, rate, relay)
        if lat > l_max:
            continue
        g = v["goodput"][snr_idx]
        if best is None or g > best["goodput"]:
            best = {"mod": mod, "rate": rate, "latency": lat,
                    "rate_info_bits_per_symbol": v["rate_info_bits_per_symbol"],
                    "fer": v["fer"][snr_idx], "goodput": g}
    return best


def main():
    with open("results/coded_rate_adaptation.json") as f:
        d = json.load(f)
    snrs = d["snr_db"]
    entries = {}
    for key, v in d["mcs"].items():
        mod, rate, relay = key.split("|")
        entries[(mod, rate, relay)] = v

    # ---- latency table for every (mod, rate, relay) combination ----------
    latency_table = []
    for (mod, rate, relay) in entries:
        latency_table.append({"mod": mod, "rate": rate, "relay": relay,
                              "round_trip_symbols": total_latency(mod, rate, relay)})
    latency_table.sort(key=lambda r: r["round_trip_symbols"])

    print("=" * 78)
    print("ROUND-TRIP LATENCY BY (MODULATION, CODE RATE, RELAY STRATEGY)")
    print("=" * 78)
    for r in latency_table:
        print(f"  {r['mod']:6s} {r['rate']:8s} {r['relay']:8s} "
              f"{r['round_trip_symbols']:4d} symbols")

    # ---- snapshot table at TABLE_BUDGET, same layout as table42 ----------
    print()
    print("=" * 78)
    print(f"LINK-ADAPTATION ENVELOPE UNDER A {TABLE_BUDGET}-SYMBOL ROUND-TRIP BUDGET")
    print("=" * 78)
    snapshot = []
    for i, snr in enumerate(snrs):
        row = {"snr_db": snr}
        for relay in ("blockdf", "denoise"):
            b = best_at_budget(entries, relay, i, TABLE_BUDGET)
            row[relay] = b
        snapshot.append(row)
        bd, dn = row["blockdf"], row["denoise"]
        print(f"{snr:>4} dB  blockdf: {bd['mod']} {bd['rate']:6s} G={bd['goodput']:.4f}"
              f"   denoise: {dn['mod']} {dn['rate']:6s} G={dn['goodput']:.4f}")

    # ---- full capacity-vs-latency-budget curves, for plotting ------------
    breakpoints = sorted(set(r["round_trip_symbols"] for r in latency_table))
    curves = {}
    for relay in ("blockdf", "denoise"):
        curves[relay] = {}
        for i, snr in enumerate(snrs):
            pts = []
            for l_max in breakpoints:
                b = best_at_budget(entries, relay, i, l_max)
                pts.append({"l_max": l_max, "goodput": b["goodput"] if b else 0.0})
            curves[relay][str(snr)] = pts

    with open("results/coded_latency_capacity.json", "w") as f:
        json.dump({"snr_db": snrs, "table_budget": TABLE_BUDGET,
                   "latency_table": latency_table, "snapshot": snapshot,
                   "curves": curves}, f, indent=2)
    print("\nSaved results/coded_latency_capacity.json")


if __name__ == "__main__":
    main()
