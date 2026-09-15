"""Deep-exposure Viterbi at 16 and 20 dB, matched to e6_matched_protocol.py.

The matched-protocol run gives every relay 10 x 100,000 bits per SNR point.
That is enough to resolve the comparison up to about 10 dB, but above it the
classical detector produces no errors at all, so its cells become rule-of-three
bounds at 3e-6 while the learned relay is measured down to ~1.5e-7 by the
dedicated fixed-budget runs. Reporting those side by side would suggest the
learned relay wins at high SNR when in fact the comparison is simply
unresolved at the shallower exposure -- the mirror image of the convention
error this whole exercise is correcting.

This script therefore gives genie-CSI and pilot-LS Viterbi the same order of
exposure the learned relay already has at 16 and 20 dB: a predeclared 1e8
bits per detector per SNR point, no data-dependent stopping. Channel, taps,
hop-2 convention and detector configuration are identical to
e6_matched_protocol.py.

Writes results/e6_matched_highsnr.json. Run: python e6_matched_highsnr.py
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

import e6_sim_ported as e6
from relaynet.relays import ViterbiMLSERelay

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "e6_matched_highsnr.json")

H_CHANNEL = np.array([1.0, 0.7, 0.5])
H_ISI = H_CHANNEL / np.linalg.norm(H_CHANNEL)   # the channel's effective taps
SNRS = (16, 20)
N_PILOT = 200
BLOCK = 250_000


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main(total_bits):
    hop1 = e6.create_channel("isi", seed=1)
    hop2 = e6.create_channel("awgn", seed=2)
    dst = e6.Destination(modulation="bpsk")
    rows = {}
    t0 = time.time()

    for snr in SNRS:
        acc = {"VIT-genie": 0, "VIT-est": 0}
        sent = 0
        while sent < total_bits:
            n = min(BLOCK, total_bits - sent)
            src = e6.Source(seed=900_000 + 13 * snr + sent // BLOCK,
                            modulation="bpsk")
            tx_bits, tx_sym = src.transmit(n)
            y = hop1(tx_sym, snr)
            dets = {
                "VIT-genie": ViterbiMLSERelay(channel_taps=H_ISI),
                "VIT-est": ViterbiMLSERelay(
                    pilot_symbols=(y[:N_PILOT], tx_sym[:N_PILOT])),
            }
            for name, det in dets.items():
                rx = dst.receive(hop2(det.process(y), snr))
                acc[name] += int(np.count_nonzero(tx_bits != rx))
            sent += n
        for name, errs in acc.items():
            rows.setdefault(name, {})[snr] = {
                "errors": errs, "bits": sent,
                "ber": (errs / sent) if errs else None,
                "rule_of_three_upper_95": None if errs else 3.0 / sent,
            }
            shown = f"{errs / sent:.3e}" if errs else f"< {3.0 / sent:.2e}"
            print(f"  {snr} dB  {name:10s} {errs:4d} errors / {sent:,} bits "
                  f"-> {shown}")

    payload = {
        "description": "Deep-exposure genie and pilot-LS Viterbi at 16 and "
                       "20 dB, matched to e6_matched_protocol.py.",
        "taps_effective": H_ISI.tolist(),
        "hop2": "awgn, sigma^2 = N0/2 per real dimension (current convention)",
        "bits_per_detector_per_snr": total_bits,
        "declared_before_run": True,
        "n_pilot": N_PILOT,
        "git_rev": git_rev(),
        "seconds": round(time.time() - t0, 1),
        "results": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=100_000_000)
    main(ap.parse_args().bits)
