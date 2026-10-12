#!/usr/bin/env python3
"""Required-SNR penalty of the MLP against genie Viterbi on the fixed 3-tap ISI chain.

The four penalties quoted in Ch. 6 (Table tbl:layers, Layer 2) and Ch. 9 (finding 8)
were computed by hand from results/e6_matched_protocol.json and never committed.
This script derives them with the thesis's single penalty definition
(ber_metrics.snr_at_ber: linear interpolation of ln BER against SNR in dB between
the two points that bracket the target, last crossing) and writes them to the
repository, so the published numbers have a reproducible source.

No simulation is run: the input is the committed matched-protocol BER data.
"""

import json
import os

from ber_metrics import snr_at_ber

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "results", "e6_matched_protocol.json")
OUT = os.path.join(ROOT, "results", "e6_isi_penalty.json")
TARGETS = (1e-1, 1e-2, 1e-3, 1e-4)
RELAY, REFERENCE = "MLP", "VIT-genie"


def main():
    d = json.load(open(SRC))
    snrs, res = d["snrs"], d["results"]
    rows = {}
    for t in TARGETS:
        a = snr_at_ber(snrs, res[RELAY]["ber"], t)
        b = snr_at_ber(snrs, res[REFERENCE]["ber"], t)
        rows[f"{t:.0e}"] = {"snr_relay_db": a, "snr_reference_db": b,
                            "penalty_db": a - b}
        print(f"BER {t:.0e}: {RELAY} {a:6.3f} dB  {REFERENCE} {b:6.3f} dB  "
              f"penalty {a - b:+.3f} dB")
    out = {"source": "results/e6_matched_protocol.json",
           "relay": RELAY, "reference": REFERENCE,
           "metric": "ber_metrics.snr_at_ber, ln-BER linear interpolation, last crossing",
           "targets": list(TARGETS), "penalties": rows}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
