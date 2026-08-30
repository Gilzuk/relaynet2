#!/usr/bin/env python3
"""Self-test: bounded-traceback MLSE must converge to the full-block decoder.

TruncatedViterbiQPSKRelay is the only equalizer in the framework with an
explicit decision delay, so the latency claims made from it rest on this
check: as the traceback depth grows it has to reproduce
ViterbiMLSEQPSKRelay symbol for symbol, and at depth 0 it must degrade to
a symbol-by-symbol decision rather than silently misalign its output.
"""

import numpy as np
from relaynet.relays import ViterbiMLSEQPSKRelay, TruncatedViterbiQPSKRelay
from relaynet.channels.e6_channels import ComplexISIChannel


def main():
    taps = np.array([1.0, 0.7, 0.5])
    taps /= np.linalg.norm(taps)
    rng = np.random.default_rng(0)
    x = ViterbiMLSEQPSKRelay.ALPHABET[rng.integers(0, 4, 4000)]
    y = ComplexISIChannel(taps, seed=1)(x, 12)

    reference = ViterbiMLSEQPSKRelay(channel_taps=taps).process(y)
    assert np.mean(reference != x) < 1e-9, "full-block MLSE should be error-free at 12 dB"

    agreement = {}
    for D in (0, 1, 3, 5, 10, 15, 30):
        out = TruncatedViterbiQPSKRelay(channel_taps=taps, traceback=D).process(y)
        assert len(out) == len(y), f"D={D}: output length changed"
        agreement[D] = float(np.mean(out == reference))
        print(f"D={D:3d}  SER={np.mean(out != x):.5f}  "
              f"agree_with_full_block={agreement[D]:.5f}")

    assert agreement[0] < 1.0, "depth 0 should not already match the full decoder"
    for D in (5, 10, 15, 30):
        assert agreement[D] == 1.0, f"D={D} must match the full-block decoder exactly"
    assert all(agreement[a] <= agreement[b]
               for a, b in zip((0, 1, 3, 5), (1, 3, 5, 10))), \
        "agreement must be non-decreasing in traceback depth"
    print("\nOK: truncated MLSE converges to the full-block decoder by D=5")


if __name__ == "__main__":
    main()
