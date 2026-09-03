"""The three trellis controls behind the QPSK benchmark withdrawal, committed.

Chapter 7's QPSK subsection (sec:qpsk-unknown-channel) separates two possible
explanations for the withdrawn "MLP beats genie-CSI Viterbi" result with three
controls on the same trellis at 20 dB: the taps-only trellis on the faded
channel, the same trellis with the fading removed, and the fading-aware
(genie-CSI) trellis on the faded channel. The numbers those controls first
produced were run ad hoc and never committed, and they disagree with the
committed decomposition (`results/qpsk_error_decomposition.json`) because they
were produced under a different configuration. This script reruns them under
the exact configuration of the published table (`tbl:tableE6qpsk`, produced by
`e6_qpsk_unknown_channel.py`): same taps, same channel classes, same trellis
classes, same trial protocol and seeds as `qpsk_error_decomposition.py`.

Measured at the relay output (hop 2 would add its own errors and blur exactly
the distinction under test), symbol error rate via nearest-symbol decision.

SNR convention: gamma = 10^(SNR_dB/10), per memory-bank/techContext.md.

Output: results/qpsk_trellis_controls.json
"""

import json
import os

import numpy as np

from e6_qpsk_unknown_channel import H_ISI, ComplexISIRayleighChannel, qpsk_mod
from qpsk_error_decomposition import nearest_symbol
from relaynet.relays.viterbi import (FadingAwareViterbiQPSKRelay,
                                     ViterbiMLSEQPSKRelay)

N_TRIALS, N_BITS = 10, 100_000          # project-standard scale
SNR_DB = 20                             # the SNR point the prose quotes

# The published table's trellises carry the *normalized* taps: the channel
# classes normalize their taps in place (np.asarray returns the caller's
# array), so in `e6_qpsk_unknown_channel.run_setup` constructing hop 1 first
# normalizes the shared H_ISI before the trellises read it. Construction order
# must not decide the physics here, so the normalization is made explicit --
# on a copy, to leave the imported H_ISI untouched for anyone imported later.
H_NORM = np.asarray(H_ISI, dtype=float).copy()
H_NORM /= np.linalg.norm(H_NORM)


class ComplexISIChannel:
    """The fading-removed control: conv with the same normalized taps + complex
    AWGN, i.e. `ComplexISIRayleighChannel` with g[n] forced to 1. Noise and
    normalization conventions copied from that class so the control differs
    from the faded channel in the fading term only."""

    def __init__(self, taps, seed=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        self.rng = np.random.default_rng(seed)

    def __call__(self, signal, snr_db):
        isi_output = np.convolve(signal, self.taps)[:signal.size]
        self.last_gains = np.ones(signal.size)
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * (
            self.rng.standard_normal(signal.size) +
            1j * self.rng.standard_normal(signal.size)
        ) / np.sqrt(2)
        return isi_output + noise


def measure(relay, hop1, n_bits, snr_db, seed):
    """SER and BER at the relay output, same protocol as the decomposition."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    if bits.size % 2:
        bits = bits[:-1]
    x = qpsk_mod(bits)

    hop1.rng = np.random.default_rng(seed + 101)
    y = hop1(x, snr_db)
    if isinstance(relay, FadingAwareViterbiQPSKRelay):
        relay.set_gains(getattr(hop1, "last_gains", None))
    out = relay.process(y)

    tx_idx = nearest_symbol(x)
    rx_idx = nearest_symbol(out)
    ser = float(np.mean(tx_idx != rx_idx))
    # Gray map: adjacent symbols differ in one bit, diagonal in two.
    diff = tx_idx ^ rx_idx
    bit_errs = ((diff & 1) != 0).sum() + ((diff & 2) != 0).sum()
    ber = float(bit_errs) / (2 * tx_idx.size)
    return ser, ber


def main():
    controls = {
        "taps_only_faded": (
            ViterbiMLSEQPSKRelay(channel_taps=H_NORM),
            ComplexISIRayleighChannel(H_NORM.copy(), seed=1)),
        "taps_only_fading_removed": (
            ViterbiMLSEQPSKRelay(channel_taps=H_NORM),
            ComplexISIChannel(H_NORM.copy(), seed=1)),
        "genie_csi_faded": (
            FadingAwareViterbiQPSKRelay(channel_taps=H_NORM),
            ComplexISIRayleighChannel(H_NORM.copy(), seed=1)),
    }

    out = {"snr_db": SNR_DB, "n_trials": N_TRIALS, "n_bits": N_BITS,
           "taps": H_NORM.tolist(), "controls": {}}
    print(f"{N_TRIALS} trials x {N_BITS} bits at {SNR_DB} dB, relay output\n")
    for name, (relay, hop1) in controls.items():
        sers, bers = zip(*[measure(relay, hop1, N_BITS, SNR_DB, 1000 + t)
                           for t in range(N_TRIALS)])
        out["controls"][name] = {
            "ser": float(np.mean(sers)), "ber": float(np.mean(bers)),
            "ser_per_trial": [float(s) for s in sers],
        }
        print(f"  {name:28s} SER {np.mean(sers):.4f}  BER {np.mean(bers):.4f}",
              flush=True)

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "qpsk_trellis_controls.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWritten to {dest}")
    return out


if __name__ == "__main__":
    main()
