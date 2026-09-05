"""BER-optimal (BCJR/APP) benchmark against genie-CSI Viterbi MLSE.

Closes Future Work item 4. Chapter 7 benchmarks the learned relay against
genie-CSI Viterbi MLSE, which is sequence-optimal; the thesis reports BER, and
sequence-ML is not BER-optimal on a channel with memory. This measures the
margin against a detector that *is* BER-optimal.

Run under the exact configuration of Table tbl:tableE6qpsk: same taps, same
channel classes, same trial protocol and seeds as qpsk_trellis_controls.py.
Measured at the relay output, since hop 2 would add its own errors and blur
the distinction under test.

Controls run FIRST and gate the comparison, because the withdrawn QPSK result
in this thesis came from a benchmark that looked plausible and was
model-mismatched:

  C1  fading removed: BCJR and Viterbi must agree within Monte Carlo error.
  C2  no ISI: the BER-optimal rule must reduce to the per-axis slicer.

SNR convention: gamma = 10^(SNR_dB/10), per memory-bank/techContext.md.

Output: results/qpsk_bcjr_benchmark.json
"""

import json
import os

import numpy as np

from e6_qpsk_unknown_channel import H_ISI, ComplexISIRayleighChannel, qpsk_mod
from qpsk_error_decomposition import nearest_symbol
from relaynet.relays.bcjr import BCJRQPSKRelay
from relaynet.relays.viterbi import FadingAwareViterbiQPSKRelay

N_TRIALS, N_BITS = 10, 100_000
SNRS = [0, 4, 8, 12, 16, 20]
ROOT = os.path.dirname(os.path.abspath(__file__))

H_NORM = np.asarray(H_ISI, dtype=float).copy()
H_NORM /= np.linalg.norm(H_NORM)


class NoFadingChannel:
    """The C1 control: same taps and noise, fading forced to unity."""

    def __init__(self, taps, seed=None):
        self.taps = np.asarray(taps, dtype=float)
        self.taps /= np.linalg.norm(self.taps)
        self.rng = np.random.default_rng(seed)

    def __call__(self, signal, snr_db):
        isi = np.convolve(signal, self.taps)[:signal.size]
        self.last_gains = np.ones(signal.size)
        sigma = 10 ** (-snr_db / 20.0)
        noise = sigma * (self.rng.standard_normal(signal.size)
                         + 1j * self.rng.standard_normal(signal.size)) / np.sqrt(2)
        return isi + noise


def _one_trial(channel, snr_db, seed):
    """Returns (tx symbols, received, gains, sigma)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, N_BITS)
    if bits.size % 2:
        bits = bits[:-1]
    x = qpsk_mod(bits)
    y = channel(x, snr_db)
    return x, y, channel.last_gains, 10 ** (-snr_db / 20.0)


def _rates(decoded, x):
    """Symbol error rate and bit error rate at the relay output."""
    di, xi = nearest_symbol(decoded), nearest_symbol(x)
    ser = float(np.mean(di != xi))
    # symbol index = 2*b0 + b1, so bit errors are the Hamming distance of indices
    ber = float(np.mean(((di >> 1) != (xi >> 1)).astype(float)
                        + ((di & 1) != (xi & 1)).astype(float)) / 2)
    return ser, ber


def _sweep(channel_factory, snrs, label):
    out = {}
    for snr in snrs:
        v_ser, v_ber, b_ser, b_ber = [], [], [], []
        for t in range(N_TRIALS):
            ch = channel_factory(1000 + t)
            x, y, g, sigma = _one_trial(ch, snr, 2000 + t)
            vit = FadingAwareViterbiQPSKRelay(channel_taps=H_NORM).set_gains(g).process(y)
            bcjr = (BCJRQPSKRelay(channel_taps=H_NORM, decision="bit")
                    .set_sigma(sigma=sigma).set_gains(g).process(y))
            s, b = _rates(vit, x);  v_ser.append(s);  v_ber.append(b)
            s, b = _rates(bcjr, x); b_ser.append(s);  b_ber.append(b)
        out[str(snr)] = {
            "viterbi_ser": float(np.mean(v_ser)), "viterbi_ber": float(np.mean(v_ber)),
            "bcjr_ser": float(np.mean(b_ser)), "bcjr_ber": float(np.mean(b_ber)),
            "viterbi_ber_per_trial": v_ber, "bcjr_ber_per_trial": b_ber,
        }
        print(f"  {label} {snr:2d} dB  Viterbi BER {np.mean(v_ber):.6f}   "
              f"BCJR BER {np.mean(b_ber):.6f}", flush=True)
    return out


def main():
    print("C1 control -- fading removed; the two detectors must agree:")
    c1 = _sweep(lambda s: NoFadingChannel(H_NORM, seed=s), [8, 20], "C1")

    print("\nC2 control -- no ISI; BER-optimal must be the per-axis slicer:")
    rng = np.random.default_rng(99)
    x = qpsk_mod(rng.integers(0, 2, 20_000))
    y = x + 0.4 * (rng.standard_normal(x.size)
                   + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
    got = (BCJRQPSKRelay(channel_taps=[1.0], channel_len=1)
           .set_sigma(sigma=0.4).process(y))
    slicer = (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)
    c2_ok = bool(np.array_equal(got, slicer))
    print(f"  reduces to the slicer: {c2_ok}")

    print("\nComparison -- faded channel, both detectors given the gains:")
    main_sweep = _sweep(lambda s: ComplexISIRayleighChannel(H_NORM, seed=s),
                        SNRS, "  ")

    out = {
        "snr_convention": "gamma = 10^(SNR_dB/10)",
        "n_trials": N_TRIALS, "n_bits": N_BITS, "taps": H_NORM.tolist(),
        "measured_at": "relay output",
        "controls": {"c1_fading_removed": c1, "c2_no_isi_is_slicer": c2_ok},
        "comparison": main_sweep,
    }
    path = os.path.join(ROOT, "results", "qpsk_bcjr_benchmark.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
