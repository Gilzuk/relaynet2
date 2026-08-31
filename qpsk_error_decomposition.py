"""Where Viterbi MLSE's bit errors come from on the QPSK ISI channel.

Chapter 7 reports a reversal it does not explain: on the QPSK unknown-ISI
channel the 193-parameter learned relay matches or beats genie-CSI Viterbi MLSE
from 2 dB upward, where on the BPSK version of the same channel MLSE led by
1--1.5 dB throughout. The chapter offers a criterion mismatch as a conjecture --
MLSE minimizes *sequence* error probability, BER is what a bit-wise MAP (BCJR)
detector minimizes, and the MLP is trained on a per-symbol softmax objective --
and names two measurements as future work: the BCJR benchmark, and a
decomposition of the Viterbi output into single- versus double-bit symbol
errors. This is the second of those.

WHAT IT MEASURES. Under Gray coding, adjacent QPSK symbols differ in one bit and
diagonal ones in two, so a detector's BER is not determined by its symbol error
rate alone: it also depends on *which* wrong symbol the errors land on. Two
detectors with identical SER differ in BER by up to a factor of two depending on
how many of their errors are diagonal. This script reports, for each detector,
the symbol error rate, the bit error rate, and the split of symbol errors into
one-bit and two-bit cases, so the SER-vs-BER question can be answered from
measurement rather than from the conjecture.

WHAT IT DOES NOT SETTLE. It cannot by itself establish that the criterion
mismatch causes the reversal -- that needs the BCJR benchmark, which is still
future work. A finding that MLSE leads on SER while trailing on BER is
consistent with the conjecture and would rule out "the decoder is simply worse";
a finding that MLSE trails on SER too would rule the conjecture out.

The trellis itself was audited alongside this: on the same channel, MLSE never
returns a path whose metric exceeds that of the true transmitted sequence, and
on an ISI-free channel it agrees with the nearest-symbol slicer on every symbol.
Those are asserted in _selftest.
"""

import json
import os

import numpy as np

from e6_qpsk_unknown_channel import (H_ISI, SNRS, ComplexISIRayleighChannel,
                                     awgn_channel, qpsk_demod, qpsk_mod,
                                     train_mlp_qpsk)
from relaynet.relays.viterbi import ViterbiMLSEQPSKRelay

N_TRIALS, N_BITS = 10, 100_000          # project-standard scale
ALPHABET = ViterbiMLSEQPSKRelay.ALPHABET


def nearest_symbol(sym):
    """Index of the closest alphabet point to each sample."""
    return np.argmin(np.abs(np.asarray(sym)[:, None] - ALPHABET[None, :]), axis=1)


def decompose(relay, hop1, n_bits, snr_db, seed):
    """Symbol/bit error rates and the 1-bit / 2-bit split, at the relay output.

    Measured at the relay rather than end to end: hop 2 adds its own errors,
    which would blur exactly the distinction being measured.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    if bits.size % 2:
        bits = bits[:-1]
    x = qpsk_mod(bits)

    hop1.rng = np.random.default_rng(seed + 101)
    y = hop1(x, snr_db)
    out = y if relay is None else relay.process(y)

    tx_idx = nearest_symbol(x)
    rx_idx = nearest_symbol(out)
    wrong = tx_idx != rx_idx

    # bit errors per symbol, from the Gray map the alphabet already encodes
    tx_bits = qpsk_demod(ALPHABET[tx_idx]).reshape(-1, 2)
    rx_bits = qpsk_demod(ALPHABET[rx_idx]).reshape(-1, 2)
    per_symbol = (tx_bits != rx_bits).sum(axis=1)

    n = tx_idx.size
    n_wrong = int(wrong.sum())
    return {
        "ser": n_wrong / n,
        "ber": float(per_symbol.sum()) / (2 * n),
        "n_symbols": n, "n_symbol_errors": n_wrong,
        "one_bit": int((per_symbol == 1).sum()),
        "two_bit": int((per_symbol == 2).sum()),
        "frac_two_bit": (float((per_symbol == 2).sum()) / n_wrong
                         if n_wrong else float("nan")),
        # bits lost per symbol error: 1.0 if every error is adjacent, 2.0 if
        # every error is diagonal. This is the whole SER -> BER conversion.
        "bits_per_symbol_error": (float(per_symbol.sum()) / n_wrong
                                  if n_wrong else float("nan")),
    }


def _selftest():
    """The trellis audit: optimality on the ISI channel, identity without ISI."""
    h = H_ISI / np.linalg.norm(H_ISI)
    rng = np.random.default_rng(0)

    v1 = ViterbiMLSEQPSKRelay(channel_taps=[1.0, 0.0, 0.0])
    x = ALPHABET[rng.integers(0, 4, 2000)]
    y = x + (rng.normal(0, .3, 2000) + 1j * rng.normal(0, .3, 2000))
    assert np.array_equal(nearest_symbol(v1.process(y)), nearest_symbol(y)), \
        "with no ISI, MLSE must agree with the nearest-symbol slicer"
    print("  no ISI: MLSE == nearest-symbol slicer on every symbol: OK")

    v = ViterbiMLSEQPSKRelay(channel_taps=h)
    L = len(h)

    def metric(seq, obs):
        """Path metric over indices L-1.. only.

        The first L-1 terms depend on a pre-history the decoder never sees.
        Viterbi starts every state at metric 0, so it is free to assume any
        pre-history of alphabet symbols; the true sequence's pre-history is one
        of those, but the *zero* pre-history the transmitter used is not in the
        alphabet and so is not a path the trellis can represent. Scoring from
        index L-1 removes that mismatch and compares the two on the terms both
        models actually determine.
        """
        exp = np.array([np.dot(h, seq[i - L + 1:i + 1][::-1])
                        for i in range(L - 1, len(seq))])
        return float(np.sum(np.abs(obs[L - 1:] - exp) ** 2))

    worst = -np.inf
    for _ in range(20):
        x = ALPHABET[rng.integers(0, 4, 300)]
        xp = np.concatenate([np.zeros(L - 1, complex), x])
        clean = np.array([np.dot(h, xp[i:i + L][::-1]) for i in range(len(x))])
        for snr in (0, 6, 12):
            sig = np.sqrt(1 / (2 * 10 ** (snr / 10)))
            obs = clean + (rng.normal(0, sig, len(x))
                           + 1j * rng.normal(0, sig, len(x)))
            worst = max(worst, metric(v.process(obs), obs) - metric(x, obs))
    assert worst <= 1e-9, f"MLSE returned a less likely path (excess {worst:.3e})"
    print(f"  3-tap ISI: MLSE path never less likely than the true sequence "
          f"(max excess {worst:.1e}): OK")
    return True


def main():
    _selftest()
    hop1 = ComplexISIRayleighChannel(H_ISI, seed=1)
    mlp = train_mlp_qpsk(ComplexISIRayleighChannel(H_ISI, seed=3), seed=0)
    detectors = {
        "Viterbi (genie CSI)": ViterbiMLSEQPSKRelay(channel_taps=H_ISI),
        f"MLP-QPSK ({mlp.n_params()}p)": mlp,
    }

    out = {"snr_db": [int(s) for s in SNRS], "n_trials": N_TRIALS,
           "n_bits": N_BITS, "taps": H_ISI.tolist(), "detectors": {}}
    print(f"\n{N_TRIALS} trials x {N_BITS} bits per SNR point, relay output\n")
    for name, relay in detectors.items():
        print(f"  === {name} ===")
        print(f"  {'SNR':>4} {'SER':>10} {'BER':>10} {'2-bit frac':>11} "
              f"{'bits/err':>9}")
        rows = []
        for snr in SNRS:
            per = [decompose(relay, hop1, N_BITS, int(snr), 1000 + t)
                   for t in range(N_TRIALS)]
            agg = {k: float(np.mean([p[k] for p in per]))
                   for k in ("ser", "ber", "frac_two_bit",
                             "bits_per_symbol_error")}
            agg["snr_db"] = int(snr)
            agg["n_symbol_errors"] = int(sum(p["n_symbol_errors"] for p in per))
            rows.append(agg)
            print(f"  {snr:>4} {agg['ser']:>10.5f} {agg['ber']:>10.5f} "
                  f"{agg['frac_two_bit']:>11.4f} "
                  f"{agg['bits_per_symbol_error']:>9.4f}", flush=True)
        out["detectors"][name] = rows

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "qpsk_error_decomposition.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWritten to {dest}")
    return out


if __name__ == "__main__":
    main()
