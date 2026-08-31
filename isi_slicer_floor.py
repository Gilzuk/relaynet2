"""Closed-form BER of a memoryless slicer on the 3-tap ISI channel.

Chapter 7 asserts that symbol-wise DF on the unknown-ISI channel has an
irreducible BER floor of 0.25, and that it is approached from below, so more
transmit power makes DF strictly worse above about 6 dB. The thesis stated the
asymptote and left the rest of the curve to the simulation. It does not have to:
the whole curve is available in closed form, and having it turns the measured DF
column from an assertion into something with an independent prediction to agree
with.

DERIVATION. With BPSK symbols x[n] in {+1,-1}, i.i.d. and equiprobable, and a
real 3-tap channel h (unit norm), the relay observes

    y[n] = h0 x[n] + h1 x[n-1] + h2 x[n-2] + v[n],   v ~ N(0, sigma^2)

and a memoryless slicer returns sign(y[n]). Conditioned on x[n] = +1, the two
interfering symbols take one of 2^2 = 4 equiprobable sign patterns, each giving
an effective amplitude

    A(s1, s2) = h0 + s1 h1 + s2 h2.

An error is the event y[n] < 0, so

    Pe = (1/4) sum over the four patterns of Phi(-A / sigma),

with Phi the standard normal CDF and sigma^2 = 1/(2 gamma) per real dimension,
gamma = 10^(SNR_dB/10) -- the convention used throughout this work.

THE FLOOR. As sigma -> 0 each term goes to 0 if A > 0 and to 1 if A < 0, so the
limit is the *fraction of interference patterns that flip the sign of the
symbol*. For h = [1.0, 0.7, 0.5] normalized the four amplitudes are 1.6678,
0.9097, 0.6065 and -0.1516: exactly one is negative, hence 1/4.

WHAT IS AND IS NOT GENERAL. The value 0.25 is a property of this tap vector, not
of ISI. The floor is k/4 where k is how many of the four amplitudes are negative,
so a different 3-tap channel gives 0, 0.25, 0.5 or 0.75; a channel with
sum|h_i| < 2 h0 has no floor at all. What *is* general for a memoryless relay is
that the limit is a fixed fraction independent of SNR, so transmit power cannot
reach it -- which is the claim Chapter 7 actually rests on.

The model ignores hop-2 noise, so it is a relay-side prediction and is expected
to sit below the measured end-to-end BER at low SNR, where the second hop still
contributes. Above roughly 6 dB the second hop is clean and the two coincide.
"""

import itertools
import json
import os

import numpy as np
from scipy.stats import norm

# The Chapter 7 unknown-ISI channel, before normalization.
H_RAW = np.array([1.0, 0.7, 0.5])
SNRS = list(range(0, 21, 2))


def effective_amplitudes(h):
    """The 2^(L-1) effective slicer amplitudes h0 + sum s_i h_i, s_i = +-1."""
    h = np.asarray(h, dtype=float)
    return np.array([h[0] + np.dot(s, h[1:])
                     for s in itertools.product([1.0, -1.0], repeat=len(h) - 1)])


def slicer_ber(h, snr_db):
    """Closed-form memoryless-slicer BER at `snr_db`, thesis SNR convention."""
    amps = effective_amplitudes(h)
    sigma = np.sqrt(1.0 / (2.0 * 10 ** (np.asarray(snr_db, float) / 10.0)))
    return np.mean(norm.cdf(-amps[:, None] / sigma[None, :]), axis=0)


def floor(h):
    """The SNR -> infinity limit: the fraction of amplitudes that flip sign."""
    amps = effective_amplitudes(h)
    return float(np.mean(amps < 0))


def main():
    h = H_RAW / np.linalg.norm(H_RAW)
    amps = effective_amplitudes(h)
    ber = slicer_ber(h, SNRS)
    out = {"taps_raw": H_RAW.tolist(), "taps_normalized": h.tolist(),
           "effective_amplitudes": amps.tolist(),
           "floor": floor(h), "snr_db": SNRS,
           "slicer_ber": ber.tolist()}

    print(f"taps (normalized): {np.round(h, 4).tolist()}")
    print(f"effective amplitudes: {np.round(amps, 4).tolist()}")
    print(f"sign-flipping patterns: {int((amps < 0).sum())} of {amps.size} "
          f"-> floor {floor(h)}")
    print("\n  SNR dB   closed-form slicer BER")
    for s, b in zip(SNRS, ber):
        print(f"  {s:>6d}   {b:.4f}")

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "isi_slicer_floor.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWritten to {dest}")
    return out


if __name__ == "__main__":
    main()
