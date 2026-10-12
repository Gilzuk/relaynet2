"""Canonical AF: block-length sensitivity, and a conventional finite-power AF.

Dr. Khina's review (comment 3): the canonical AF relay forwards the zero-forced
observation y/h = x + n/h and normalizes it by the empirical power of the whole
block. Because E[1/|h|^2] is infinite under Rayleigh fading, the forwarded power
has no finite mean, so the block gain does not converge and the most deeply
faded symbols set it for the rest of the block. He asked for a block-length
sensitivity check and a conventional finite-power AF reference with the
appropriate destination processing. This script provides both, on the
canonical QPSK / i.i.d. Rayleigh / two-hop setup and SNR axis.

(a) Implemented AF (relaynet AmplifyAndForwardRelay through
    relaynet.simulation.runner with rayleigh_fading_channel, exactly the
    canonical pipeline) at block lengths of 50, 500, 5,000 and 50,000 QPSK
    symbols, each with 1,000,000 bits per SNR point. 5,000 symbols is the
    canonical block (10,000 bits).
(b) Conventional AF. The relay forwards the unequalized r = h1 x + n1 scaled
    to a finite power: variable gain G = sqrt(P / (|h1|^2 + N0)) with
    instantaneous CSI (the textbook CSI-assisted AF), or fixed gain
    G = sqrt(P / (E|h1|^2 + N0)). The destination knows h1, h2 and G and
    detects coherently, z = d / (h2 G h1), followed by the QPSK slicer; the
    end-to-end noise is circular, so this is the symbol-wise ML rule.
    1,000,000 bits per SNR point.
    The variable-gain BER is checked against the closed form averaged over
    the fades: E[ Q( sqrt( g1 g2 / (g1 + g2 + 1) ) ) ], g_i = |h_i|^2 gamma.
(c) Symbol-wise DF through the same canonical pipeline, to confirm this script
    reproduces the committed canonical DF curve before the committed learned-
    relay figures (results/modulation/qpsk_rayleigh.json) are used beside it.

SNR convention: gamma = 10^(SNR_dB/10) = Es/N0 per hop, complex noise N0/2 per
real dimension, as rayleigh_fading_channel. Writes
results/af_block_and_conventional.json. Run: python af_block_and_conventional.py
"""

import json
import os
import subprocess

import numpy as np
from scipy.special import erfc

from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
from relaynet.simulation.runner import simulate_transmission

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "af_block_and_conventional.json")
SNRS = list(range(0, 21, 2))
TOTAL_BITS = 1_000_000
BLOCK_SYMBOLS = (50, 500, 5000, 50000)
Q = lambda x: 0.5 * erfc(x / np.sqrt(2))


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def canonical_ber(relay, snr, block_symbols, seed0):
    """Canonical pipeline, TOTAL_BITS split into blocks of block_symbols."""
    bits_per_block = 2 * block_symbols
    n_blocks = TOTAL_BITS // bits_per_block
    errors = 0
    for k in range(n_blocks):
        np.random.seed(seed0 + k)            # rayleigh_fading_channel uses np.random
        _, n_err = simulate_transmission(relay, bits_per_block, snr, seed=seed0 + k,
                                         channel_fn=rayleigh_fading_channel,
                                         modulation="qpsk")
        errors += int(n_err)
    return errors / (n_blocks * bits_per_block)


def conventional_af(snr, variable_gain, rng):
    g = 10 ** (snr / 10)
    N0 = 1.0 / g
    n = TOTAL_BITS // 2
    bits = rng.integers(0, 2, 2 * n)
    x = qpsk_modulate(bits)
    cn = lambda: (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    h1, h2 = cn(), cn()
    r = h1 * x + np.sqrt(N0) * cn()
    G = np.sqrt(1.0 / (np.abs(h1) ** 2 + N0)) if variable_gain else np.sqrt(1.0 / (1.0 + N0))
    d = h2 * G * r + np.sqrt(N0) * cn()
    z = d / (h2 * G * h1)
    xh = (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)
    xb = np.empty(2 * n, dtype=int)
    xb[0::2] = (xh.real < 0); xb[1::2] = (xh.imag < 0)
    return float(np.mean(xb != bits)), float(np.mean(G ** 2 * (np.abs(h1) ** 2 + N0)))


def variable_gain_closed_form(snr, rng, draws=4_000_000):
    g = 10 ** (snr / 10)
    g1 = g * rng.exponential(size=draws)
    g2 = g * rng.exponential(size=draws)
    return float(np.mean(Q(np.sqrt(g1 * g2 / (g1 + g2 + 1)))))


def main():
    rng = np.random.default_rng(20261012)
    out = {"snrs": SNRS, "total_bits_per_point": TOTAL_BITS,
           "implemented_af_by_block": {}, "conventional_af_variable_gain": [],
           "conventional_af_fixed_gain": [], "variable_gain_closed_form": [],
           "relay_output_power_variable_gain": [], "df_canonical_pipeline": []}
    for N in BLOCK_SYMBOLS:
        out["implemented_af_by_block"][str(N)] = [
            canonical_ber(AmplifyAndForwardRelay(target_power=1.0, prefer_gpu=False),
                          s, N, seed0=1_000_000 * (1 + SNRS.index(s))) for s in SNRS]
        print(f"  implemented AF, block {N:>6}: " +
              " ".join(f"{b:.4f}" for b in out["implemented_af_by_block"][str(N)]))
    for s in SNRS:
        bv, pv = conventional_af(s, True, rng)
        bf, _ = conventional_af(s, False, rng)
        out["conventional_af_variable_gain"].append(bv)
        out["conventional_af_fixed_gain"].append(bf)
        out["relay_output_power_variable_gain"].append(pv)
        out["variable_gain_closed_form"].append(variable_gain_closed_form(s, rng))
        out["df_canonical_pipeline"].append(
            canonical_ber(DecodeAndForwardRelay(target_power=1.0, prefer_gpu=False), s, 5000,
                          seed0=7_000_000 * (1 + SNRS.index(s))))
    print("  conventional AF (variable): " + " ".join(f"{b:.4f}" for b in out["conventional_af_variable_gain"]))
    print("  closed form (variable)    : " + " ".join(f"{b:.4f}" for b in out["variable_gain_closed_form"]))
    print("  conventional AF (fixed)   : " + " ".join(f"{b:.4f}" for b in out["conventional_af_fixed_gain"]))
    print("  DF, canonical pipeline    : " + " ".join(f"{b:.4f}" for b in out["df_canonical_pipeline"]))
    committed = json.load(open(os.path.join(ROOT, "results", "modulation", "qpsk_rayleigh.json")))
    out["committed_canonical_reference"] = {
        k: committed["results"][k]["ber_mean"] for k in ("AF", "DF", "MLP (169p)", "Hybrid", "VAE")}
    out["git_rev"] = git_rev()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2); fh.write("\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=TOTAL_BITS)
    a = ap.parse_args()
    TOTAL_BITS = a.bits
    main()
