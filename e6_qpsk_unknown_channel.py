#!/usr/bin/env python3
"""E6 QPSK unknown-channel study: ISI hop 1, AWGN *or* Rayleigh hop 2.

Mirrors the structure of the BPSK unknown-channel study (`e6_sim_ported.py`,
Chapter 7) but for QPSK, so the two modulations can be compared row by row:

    Hop 1 : unknown 3-tap ISI + coherently-compensated Rayleigh + AWGN
    Hop 2 : AWGN  or  ISI-free Rayleigh, at the same per-hop SNR
    Relays: AF, symbol-wise DF (hard), 4-class MLP-QPSK classifier (learned),
            Viterbi MLSE with genie CSI (optimal sequence detector)

Why this exists
---------------
The committed QPSK results (`e6_qpsk_rescaled_results/`) evaluate *symmetric*
hops only -- both hops carry ISI+Rayleigh -- and omit a learned relay from the
QPSK set. Chapter 7's BPSK study instead reports two hop-2 variants (AWGN and
Rayleigh) and centres on the learned relay. This script supplies the missing
QPSK counterpart on both axes.

Note on the analytic floor
--------------------------
The 0.25 memoryless-relay BER floor derived in Chapter 7 is specific to BPSK:
it counts the fraction of the four equiprobable ISI patterns whose combined
amplitude flips the slicer's sign. QPSK applies that argument independently per
I/Q axis over a larger symbol alphabet, so the BPSK figure does NOT carry over
and no floor is asserted here -- the empirical curves are reported as measured.

SNR convention: gamma = 10^(SNR_dB/10), per memory-bank/techContext.md.

Output: e6_unknown_channel_results/e6_qpsk_unknown_channel_results.npy
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relaynet.relays import AmplifyAndForwardRelay, ViterbiMLSEQPSKRelay
from relaynet.relays.viterbi import FadingAwareViterbiQPSKRelay
from relaynet.channels import ComplexISIRayleighChannel, awgn_channel
from e6_sim_enhanced_multimod import DFHardRelay
from e6_mlp_qpsk_vs_viterbi import train_mlp_qpsk, W

# ── configuration ───────────────────────────────────────────────────
H_ISI = np.array([1.0, 0.7, 0.5])
SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 10, 100_000          # project-standard scale
TRAIN_SNRS = [5, 10, 15]
OUT_DIR = "e6_unknown_channel_results"


def qpsk_mod(bits):
    """Gray-coded QPSK, unit average power."""
    return ((1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2])) / np.sqrt(2)


def qpsk_demod(sym):
    """Hard per-axis decision back to a bit vector."""
    b = np.empty(2 * sym.size, dtype=int)
    b[0::2] = (sym.real < 0).astype(int)
    b[1::2] = (sym.imag < 0).astype(int)
    return b


def make_hop2(kind, seed):
    """Hop-2 channel: AWGN, or ISI-free Rayleigh (single unit tap)."""
    if kind == "awgn":
        return awgn_channel
    if kind == "rayleigh":
        return ComplexISIRayleighChannel(np.array([1.0]), seed=seed)
    raise ValueError(kind)


def run_trial(relay, hop1, hop2, n_bits, snr_db, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    if bits.size % 2:
        bits = bits[:-1]
    x = qpsk_mod(bits)

    hop1.rng = np.random.default_rng(seed + 101)
    y_relay = hop1(x, snr_db)

    # A genie-CSI detector on this channel needs the per-symbol fading gains as
    # well as the taps -- hop 1 is g[n] * conv(x, h) + v, and a trellis told
    # only h is model-mismatched, not genie. See FadingAwareViterbiQPSKRelay.
    if isinstance(relay, FadingAwareViterbiQPSKRelay):
        relay.set_gains(getattr(hop1, "last_gains", None))

    x_relay = y_relay if relay is None else relay.process(y_relay)
    # power-normalise the relay output so all strategies transmit equal power
    p = np.sqrt(np.mean(np.abs(x_relay) ** 2)) + 1e-12
    x_relay = x_relay / p

    if hop2 is awgn_channel:
        y_dest = awgn_channel(x_relay, snr_db)
    else:
        hop2.rng = np.random.default_rng(seed + 202)
        y_dest = hop2(x_relay, snr_db)

    return float(np.mean(qpsk_demod(y_dest) != bits))


def run_setup(hop2_kind, mlp):
    print(f"\n=== QPSK: unknown ISI -> {hop2_kind.upper()} hop 2 ===")
    hop1 = ComplexISIRayleighChannel(H_ISI, seed=1)
    hop2 = make_hop2(hop2_kind, seed=2)

    relays = {
        "AF": AmplifyAndForwardRelay(target_power=1.0),
        "DF": DFHardRelay("qpsk"),
        "MLP-QPSK": mlp,
        # Renamed for what it is: given the ISI taps only. On this channel that
        # is not genie CSI, and the original label was wrong.
        "Viterbi (taps only)": ViterbiMLSEQPSKRelay(channel_taps=H_ISI),
        "Viterbi (genie CSI)": FadingAwareViterbiQPSKRelay(channel_taps=H_ISI),
    }

    out = {k: np.zeros((len(SNRS), N_TRIALS)) for k in relays}
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed = 9000 * si + tr
            for name, r in relays.items():
                out[name][si, tr] = run_trial(r, hop1, hop2, N_BITS, snr, seed)
        print(f"  {snr:2d} dB  " + "  ".join(
            f"{n}={out[n][si].mean():.4f}" for n in relays))

    summary = {}
    for n in relays:
        mu = out[n].mean(axis=1)
        ci = 1.96 * out[n].std(axis=1) / np.sqrt(N_TRIALS)
        summary[n] = (mu, ci)
    return summary


def plot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (kind, summary) in zip(axes, results.items()):
        for name, (mu, ci) in summary.items():
            ax.semilogy(SNRS, np.maximum(mu, 1e-5), marker="o", label=name)
            ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6),
                            np.maximum(mu + ci, 1e-6), alpha=0.15)
        ax.set_title(f"QPSK: unknown ISI $\\to$ {kind.upper()} hop 2")
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nFigure -> {path}")


def main():
    t0 = time.time()
    print("=" * 78)
    print("E6 QPSK UNKNOWN CHANNEL — ISI hop 1, AWGN / Rayleigh hop 2")
    print(f"{N_TRIALS} trials x {N_BITS} bits per SNR point, SNR {SNRS[0]}-{SNRS[-1]} dB")
    print("=" * 78)

    print("\nTraining 4-class MLP-QPSK classifier on the ISI family...")
    mlp = train_mlp_qpsk(ComplexISIRayleighChannel(H_ISI, seed=3), seed=0)
    print(f"  MLP-QPSK: {mlp.n_params()} parameters (window={W})")

    results = {k: run_setup(k, mlp) for k in ("awgn", "rayleigh")}

    os.makedirs(OUT_DIR, exist_ok=True)
    npy = os.path.join(OUT_DIR, "e6_qpsk_unknown_channel_results.npy")
    np.save(npy, {"snrs": SNRS, "results": results,
                  "n_trials": N_TRIALS, "n_bits": N_BITS,
                  "mlp_params": mlp.n_params(), "taps": H_ISI},
            allow_pickle=True)
    print(f"\nResults -> {npy}")
    plot(results, os.path.join(OUT_DIR, "unkchan_qpsk.png"))
    print(f"\nElapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
