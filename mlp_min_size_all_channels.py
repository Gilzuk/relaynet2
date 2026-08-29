"""Minimum MLP relay size that matches DF, on every channel family in the thesis.

This generalizes mlp_min_size_rayleigh.py. That script answered the question on
the canonical Rayleigh channel and found the floor at 7 parameters (window 1,
hidden 2) -- and, unexpectedly, that on a memoryless channel a *window* is
actively harmful: penalty vs DF grew with both window and hidden size, and the
canonical 169-parameter relay was the worst configuration in the grid.

That result only makes sense if the floor is set by channel memory. A
memoryless channel makes y_R[i] a sufficient statistic for x[i], so window taps
carry no information and capacity spent on them fits noise. A channel with
memory inverts the argument: the taps carry the information needed to undo the
interference, and a window-1 relay cannot in principle equalize it. The
prediction is therefore a crossover -- window 1 wins on memoryless channels,
loses on channels with memory, and the minimum size tracks channel memory.

This script tests that across every channel family the thesis uses.

VEHICLE. Chapter 7 experiments use MLPRelay (relaynet/relays/mlp.py), not
MinimalGenAIRelay, and MLPRelay is the better instrument here for three
reasons: it is pure NumPy and seeded by construction (default_rng(seed)), so
it is reproducible without the torch-seeding workaround; it trains on the
actual channel via train_on_data() rather than on AWGN surrogate data, which
is what "can a network of this size learn this channel" has to mean; and it
is what the thesis's own unknown-channel chapter uses.

    params = hidden * (window + 2) + 1

is the same formula as the per-axis relay in the Rayleigh study, so parameter
counts are directly comparable between the two.

WHAT IS HELD FIXED. Every channel is run with the same grid, the same training
budget, the same Monte Carlo budget, and the same DF comparator, measured in
the same process on identical channel draws. The only variables are channel
and relay size.

BASELINE CAVEAT, stated rather than buried. DF is the comparator on every
channel, which keeps the study internally consistent and keeps "matches DF"
meaning one thing. But DF is a symbol-by-symbol slicer, so on the ISI channels
it is a *weak* baseline -- it cannot equalize either, and the thesis compares
against Viterbi/MLSE there (Chapter 7). A small relay "matching DF" on an ISI
channel therefore means it matches a slicer, not that it matches the classical
state of the art. Read the ISI rows as a memory-vs-size measurement, not as a
claim about MLSE.

SEEDING. Three RNGs, all pinned (the lesson from the Rayleigh sweep):
  1. Payload bits   -- run_monte_carlo seeds Source(seed=seed_offset+trial).
  2. Channel draws  -- the fading channels draw from the *global* numpy RNG
     (techContext gotcha), so np.random.seed() is re-applied before every
     relay's evaluation. DF and every MLP therefore see identical draws.
  3. Network init   -- MLPRelay(seed=...) seeds a local default_rng; batch
     shuffling inside train_on_data uses a fixed default_rng(42). Both are
     deterministic. Initialization is still treated as a variable: every
     configuration is trained under TRAIN_SEEDS independent inits and only
     counts as matching if it matches under all of them.

Two criteria are reported side by side, as in the Rayleigh study:
  tolerance : mean BER penalty vs DF <= TOL_REL at every SNR, every seed.
  Wilcoxon  : paired signed-rank on per-trial BER vs DF; a configuration loses
              at an SNR if p < ALPHA and the MLP is the worse of the pair.

SNR convention follows memory-bank/techContext.md: gamma = 10^(SNR_dB/10).
"""

import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relaynet.channels import (
    awgn_channel,
    rayleigh_fading_channel,
    ISIChannel,
    ComplexISIChannel,
    ComplexISIRayleighChannel,
    NonlinearBiasChannel,
    FlatPhaseChannel,
    FlatGainChannel,
    BranchAsymmetryChannel,
    PowerAmplifierChannel,
)
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, MLPRelay
from relaynet.simulation.runner import run_monte_carlo

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 20
BITS_PER_TRIAL = 20000
TRAIN_SNRS = [5, 10, 15]
TRAIN_SAMPLES = 60000          # per axis, split across TRAIN_SNRS
EPOCHS = 25                    # the Chapter 7 recipe (e6_sim_ported.train_mlp)
BATCH = 256
LR = 3e-3
SEED = 0                       # bits + channel; identical for every relay
TRAIN_SEEDS = [0, 1, 2]        # independent inits per configuration
CHANNEL_SEED = 1               # the seed the E6 scripts use for hop-1 channels

TOL_REL = 0.02
ALPHA = 0.05

# The normalized 3-tap response used throughout the Chapter 7 experiments
# (e6_sim_enhanced.py, e6_viterbi_qpsk.py, e6_relay_comparison_symmetric.py).
# Note e6_sim_ported.py uses the same taps *unnormalized*; the normalized form
# is used here because it preserves symbol power and so keeps the SNR
# convention meaning the same thing on this channel as on every other.
H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)

# (window, hidden). Window 7 is included beyond the canonical maximum of 5
# because the ISI channels have 3 taps over two hops and may need the reach.
GRID = [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16),
    (3, 1), (3, 2), (3, 4), (3, 8), (3, 16),
    (5, 1), (5, 2), (5, 4), (5, 8), (5, 24),   # (5,24) = MLP-169, the control
    (7, 4), (7, 8),
]

# Channel families. "memory" is the number of taps in the channel's impulse
# response: 1 means memoryless, so the crossover prediction applies to it.
CHANNELS = {
    "awgn": dict(
        make=lambda s: awgn_channel, mod="bpsk", memory=1,
        note="Ch5 calibration reference (closed-form BER)"),
    "rayleigh": dict(
        make=lambda s: rayleigh_fading_channel, mod="qpsk", memory=1,
        note="Ch5 canonical operating point"),
    "flat_phase": dict(
        make=lambda s: FlatPhaseChannel(seed=s), mod="qpsk", memory=1,
        note="Ch7 E6_FLAT unknown phase, theta ~ U[0,2pi)"),
    "flat_gain": dict(
        make=lambda s: FlatGainChannel(gain_min=0.3, gain_max=2.0, seed=s),
        mod="bpsk", memory=1,
        note="Ch7 E6_FLAT unknown gain, g ~ U[0.3,2.0]"),
    "branch_asym": dict(
        make=lambda s: BranchAsymmetryChannel(seed=s), mod="bpsk", memory=1,
        note="Ch7 E6_FLAT branch asymmetry, a+- ~ U[0.6,1.4]"),
    "nlbias": dict(
        make=lambda s: NonlinearBiasChannel(saturation=1.5, dc_bias=0.5, seed=s),
        mod="bpsk", memory=1,
        note="Ch7 nonlinear saturation + DC bias (memoryless but nonlinear)"),
    "pa": dict(
        make=lambda s: PowerAmplifierChannel(saturation=1.2, seed=s),
        mod="qpsk", memory=1,
        note="Ch7 power-amplifier saturation (memoryless but nonlinear)"),
    "isi": dict(
        make=lambda s: ISIChannel(H_ISI, seed=s), mod="bpsk", memory=3,
        note="Ch7 real 3-tap ISI"),
    "isi_complex": dict(
        make=lambda s: ComplexISIChannel(H_ISI, seed=s), mod="qpsk", memory=3,
        note="Ch7 complex 3-tap ISI"),
    "isi_rayleigh": dict(
        make=lambda s: ComplexISIRayleighChannel(H_ISI, seed=s), mod="qpsk",
        memory=3, note="Ch7 3-tap ISI on top of Rayleigh fading"),
}


def n_params(window, hidden):
    return hidden * (window + 2) + 1


def make_training_data(channel, mod, window, rng):
    """Build (X, target) for train_on_data, using the actual channel.

    Mirrors how the runner applies the relay: a real channel feeds the network
    one real stream; a complex channel is split into its I and Q axes and the
    same real-valued network is trained on both, which is exactly how it is
    applied at test time (runner.py, _apply_relay).
    """
    per_snr = TRAIN_SAMPLES // len(TRAIN_SNRS)
    pad = window // 2
    X_list, T_list = [], []

    for snr in TRAIN_SNRS:
        if mod == "bpsk":
            x = 1.0 - 2.0 * rng.integers(0, 2, per_snr).astype(float)
            y = channel(x, snr)
            if isinstance(y, tuple):
                y = y[0]
            axes = [(np.real(y) if np.iscomplexobj(y) else y, x)]
        else:  # qpsk, per-axis at +-1/sqrt(2)
            b = rng.integers(0, 2, (per_snr, 2))
            xr = (1.0 - 2.0 * b[:, 0]) / np.sqrt(2.0)
            xi = (1.0 - 2.0 * b[:, 1]) / np.sqrt(2.0)
            y = channel(xr + 1j * xi, snr)
            if isinstance(y, tuple):
                y = y[0]
            y = np.asarray(y)
            if np.iscomplexobj(y):
                axes = [(y.real, xr), (y.imag, xi)]
            else:
                axes = [(y, xr)]
        for yy, tt in axes:
            yp = np.pad(np.asarray(yy, dtype=float), (pad, pad), mode="constant")
            X_list.append(sliding_window_view(yp, window))
            T_list.append(np.asarray(tt, dtype=float))

    return np.vstack(X_list), np.concatenate(T_list)


def evaluate(relay, channel, mod, tag):
    """Monte Carlo from a fixed global RNG state, so every relay on a given
    channel sees identical fading and noise realizations."""
    np.random.seed(SEED % (2 ** 31))
    _, ber, trials = run_monte_carlo(
        relay, SNRS,
        num_bits_per_trial=BITS_PER_TRIAL,
        num_trials=N_TRIALS,
        channel_fn=channel,
        modulation=mod,
        seed_offset=SEED,
    )
    if tag:
        print(f"    {tag:<30} " + "  ".join(f"{b:.4f}" for b in ber), flush=True)
    return np.asarray(ber), np.asarray(trials)


def compare(ber, trials, df_ber, df_trials):
    per_snr = []
    for i, snr in enumerate(SNRS):
        d = trials[i] - df_trials[i]
        pval = 1.0 if np.allclose(d, 0) else float(wilcoxon(trials[i], df_trials[i])[1])
        # guard against a DF BER of exactly zero at high SNR
        rel = float((ber[i] - df_ber[i]) / df_ber[i]) if df_ber[i] > 0 else float("nan")
        per_snr.append({
            "snr_db": snr,
            "mlp_ber": float(ber[i]), "df_ber": float(df_ber[i]),
            "rel_penalty": rel, "wilcoxon_p": pval,
            "wilcoxon_loses": bool(pval < ALPHA and (rel > 0 if rel == rel else False)),
            "wins": int(np.sum(d < 0)), "losses": int(np.sum(d > 0)),
        })
    return per_snr


def run_channel(name, spec):
    print(f"\n{'=' * 78}\n  {name}   [{spec['note']}]   "
          f"modulation {spec['mod']}, channel memory {spec['memory']} tap(s)")
    print(f"{'=' * 78}")
    print("  " + " " * 30 + "  ".join(f"{s:>6}dB" for s in SNRS))

    channel = spec["make"](CHANNEL_SEED)
    mod = spec["mod"]

    print("\n  classical baselines")
    df_ber, df_trials = evaluate(DecodeAndForwardRelay(), channel, mod, "DF (0 params)")
    af_ber, _ = evaluate(AmplifyAndForwardRelay(), channel, mod, "AF (0 params)")

    print("\n  MLP sweep")
    rows = []
    for window, hidden in GRID:
        p = n_params(window, hidden)
        seed_runs = []
        for ts in TRAIN_SEEDS:
            rng = np.random.default_rng(1000 + ts)
            X, T = make_training_data(channel, mod, window, rng)
            relay = MLPRelay(input_size=window, hidden_size=hidden,
                             output_size=1, window_size=window, seed=ts)
            relay.train_on_data(X, T, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            ber, trials = evaluate(relay, channel, mod, None)
            per_snr = compare(ber, trials, df_ber, df_trials)
            finite = [r["rel_penalty"] for r in per_snr if r["rel_penalty"] == r["rel_penalty"]]
            worst = max(finite) if finite else float("nan")
            seed_runs.append({
                "train_seed": ts,
                "ber": [float(b) for b in ber],
                "worst_rel_penalty": float(worst),
                "tolerance_ok": bool(worst <= TOL_REL),
                "wilcoxon_ok": not any(r["wilcoxon_loses"] for r in per_snr),
                "per_snr": per_snr,
            })

        worst = max(s["worst_rel_penalty"] for s in seed_runs)
        best = min(s["worst_rel_penalty"] for s in seed_runs)
        tol_all = all(s["tolerance_ok"] for s in seed_runs)
        wil_all = all(s["wilcoxon_ok"] for s in seed_runs)
        rows.append({
            "window": window, "hidden": hidden, "params": p,
            "worst_rel_penalty_over_seeds": float(worst),
            "best_rel_penalty_over_seeds": float(best),
            "matches_tolerance_all_seeds": bool(tol_all),
            "matches_wilcoxon_all_seeds": bool(wil_all),
            "seed_runs": seed_runs,
        })
        print(f"    w={window} h={hidden} ({p}p)".ljust(24)
              + f"penalty {100*best:+7.1f}% .. {100*worst:+7.1f}%"
              + f"   tol {'ok' if tol_all else 'NO'}"
              + f"   wilcoxon {'ok' if wil_all else 'NO'}", flush=True)

    tol_match = [r for r in rows if r["matches_tolerance_all_seeds"]]
    both = [r for r in rows if r["matches_tolerance_all_seeds"]
            and r["matches_wilcoxon_all_seeds"]]
    result = {
        "note": spec["note"], "modulation": mod, "memory": spec["memory"],
        "df_ber": [float(b) for b in df_ber], "af_ber": [float(b) for b in af_ber],
        "sweep": rows,
        "min_params_tolerance": min((r["params"] for r in tol_match), default=None),
        "min_params_both_criteria": min((r["params"] for r in both), default=None),
        "best_config_both": (min(both, key=lambda r: r["params"])
                             if both else None),
    }
    if both:
        b = min(both, key=lambda r: r["params"])
        print(f"\n  -> smallest passing both: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    elif tol_match:
        b = min(tol_match, key=lambda r: r["params"])
        print(f"\n  -> none passes both; smallest within tolerance: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    else:
        print("\n  -> no configuration in the grid matched DF")
    return result


def main():
    only = sys.argv[1:] or list(CHANNELS)
    print(f"MLP minimum-size study across thesis channel families")
    print(f"SNRs {SNRS} | {N_TRIALS} trials x {BITS_PER_TRIAL} bits | "
          f"{len(GRID)} configs x {len(TRAIN_SEEDS)} inits per channel")

    out = {
        "snr_db": SNRS, "n_trials": N_TRIALS, "bits_per_trial": BITS_PER_TRIAL,
        "tolerance_rel": TOL_REL, "alpha": ALPHA, "train_seeds": TRAIN_SEEDS,
        "train_snrs": TRAIN_SNRS, "train_samples": TRAIN_SAMPLES,
        "epochs": EPOCHS, "isi_taps": [float(t) for t in H_ISI],
        "grid": [{"window": w, "hidden": h, "params": n_params(w, h)} for w, h in GRID],
        "channels": {},
    }
    path = "results/mlp_min_size_all_channels.json"
    for name in only:
        out["channels"][name] = run_channel(name, CHANNELS[name])
        with open(path, "w") as fh:          # checkpoint after every channel
            json.dump(out, fh, indent=2)

    print(f"\n{'=' * 78}\n  SUMMARY: minimum size matching DF, by channel\n{'=' * 78}")
    print(f"  {'channel':<14} {'mod':<6} {'mem':>3}  {'tol only':>9}  {'both':>6}  config")
    for name, r in out["channels"].items():
        b = r["best_config_both"]
        cfg = f"w={b['window']} h={b['hidden']}" if b else "--"
        print(f"  {name:<14} {r['modulation']:<6} {r['memory']:>3}  "
              f"{str(r['min_params_tolerance']):>9}  {str(r['min_params_both_criteria']):>6}  {cfg}")
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
