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

THE COMPARATOR IS PER-CHANNEL. An earlier version of this study used
symbol-wise DF on every channel and carried a prose caveat saying DF is a weak
baseline on the channels with memory. That caveat was both under-scoped and
skippable, so it has been replaced by two mechanisms.

First, each channel now names its own classical comparator, the one the thesis
uses on it: symbol-wise DF where the channel is memoryless and symbol-wise
detection is the right classical answer, and Viterbi/MLSE where the channel
has memory (relaynet/relays/viterbi.py, as in Chapter 7). "Matches the
classical baseline" therefore means something different per channel -- which is
honest, because the classical baseline *is* different per channel. DF is still
measured everywhere and reported, so the numbers stay comparable with the
earlier DF-only run.

Second, the validity of each baseline is computed rather than asserted, and
tagged on its own row. Three checks:

    monotone   BER must fall as SNR rises. A baseline that gets *worse* with
               more SNR has failed, and "beating" it means nothing. The
               DF-only run had exactly this on isi_rayleigh, where DF ran
               0.4231 0.3804 0.3496 0.3423 0.3566 0.3832 -- rising above
               12 dB -- and still produced a headline 4-parameter minimum.
    beats_af   the classical relay should beat plain amplify-and-forward.
    floor      the BER at the top SNR. When this reaches ~0 the *relative*
               penalty metric stops being usable, because a negligible
               absolute gap divides into a large percentage (awgn: no size
               "matched" DF at +11% to +28% relative, on an absolute gap of
               0.0004 BER).

WHICH WAY A WEAK BASELINE ERRS. A baseline the relay can beat easily flatters
the relay, so any minimum measured against a weak comparator is optimistic --
the true floor against a strong classical scheme is higher, never lower. This
is why the memory channels are now scored against MLSE: the DF-only run put
their minimum at 6 parameters, and that number was an upper bound on the
relay's difficulty, not a measurement of it.

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
    FlatGainChannel,
    BranchAsymmetryChannel,
    CompositeChannel,
)
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, MLPRelay
from relaynet.relays.viterbi import ViterbiMLSERelay, ViterbiMLSEQPSKRelay
from relaynet.simulation.runner import (run_monte_carlo, _process_relay,
                                        Source, Destination)
from relaynet.modulation import calculate_ber

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

# The composite cascade uses its own taps (e6_composite_ported.py:38).
H_COMPOSITE = np.array([1.0, 0.6, 0.4])
H_COMPOSITE = H_COMPOSITE / np.linalg.norm(H_COMPOSITE)

# (window, hidden). Window 7 is included beyond the canonical maximum of 5
# because the ISI channels have 3 taps over two hops and may need the reach.
GRID = [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16),
    (3, 1), (3, 2), (3, 4), (3, 8), (3, 16),
    (5, 1), (5, 2), (5, 4), (5, 8), (5, 24),   # (5,24) = MLP-169, the control
    (7, 4), (7, 8),
    # Extended after the MLP-169 comparison put w=7 h1-8p (73p) at +5.2% on
    # isi_rayleigh and +2.5% on composite -- just outside the 2% bar, and the
    # widest w=7 point in the grid. The trend in width at w=7 is cleanly
    # monotone there (h1-4p +38.4% -> h1-8p +5.2%), so that is a grid ceiling
    # rather than a measured limit. These four close it out; every one is run
    # and reported whether it passes or not.
    (7, 16), (7, 24), (7, 32), (7, 48),
]

# Channel families. "memory" is the number of taps in the channel's impulse
# response: 1 means memoryless, so the crossover prediction applies to it.
# HOP 2, per channel family. e6_sim_ported.py:3-6 defines the Chapter 7 model
# as "Hop 1 = unknown channel ... Hop 2 = AWGN or coherently-compensated
# Rayleigh at the same SNR" (setups S1 isi->awgn, S3 nlbias->awgn, S4 control
# rayleigh->rayleigh). e6_flat_ported.py uses Rayleigh-magnitude fading plus
# AWGN for hop 2 on the flat unknown channels, and e6_composite_ported.py:202
# uses AdaptiveRayleighChannel. Only the Chapter 5 controls are symmetric.

def _hop2_awgn():
    return awgn_channel


def _hop2_rayleigh():
    return rayleigh_fading_channel


def _hop2_rayleigh_mag(seed=2):
    """Coherently-compensated Rayleigh: magnitude fading plus AWGN, the hop 2
    used by the E6 flat and composite experiments."""
    rng = np.random.default_rng(seed)

    def ch(signal, snr_db):
        n = len(signal)
        h = np.abs((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2))
        sigma = 1.0 / np.sqrt(2.0 * 10 ** (snr_db / 10.0))
        if np.iscomplexobj(signal):
            noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        else:
            noise = sigma * rng.standard_normal(n)
        return h * signal + noise

    return ch


CHANNELS = {
    "awgn": dict(
        make=lambda s: awgn_channel, mod="bpsk", memory=1,
        hop2=lambda: _hop2_awgn(),
        baseline=lambda: ("DF", DecodeAndForwardRelay()),
        note="Ch5 calibration reference (closed-form BER)"),
    "rayleigh": dict(
        make=lambda s: rayleigh_fading_channel, mod="qpsk", memory=1,
        hop2=lambda: _hop2_rayleigh(),
        baseline=lambda: ("DF", DecodeAndForwardRelay()),
        note="Ch5 canonical operating point"),
    "flat_gain": dict(
        make=lambda s: FlatGainChannel(gain_min=0.3, gain_max=2.0, seed=s),
        mod="bpsk", memory=1,
        hop2=lambda: _hop2_rayleigh_mag(),
        baseline=lambda: ("DF", DecodeAndForwardRelay()),
        note="Ch7 E6_FLAT unknown gain, g ~ U[0.3,2.0]"),
    "branch_asym": dict(
        make=lambda s: BranchAsymmetryChannel(seed=s), mod="bpsk", memory=1,
        hop2=lambda: _hop2_rayleigh_mag(),
        baseline=lambda: ("DF", DecodeAndForwardRelay()),
        note="Ch7 E6_FLAT branch asymmetry, a+- ~ U[0.6,1.4]"),
    "nlbias": dict(
        make=lambda s: NonlinearBiasChannel(saturation=1.5, dc_bias=0.5, seed=s),
        mod="bpsk", memory=1,
        hop2=lambda: _hop2_awgn(),
        baseline=lambda: ("DF", DecodeAndForwardRelay()),
        note="Ch7 nonlinear saturation + DC bias (memoryless but nonlinear)"),
    "isi": dict(
        make=lambda s: ISIChannel(H_ISI, seed=s), mod="bpsk", memory=3,
        hop2=lambda: _hop2_awgn(),
        baseline=lambda: ("MLSE", ViterbiMLSERelay(channel_taps=H_ISI)),
        note="Ch7 real 3-tap ISI"),
    "isi_complex": dict(
        make=lambda s: ComplexISIChannel(H_ISI, seed=s), mod="qpsk", memory=3,
        hop2=lambda: _hop2_awgn(),
        baseline=lambda: ("MLSE", _complex_native(
            ViterbiMLSEQPSKRelay(channel_taps=H_ISI))),
        note="Ch7 complex 3-tap ISI"),
    "isi_rayleigh": dict(
        make=lambda s: ComplexISIRayleighChannel(H_ISI, seed=s), mod="qpsk",
        memory=3,
        hop2=lambda: _hop2_awgn(),
        # Taps only, and named so. This channel fades per symbol, and the
        # runner has no way to hand the gains to a relay, so a genie-CSI
        # comparator is not available here -- see the note in evaluate_two_hop.
        baseline=lambda: ("MLSE (taps only)", _complex_native(
            ViterbiMLSEQPSKRelay(channel_taps=H_ISI))), note="Ch7 3-tap ISI on top of Rayleigh fading"),
    "composite": dict(
        make=lambda s: CompositeChannel(isi_taps=H_COMPOSITE, pa_sat=1.2,
                                        include_phase=True, seed=s),
        mod="bpsk", memory=3,
        hop2=lambda: _hop2_rayleigh_mag(),
        baseline=lambda: ("MLSE", ViterbiMLSERelay(channel_taps=H_COMPOSITE)),
        note="Ch7 composite cascade: ISI -> PA -> phase -> AWGN"),
}

# TWO CHANNELS DELIBERATELY EXCLUDED, both because the standalone pairing is
# not a valid measurement rather than because the result was unwelcome:
#
# flat_phase  FlatPhaseChannel applies a constant unknown rotation per block
#             (its docstring: "the DBPSK scenario"). A coherent constellation
#             through an unknown rotation has no phase reference, so DF and
#             the relay both guess: the run gave ~0.5 BER at every size,
#             non-monotonic in SNR, Wilcoxon p 0.60-0.96. The thesis pairs
#             this channel with a DBPSK source and differential detection,
#             which needs a different comparator and so lives in
#             mlp_min_size_flat_phase_dbpsk.py.
#
# pa          PowerAmplifierChannel is explicitly noiseless -- "This is a
#             non-noisy channel (noise added elsewhere in composite)", with
#             snr_db "Unused (for API compatibility)". Standalone it gives
#             BER 0.00000 for DF, AF and every MLP at every SNR, which
#             measures nothing. It is a component of CompositeChannel, and
#             the composite is included above as the thesis's actual PA
#             scenario (e6_composite_ported.py).


def _complex_native(relay):
    """Mark a relay as taking the complex signal whole, not axis by axis."""
    relay.handles_complex_natively = True
    return relay


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


def two_hop_ber(relay, hop1, hop2, mod, num_bits, snr_db, seed):
    """One two-hop trial with *independent* hop-1 and hop-2 channels.

    relaynet's simulate_transmission reuses one channel_fn for both hops
    (runner.py: rx_dest = channel_fn(relay_out, snr_db)). That is right for
    the Chapter 5 controls, where both hops really are the same channel, but
    wrong for every Chapter 7 channel: e6_sim_ported.py states the model as
    "Hop 1 = unknown channel ... Hop 2 = AWGN or coherently-compensated
    Rayleigh at the same SNR", with setups S1 isi -> awgn and S3 nlbias ->
    awgn. Putting the unknown channel on both hops leaves the destination
    facing un-equalized distortion it has no way to undo, which is why BER
    then *rises* with SNR: the noise shrinks but the distortion does not.
    """
    source = Source(seed=seed, modulation=mod)
    destination = Destination(modulation=mod)
    tx_bits, tx_symbols = source.transmit(num_bits)

    rx_relay = hop1(tx_symbols, snr_db)
    if getattr(relay, "handles_complex_natively", False):
        # runner._apply_relay splits a complex signal into I and Q and calls
        # process() on each axis. That is right for the per-axis MLPs and
        # wrong for a QPSK MLSE, whose trellis is defined over the complex
        # constellation: handing it y.real alone destroys it. This is what
        # made the isi_complex and isi_rayleigh MLSE baselines look "weak".
        # Standalone on isi_complex that relay recovers the symbols exactly
        # (0.00000 symbol error at 20 dB), so on that channel the fault was in
        # the dispatch, not the comparator.
        #
        # isi_rayleigh is different and the sentence above used to over-reach
        # to it. That channel is g[n] * conv(x, h) + v with an independent
        # fading magnitude per symbol, and a trellis given only h is genuinely
        # model-mismatched there: 0.18 symbol error at 20 dB, against 0.02 for
        # the same trellis handed the gains (FadingAwareViterbiQPSKRelay). Its
        # comparator below is therefore labelled MLSE (taps only) for what it
        # is. tbl:table-minsize is unaffected -- report_minsize_vs_169.analyse
        # scores every row against the MLP-169 sweep entry, not against this
        # baseline -- but min_params_both_criteria in the JSON is scored
        # against it and is optimistic for that one channel.
        relay_out = relay.process(rx_relay)
    else:
        relay_out = _process_relay(relay, rx_relay, mod)

    rx_dest = hop2(relay_out, snr_db)
    if isinstance(rx_dest, tuple):
        rx_dest = rx_dest[0]
    return calculate_ber(tx_bits, destination.receive(rx_dest))[0]


def evaluate_two_hop(relay, hop1, hop2, mod, tag):
    """Monte Carlo over the asymmetric two-hop chain, from a fixed global RNG
    state so every relay on a channel sees identical realizations."""
    np.random.seed(SEED % (2 ** 31))
    ber, trials = [], []
    for snr in SNRS:
        t = [two_hop_ber(relay, hop1, hop2, mod, BITS_PER_TRIAL, snr, SEED + i)
             for i in range(N_TRIALS)]
        trials.append(t)
        ber.append(float(np.mean(t)))
    if tag:
        print(f"    {tag:<30} " + "  ".join(f"{b:.4f}" for b in ber), flush=True)
    return np.asarray(ber), np.asarray(trials)


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


def baseline_diagnostics(base_ber, af_ber):
    """Is this baseline a valid yardstick? Computed, not asserted.

    A prose caveat about baseline quality is skippable and, in the DF-only
    version of this study, was also under-scoped -- it warned about the ISI
    channels and missed that DF had outright failed on isi_rayleigh. These
    checks travel with the row instead.
    """
    b = np.asarray(base_ber, dtype=float)
    monotone = bool(np.all(np.diff(b) <= 1e-12))
    beats_af = bool(np.all(b <= np.asarray(af_ber, dtype=float) + 1e-12))
    floor = float(b[-1])
    if not monotone:
        verdict = "BROKEN"          # worse with more SNR: beating it means nothing
    elif floor <= 1e-5:
        verdict = "near-optimal"    # relative penalties unusable at the top end
    elif not beats_af:
        verdict = "weak"            # loses to plain AF somewhere
    else:
        verdict = "ok"
    return {"monotone": monotone, "beats_af": beats_af, "floor": floor,
            "verdict": verdict}


def run_channel(name, spec):
    print(f"\n{'=' * 78}\n  {name}   [{spec['note']}]   "
          f"modulation {spec['mod']}, channel memory {spec['memory']} tap(s)")
    print(f"{'=' * 78}")
    print("  " + " " * 30 + "  ".join(f"{s:>6}dB" for s in SNRS))

    channel = spec["make"](CHANNEL_SEED)
    hop2 = spec["hop2"]()
    mod = spec["mod"]

    def ev(relay, tag):
        return evaluate_two_hop(relay, channel, hop2, mod, tag)

    print("\n  classical baselines")
    df_ber, df_trials = ev(DecodeAndForwardRelay(), "DF (0 params)")
    af_ber, af_trials = ev(AmplifyAndForwardRelay(), "AF (0 params)")

    # the comparator the thesis actually uses on this channel
    base_name, base_relay = spec["baseline"]()
    if base_name == "DF":
        base_ber, base_trials = df_ber, df_trials
    else:
        base_ber, base_trials = ev(base_relay, f"{base_name} (0 params)")
    diag = baseline_diagnostics(base_ber, af_ber)
    print(f"    baseline for scoring: {base_name}   "
          f"monotone {diag['monotone']}   beats AF {diag['beats_af']}   "
          f"floor {diag['floor']:.5f}   -> {diag['verdict']}")
    if diag["verdict"] == "BROKEN":
        print("    WARNING: this baseline gets worse as SNR rises. Any "
              "'match' on this channel clears a failed bar and is not a floor.")

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
            ber, trials = ev(relay, None)
            per_snr = compare(ber, trials, base_ber, base_trials)
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
        "baseline": base_name, "baseline_ber": [float(b) for b in base_ber],
        "hop2": spec["hop2"].__code__.co_names[0],
        "baseline_diagnostics": diag,
        "df_ber": [float(b) for b in df_ber], "af_ber": [float(b) for b in af_ber],
        "sweep": rows,
        "min_params_tolerance": min((r["params"] for r in tol_match), default=None),
        "min_params_both_criteria": min((r["params"] for r in both), default=None),
        "best_config_both": (min(both, key=lambda r: r["params"])
                             if both else None),
    }
    if both:
        b = min(both, key=lambda r: r["params"])
        print(f"\n  -> smallest passing both vs {base_name}: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    elif tol_match:
        b = min(tol_match, key=lambda r: r["params"])
        print(f"\n  -> none passes both vs {base_name}; smallest within tolerance: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    else:
        print(f"\n  -> no configuration in the grid matched {base_name}")
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
    if os.path.exists(path):
        # merge, so one channel can be re-run without discarding the rest
        prev = json.load(open(path))
        out["channels"] = {k: v for k, v in prev.get("channels", {}).items()
                           if k in CHANNELS}
    for name in only:
        out["channels"][name] = run_channel(name, CHANNELS[name])
        with open(path, "w") as fh:          # checkpoint after every channel
            json.dump(out, fh, indent=2)

    print(f"\n{'=' * 78}\n  SUMMARY: minimum size matching DF, by channel\n{'=' * 78}")
    print(f"  {'channel':<14} {'mod':<6} {'mem':>3} {'base':>5} {'valid':>12}  "
          f"{'tol':>5} {'both':>5}  config")
    for name, r in out["channels"].items():
        b = r["best_config_both"]
        cfg = f"w={b['window']} h={b['hidden']}" if b else "--"
        d = r["baseline_diagnostics"]
        print(f"  {name:<14} {r['modulation']:<6} {r['memory']:>3} {r['baseline']:>5} "
              f"{d['verdict']:>12}  {str(r['min_params_tolerance']):>5} "
              f"{str(r['min_params_both_criteria']):>5}  {cfg}")
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
