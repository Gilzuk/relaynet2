"""Minimum MLP size on the unknown-phase channel -- DBPSK companion study.

Why this is a separate script. mlp_min_size_all_channels.py runs every channel
through the shared run_monte_carlo pipeline with a QPSK/BPSK source and
symbol-wise DF as the comparator. That configuration is meaningless on
FlatPhaseChannel, and the first run proved it: every configuration sat at
~0.5 BER, non-monotonic in SNR, with Wilcoxon p between 0.60 and 0.96. Pure
noise, no measurement.

The reason is structural, not a bug in the sweep. FlatPhaseChannel applies a
constant unknown rotation theta ~ U[0,2pi) per block (its own docstring calls
it "the DBPSK scenario"). A QPSK constellation through an unknown rotation is
undecodable in principle -- there is no phase reference -- so both DF and the
relay are guessing. The thesis pairs this channel with a DBPSK source, whose
information lives in the phase *difference* between consecutive symbols and is
therefore invariant to a constant rotation, and its classical baseline is
differential detection, sign(Re{y[i] conj(y[i-1])}), not constellation
slicing (e6_flat_ported.py:12).

So this channel needs both a different source and a different comparator. It
cannot share the cross-channel table without saying so, which is why it lives
here and is reported as a companion result rather than as another row.

Everything else is held to the main study: same size grid logic, same three
initializations per configuration, same two verdicts (2% relative tolerance
and a paired Wilcoxon test), same Monte Carlo budget.

PARAMETER COUNT DIFFERS. On this channel the relay sees I and Q jointly
(e6_flat_ported.extract_windows concatenates them), not one axis at a time:

    params = hidden * (2*window + 2) + 1

against hidden*(window+2)+1 for the per-axis relays elsewhere. A window-5
relay here is therefore roughly twice the size of a window-5 relay in the
main table; compare parameter counts, not window widths, across the two.

The two-hop pipeline is the one in e6_flat_ported.py: hop 1 is the unknown
phase channel, hop 2 is Rayleigh-magnitude fading plus AWGN, and the first
bit is dropped at the destination because differential encoding has no
predecessor for it.

SNR convention follows memory-bank/techContext.md: gamma = 10^(SNR_dB/10).
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relaynet.channels import FlatPhaseChannel
from relaynet.relays import MLPRelay

SNRS = [0, 4, 8, 12, 16, 20]
N_TRIALS = 20
N_BITS = 20000
TRAIN_SNRS = [5, 10, 15]
TRAIN_SAMPLES = 60000
EPOCHS = 25
BATCH = 256
LR = 3e-3
SEED = 0
TRAIN_SEEDS = [0, 1, 2]
CHANNEL_SEED = 1

TOL_REL = 0.02
ALPHA = 0.05

# (window, hidden). Window 11 is the value e6_flat_ported.py uses.
GRID = [
    (1, 1), (1, 2), (1, 4), (1, 8),
    (3, 1), (3, 2), (3, 4), (3, 8),
    (5, 1), (5, 2), (5, 4), (5, 8),
    (7, 4), (7, 8),
    (11, 4), (11, 8),
]


def n_params(window, hidden):
    """Joint I/Q relay: 2*window inputs."""
    return hidden * (2 * window + 2) + 1


def diff_encode(x):
    """DBPSK differential encoding: running product (e6_flat_ported.py)."""
    s = np.empty_like(x, dtype=float)
    s[0] = 1.0
    for i in range(1, len(x)):
        s[i] = s[i - 1] * x[i]
    return s


def diff_detect(y):
    """Differential detection: sign(Re{y[i] conj(y[i-1])})."""
    out = np.ones(len(y))
    out[1:] = np.sign(np.real(y[1:] * np.conj(y[:-1])))
    out[out == 0] = 1.0
    return out


def extract_windows(y, window):
    """I and Q windows concatenated (e6_flat_ported.extract_windows)."""
    pad = window // 2
    wr = np.lib.stride_tricks.sliding_window_view(
        np.pad(y.real, (pad, pad), mode="constant"), window)
    wi = np.lib.stride_tricks.sliding_window_view(
        np.pad(y.imag, (pad, pad), mode="constant"), window)
    return np.concatenate([wr, wi], axis=1)


def hop2(relay_out, snr_db, rng):
    """Rayleigh-magnitude hop 2 plus AWGN, as in e6_flat_ported.py."""
    n = len(relay_out)
    h = np.abs((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2))
    sigma = 10 ** (-snr_db / 20.0)
    return h * relay_out + sigma * rng.standard_normal(n)


def run_trial(kind, snr_db, trial, channel, relay=None, window=None):
    """One end-to-end trial. Returns BER over bits 1.. (bit 0 is the
    differential reference and carries no information)."""
    rng = np.random.default_rng(10_000 + trial)
    bits = rng.integers(0, 2, N_BITS)
    x = 1.0 - 2.0 * bits
    s = diff_encode(x)
    y = channel(s.astype(complex), snr_db)

    if kind == "df":
        # classical baseline: differential detection at the relay
        relay_out = diff_detect(y)
    elif kind == "af":
        gain = np.sqrt(1.0 / (np.mean(np.abs(y) ** 2) + 1e-12))
        # AF forwards the complex sample; detection stays differential at
        # the destination, so hop 2 is applied to the complex stream
        rng2 = np.random.default_rng(50_000 + trial)
        h = np.abs((rng2.standard_normal(N_BITS)
                    + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2))
        sigma = 10 ** (-snr_db / 20.0)
        nz = sigma * (rng2.standard_normal(N_BITS)
                      + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2)
        y_dest = h * (gain * y) + nz
        out = (diff_detect(y_dest) < 0).astype(int)
        return float(np.mean(out[1:] != bits[1:]))
    else:
        w = extract_windows(y, window)
        raw = relay.fwd(w).ravel()
        p = np.sqrt(np.mean(raw ** 2) + 1e-12)
        relay_out = raw / p

    rng2 = np.random.default_rng(50_000 + trial)
    y_dest = hop2(relay_out, snr_db, rng2)
    out = (y_dest < 0).astype(int)
    return float(np.mean(out[1:] != bits[1:]))


def evaluate(kind, channel, relay=None, window=None):
    ber, trials = [], []
    for snr in SNRS:
        t = [run_trial(kind, snr, i, channel, relay, window) for i in range(N_TRIALS)]
        trials.append(t)
        ber.append(float(np.mean(t)))
    return np.asarray(ber), np.asarray(trials)


def make_training_data(channel, window, rng):
    """Target the *information* symbol x[i], not the transmitted symbol s[i].

    s[i] is unlearnable here and training against it produces a relay at
    chance. The block rotation theta is redrawn every call, so the same
    received sample maps to s[i] = +1 or -1 depending on a draw the network
    cannot observe -- which is exactly the property differential encoding
    exists to exploit. The phase-invariant quantity is x[i] = s[i] s[i-1],
    the information symbol, and it is what DF forwards on this channel
    (diff_detect estimates sign(Re{y[i] conj(y[i-1])}) ~ x[i]).

    Targeting x[i] makes the relay a learned differential detector, directly
    comparable to the classical one. Note the consequence for the sweep: x[i]
    is a function of two consecutive samples, so a window-1 relay cannot
    represent it at all and must fail here -- unlike on the memoryless
    channels, where window 1 was sufficient and wider windows only hurt."""
    per_snr = TRAIN_SAMPLES // len(TRAIN_SNRS)
    X_list, T_list = [], []
    for snr in TRAIN_SNRS:
        b = rng.integers(0, 2, per_snr)
        x = 1.0 - 2.0 * b
        s = diff_encode(x)
        y = channel(s.astype(complex), snr)
        X_list.append(extract_windows(y, window))
        T_list.append(x)
    return np.vstack(X_list), np.concatenate(T_list)


def main():
    print("Unknown-phase channel, DBPSK source, differential-detection baseline")
    print(f"SNRs {SNRS} | {N_TRIALS} trials x {N_BITS} bits | "
          f"{len(GRID)} configs x {len(TRAIN_SEEDS)} inits")
    print("  " + " " * 26 + "  ".join(f"{s:>6}dB" for s in SNRS))

    channel = FlatPhaseChannel(seed=CHANNEL_SEED)

    print("\n  classical baselines")
    df_ber, df_trials = evaluate("df", channel)
    af_ber, _ = evaluate("af", channel)
    print("    DF diff-detect (0p)        " + "  ".join(f"{b:.4f}" for b in df_ber))
    print("    AF diff-detect (0p)        " + "  ".join(f"{b:.4f}" for b in af_ber))

    print("\n  MLP sweep")
    rows = []
    for window, hidden in GRID:
        p = n_params(window, hidden)
        seed_runs = []
        for ts in TRAIN_SEEDS:
            rng = np.random.default_rng(1000 + ts)
            X, T = make_training_data(channel, window, rng)
            relay = MLPRelay(input_size=2 * window, hidden_size=hidden,
                             output_size=1, window_size=None, seed=ts)
            relay.train_on_data(X, T, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            ber, trials = evaluate("mlp", channel, relay, window)
            per_snr = []
            for i, snr in enumerate(SNRS):
                d = trials[i] - df_trials[i]
                pv = 1.0 if np.allclose(d, 0) else float(wilcoxon(trials[i], df_trials[i])[1])
                rel = float((ber[i] - df_ber[i]) / df_ber[i]) if df_ber[i] > 0 else float("nan")
                per_snr.append({"snr_db": snr, "mlp_ber": float(ber[i]),
                                "df_ber": float(df_ber[i]), "rel_penalty": rel,
                                "wilcoxon_p": pv,
                                "wilcoxon_loses": bool(pv < ALPHA and rel == rel and rel > 0)})
            finite = [r["rel_penalty"] for r in per_snr if r["rel_penalty"] == r["rel_penalty"]]
            worst = max(finite) if finite else float("nan")
            seed_runs.append({"train_seed": ts, "ber": [float(b) for b in ber],
                              "worst_rel_penalty": float(worst),
                              "tolerance_ok": bool(worst <= TOL_REL),
                              "wilcoxon_ok": not any(r["wilcoxon_loses"] for r in per_snr),
                              "per_snr": per_snr})
        worst = max(s["worst_rel_penalty"] for s in seed_runs)
        best = min(s["worst_rel_penalty"] for s in seed_runs)
        tol_all = all(s["tolerance_ok"] for s in seed_runs)
        wil_all = all(s["wilcoxon_ok"] for s in seed_runs)
        rows.append({"window": window, "hidden": hidden, "params": p,
                     "worst_rel_penalty_over_seeds": float(worst),
                     "best_rel_penalty_over_seeds": float(best),
                     "matches_tolerance_all_seeds": bool(tol_all),
                     "matches_wilcoxon_all_seeds": bool(wil_all),
                     "seed_runs": seed_runs})
        print(f"    w={window} h={hidden} ({p}p)".ljust(26)
              + f"penalty {100*best:+7.1f}% .. {100*worst:+7.1f}%"
              + f"   tol {'ok' if tol_all else 'NO'}"
              + f"   wilcoxon {'ok' if wil_all else 'NO'}", flush=True)

    tol = [r for r in rows if r["matches_tolerance_all_seeds"]]
    both = [r for r in rows if r["matches_tolerance_all_seeds"] and r["matches_wilcoxon_all_seeds"]]
    out = {"snr_db": SNRS, "n_trials": N_TRIALS, "bits_per_trial": N_BITS,
           "tolerance_rel": TOL_REL, "alpha": ALPHA, "train_seeds": TRAIN_SEEDS,
           "source": "dbpsk", "baseline": "differential detection",
           "param_formula": "hidden*(2*window+2)+1  (joint I/Q input)",
           "df_ber": [float(b) for b in df_ber], "af_ber": [float(b) for b in af_ber],
           "sweep": rows,
           "min_params_tolerance": min((r["params"] for r in tol), default=None),
           "min_params_both_criteria": min((r["params"] for r in both), default=None)}
    path = "results/mlp_min_size_flat_phase_dbpsk.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 70)
    if both:
        b = min(both, key=lambda r: r["params"])
        print(f"  smallest passing both: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    elif tol:
        b = min(tol, key=lambda r: r["params"])
        print(f"  none passes both; smallest within tolerance: {b['params']} params "
              f"(window {b['window']}, hidden {b['hidden']})")
    else:
        print("  no configuration matched differential detection")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
