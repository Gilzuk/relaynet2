"""Exact minimum MLP size per channel, by bisection on hidden width.

The grid sweep in mlp_min_size_all_channels.py samples hidden sizes on a
coarse ladder (1, 2, 4, 8, 16, 24), so its answer is only ever "somewhere
between two rungs". This script bisects instead, which pins the boundary
exactly and reaches much larger widths for the same number of trainings:
finding the smallest passing width in 1..64 costs ~6 probes rather than 64.

WHAT MAY AND MAY NOT BE BISECTED. Bisection needs the predicate to be
monotone -- if width h matches, every width above h must match. That is a
real assumption and the grid run shows it is FALSE across the size axis as a
whole. On Rayleigh at window 5 the sequence was:

    h=1  +0.11%  pass      h=4  +2.28%  FAIL
    h=2  +1.19%  pass      h=8  +2.79%  FAIL
                           h=24 +3.18%  FAIL

Pass, then fail, as capacity grows -- because on a memoryless channel the
window taps carry no information and extra capacity fits noise. Bisecting
over "number of parameters" across window widths would therefore return
whichever side of that fold the midpoint landed on, and the answer would be
an artifact of the probe sequence.

So the two axes are treated differently, and only one is bisected:

  window   swept exhaustively (four values). The window is a structural
           choice, not a capacity knob: it must be at least the channel's
           memory for the relay to represent the channel at all, and beyond
           that it costs accuracy. Non-monotone by design.
  hidden   bisected at each fixed window, where "more units cannot represent
           less" is at least plausible.

Even that is not assumed. After bisection finds a boundary h*, the result is
audited by testing widths above it (h*+1, 2h*, and the cap). If any of them
fails, the row is reported as NON-MONOTONE and h* is labelled a local
boundary rather than a minimum. A silent monotonicity assumption is exactly
the kind of thing this study has already been bitten by.

Everything else is inherited from mlp_min_size_all_channels: the same
channels, per-channel classical comparators, computed baseline diagnostics,
asymmetric two-hop model, seeding discipline, and both verdict criteria.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlp_min_size_all_channels as base
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, MLPRelay

WINDOWS = [1, 3, 5, 7]
H_MAX = 64
TRAIN_SEEDS = base.TRAIN_SEEDS


def matches(channel, hop2, mod, window, hidden, base_ber, base_trials, cache):
    """Train TRAIN_SEEDS relays at this size; matched only if every seed
    passes both criteria. Cached -- bisection revisits sizes."""
    key = (window, hidden)
    if key in cache:
        return cache[key]

    runs = []
    for ts in TRAIN_SEEDS:
        rng = np.random.default_rng(1000 + ts)
        X, T = base.make_training_data(channel, mod, window, rng)
        relay = MLPRelay(input_size=window, hidden_size=hidden,
                         output_size=1, window_size=window, seed=ts)
        relay.train_on_data(X, T, epochs=base.EPOCHS,
                            batch_size=base.BATCH, lr=base.LR)
        ber, trials = base.evaluate_two_hop(relay, channel, hop2, mod, None)
        per_snr = base.compare(ber, trials, base_ber, base_trials)
        finite = [r["rel_penalty"] for r in per_snr
                  if r["rel_penalty"] == r["rel_penalty"]]
        worst = max(finite) if finite else float("nan")
        runs.append({
            "train_seed": ts,
            "ber": [float(b) for b in ber],
            "worst_rel_penalty": float(worst),
            "tolerance_ok": bool(worst <= base.TOL_REL),
            "wilcoxon_ok": not any(r["wilcoxon_loses"] for r in per_snr),
        })

    ok = all(r["tolerance_ok"] and r["wilcoxon_ok"] for r in runs)
    worst = max(r["worst_rel_penalty"] for r in runs)
    best = min(r["worst_rel_penalty"] for r in runs)
    out = {"window": window, "hidden": hidden,
           "params": hidden * (window + 2) + 1,
           "matches": bool(ok),
           "worst_rel_penalty_over_seeds": float(worst),
           "best_rel_penalty_over_seeds": float(best),
           "seed_runs": runs}
    cache[key] = out
    print(f"      probe w={window} h={hidden:<3} ({out['params']:>4}p)  "
          f"penalty {100*best:+8.1f}% .. {100*worst:+8.1f}%   "
          f"{'match' if ok else 'no'}", flush=True)
    return out


def bisect_window(channel, hop2, mod, window, base_ber, base_trials, cache):
    """Smallest hidden width at this window that matches, or None."""
    # upper bracket: if the cap does not match, nothing at this window does
    # (subject to the monotonicity audit below)
    if not matches(channel, hop2, mod, window, H_MAX,
                   base_ber, base_trials, cache)["matches"]:
        return None, "cap does not match"

    if matches(channel, hop2, mod, window, 1,
               base_ber, base_trials, cache)["matches"]:
        return 1, "ok"

    lo, hi = 1, H_MAX          # lo known-fail, hi known-pass
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if matches(channel, hop2, mod, window, mid,
                   base_ber, base_trials, cache)["matches"]:
            hi = mid
        else:
            lo = mid
    return hi, "ok"


def audit(channel, hop2, mod, window, h_star, base_ber, base_trials, cache):
    """Test widths above the boundary. Bisection is only valid if they pass."""
    probes = sorted({h_star + 1, 2 * h_star, H_MAX} - {h_star})
    bad = []
    for h in probes:
        if h > H_MAX:
            continue
        if not matches(channel, hop2, mod, window, h,
                       base_ber, base_trials, cache)["matches"]:
            bad.append(h)
    return bad


def run_channel(name, spec):
    print(f"\n{'=' * 78}\n  {name}   [{spec['note']}]   "
          f"modulation {spec['mod']}, channel memory {spec['memory']} tap(s)")
    print(f"{'=' * 78}")

    channel = spec["make"](base.CHANNEL_SEED)
    hop2 = spec["hop2"]()
    mod = spec["mod"]

    df_ber, df_trials = base.evaluate_two_hop(
        DecodeAndForwardRelay(), channel, hop2, mod, "  DF (0 params)")
    af_ber, _ = base.evaluate_two_hop(
        AmplifyAndForwardRelay(), channel, hop2, mod, "  AF (0 params)")
    base_name, base_relay = spec["baseline"]()
    if base_name == "DF":
        base_ber, base_trials = df_ber, df_trials
    else:
        base_ber, base_trials = base.evaluate_two_hop(
            base_relay, channel, hop2, mod, f"  {base_name} (0 params)")
    diag = base.baseline_diagnostics(base_ber, af_ber)
    print(f"    baseline {base_name}: {diag['verdict']}  "
          f"(monotone {diag['monotone']}, beats AF {diag['beats_af']}, "
          f"floor {diag['floor']:.5f})")

    cache, per_window = {}, {}
    for w in WINDOWS:
        print(f"\n    window {w}")
        h_star, why = bisect_window(channel, hop2, mod, w,
                                    base_ber, base_trials, cache)
        if h_star is None:
            print(f"      -> no width up to {H_MAX} matches ({why})")
            per_window[w] = {"hidden": None, "params": None,
                             "monotone": None, "failed_above": []}
            continue
        bad = audit(channel, hop2, mod, w, h_star, base_ber, base_trials, cache)
        p = h_star * (w + 2) + 1
        tag = "" if not bad else f"   NON-MONOTONE: fails again at h={bad}"
        print(f"      -> smallest matching width h={h_star} ({p} params){tag}")
        per_window[w] = {"hidden": h_star, "params": p,
                         "monotone": not bad, "failed_above": bad}

    valid = [(w, d) for w, d in per_window.items()
             if d["params"] is not None and d["monotone"]]
    best = min(valid, key=lambda t: t[1]["params"]) if valid else None
    if best:
        print(f"\n  => minimum over windows: {best[1]['params']} params "
              f"(window {best[0]}, hidden {best[1]['hidden']}) vs {base_name}")
    else:
        print(f"\n  => no window yields a monotone minimum vs {base_name}")

    return {
        "note": spec["note"], "modulation": mod, "memory": spec["memory"],
        "baseline": base_name, "baseline_diagnostics": diag,
        "baseline_ber": [float(b) for b in base_ber],
        "df_ber": [float(b) for b in df_ber],
        "af_ber": [float(b) for b in af_ber],
        "per_window": {str(w): d for w, d in per_window.items()},
        "min_params": best[1]["params"] if best else None,
        "min_window": best[0] if best else None,
        "min_hidden": best[1]["hidden"] if best else None,
        "probes": [v for v in cache.values()],
        "n_probes": len(cache),
    }


def main():
    only = sys.argv[1:] or list(base.CHANNELS)
    print("Minimum MLP size by bisection on hidden width")
    print(f"windows {WINDOWS} swept exhaustively; hidden bisected in 1..{H_MAX}")
    print(f"{len(TRAIN_SEEDS)} inits per probe; a size matches only if every "
          f"init passes both criteria")

    path = "results/mlp_min_size_bisect.json"
    out = {"windows": WINDOWS, "h_max": H_MAX, "train_seeds": TRAIN_SEEDS,
           "snr_db": base.SNRS, "n_trials": base.N_TRIALS,
           "bits_per_trial": base.BITS_PER_TRIAL,
           "tolerance_rel": base.TOL_REL, "alpha": base.ALPHA, "channels": {}}
    if os.path.exists(path):
        prev = json.load(open(path))
        out["channels"] = {k: v for k, v in prev.get("channels", {}).items()
                           if k in base.CHANNELS}
    for name in only:
        out["channels"][name] = run_channel(name, base.CHANNELS[name])
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    print(f"\n{'=' * 78}\n  SUMMARY\n{'=' * 78}")
    print(f"  {'channel':<14} {'base':>5} {'valid':>12} {'min p':>6} "
          f"{'w':>2} {'h':>3} {'probes':>7}")
    for n, r in out["channels"].items():
        print(f"  {n:<14} {r['baseline']:>5} "
              f"{r['baseline_diagnostics']['verdict']:>12} "
              f"{str(r['min_params']):>6} {str(r['min_window']):>2} "
              f"{str(r['min_hidden']):>3} {r['n_probes']:>7}")
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
