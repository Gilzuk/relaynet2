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
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay
from deep_mlp_relay import DeepMLPRelay, n_params, arch_label

WINDOWS = [1, 3, 5, 7]
DEPTHS = [1, 2, 3]      # hidden layers; depth 1 reproduces MLPRelay exactly
H_MAX = 64              # cap on neurons per hidden layer
TRAIN_SEEDS = base.TRAIN_SEEDS


def matches(channel, hop2, mod, window, depth, hidden,
            base_ber, base_trials, cache):
    """Train TRAIN_SEEDS relays at this size; matched only if every seed
    passes both criteria. Cached -- bisection revisits sizes."""
    key = (window, depth, hidden)
    if key in cache:
        return cache[key]

    runs = []
    for ts in TRAIN_SEEDS:
        rng = np.random.default_rng(1000 + ts)
        X, T = base.make_training_data(channel, mod, window, rng)
        relay = DeepMLPRelay(input_size=window, width=hidden, depth=depth,
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
    out = {"window": window, "depth": depth, "width": hidden,
           "arch": arch_label(hidden, depth),
           "params": n_params(window, hidden, depth),
           "matches": bool(ok),
           "worst_rel_penalty_over_seeds": float(worst),
           "best_rel_penalty_over_seeds": float(best),
           "seed_runs": runs}
    cache[key] = out
    print(f"      probe w={window} {out['arch']:<22} ({out['params']:>5}p)  "
          f"penalty {100*best:+8.1f}% .. {100*worst:+8.1f}%   "
          f"{'match' if ok else 'no'}", flush=True)
    return out


def bisect_width(channel, hop2, mod, window, depth, base_ber, base_trials, cache):
    """Smallest hidden width at this window that matches, or None.

    Galloping search, not a plain bisection over [1, H_MAX]. The first
    version of this took h=H_MAX as the known-pass upper bracket, on the
    reasoning that if the largest width fails then nothing matches. That is
    the monotonicity assumption this study has already disproved, and it
    inverted the answer immediately: on Rayleigh, h=64 fails at +2.2% while
    h=1 and h=2 pass, because extra capacity fits noise on a memoryless
    channel. The bracket reported "no width matches" for a channel whose
    minimum is 4 parameters.

    So the bracket is *found* rather than assumed: probe h = 1, 2, 4, 8, ...
    until one matches, then bisect between the last failure and that match.
    When h=1 matches -- the common case here -- it costs a single probe.
    Monotonicity is still assumed strictly inside the final bracket, and the
    audit afterwards is what tests it.
    """
    def ok(h):
        return matches(channel, hop2, mod, window, depth, h,
                       base_ber, base_trials, cache)["matches"]

    if ok(1):
        return 1, "h=1 matches"

    lo, hi = 1, None           # lo known-fail
    h = 2
    while h <= H_MAX:
        if ok(h):
            hi = h
            break
        lo = h
        h *= 2
    if hi is None:
        return None, f"no match at any probed width up to {H_MAX}"

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, "ok"


def audit(channel, hop2, mod, window, depth, h_star,
          base_ber, base_trials, cache):
    """Test widths above the boundary. Bisection is only valid if they pass."""
    probes = sorted({h_star + 1, 2 * h_star, H_MAX} - {h_star})
    bad = []
    for h in probes:
        if h > H_MAX:
            continue
        if not matches(channel, hop2, mod, window, depth, h,
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

    cache, cells = {}, {}
    for w in WINDOWS:
        for L in DEPTHS:
            print(f"\n    window {w}, depth {L}")
            h_star, why = bisect_width(channel, hop2, mod, w, L,
                                       base_ber, base_trials, cache)
            if h_star is None:
                print(f"      -> no width up to {H_MAX} matches ({why})")
                cells[(w, L)] = {"width": None, "params": None,
                                 "monotone": None, "failed_above": []}
                continue
            bad = audit(channel, hop2, mod, w, L, h_star,
                        base_ber, base_trials, cache)
            p = n_params(w, h_star, L)
            tag = "" if not bad else f"   NON-MONOTONE: fails again at h={bad}"
            print(f"      -> smallest match {arch_label(h_star, L)} "
                  f"({p} params){tag}")
            cells[(w, L)] = {"width": h_star, "params": p,
                             "arch": arch_label(h_star, L),
                             "monotone": not bad, "failed_above": bad}

    # Two different questions, both answered, because reporting only one of
    # them misleads. A cell flagged non-monotone still contains a genuine
    # match at h*: what the flag says is that wider networks at the same
    # (window, depth) stop matching, so h* is a point that works rather than
    # a threshold above which everything works.
    matched = [(k, v) for k, v in cells.items() if v["params"] is not None]
    mono = [(k, v) for k, v in matched if v["monotone"]]
    best_any = min(matched, key=lambda t: t[1]["params"]) if matched else None
    best = min(mono, key=lambda t: t[1]["params"]) if mono else None

    print(f"\n    smallest matching hidden stack by (window, depth)")
    print("      " + "depth".rjust(8)
          + "".join(f"{L:>26}" for L in DEPTHS))
    for w in WINDOWS:
        row = []
        for L in DEPTHS:
            c = cells[(w, L)]
            if c["params"] is None:
                row.append(f"{'--':>16}")
            else:
                mark = "" if c["monotone"] else "!"
                cell = "%s (%dp)%s" % (c["arch"], c["params"], mark)
                row.append(f"{cell:>26}")
        print(f"      w={w:<6}" + "".join(row))

    if best_any:
        (aw, aL), av = best_any
        print(f"\n  => smallest that matches at all: w={aw} "
              f"{arch_label(av['width'], aL)} = {av['params']} params "
              f"vs {base_name}"
              + ("" if av["monotone"] else
                 "   -- non-monotone: a working point, not a threshold"))
    else:
        print(f"\n  => nothing matches {base_name} anywhere in the search space")
    if best:
        (bw, bL), bv = best
        print(f"  => smallest with every wider net still matching: w={bw} "
              f"{arch_label(bv['width'], bL)} = {bv['params']} params")
    else:
        print("  => no (window, depth) cell is monotone above its boundary")

    return {
        "note": spec["note"], "modulation": mod, "memory": spec["memory"],
        "baseline": base_name, "baseline_diagnostics": diag,
        "baseline_ber": [float(b) for b in base_ber],
        "df_ber": [float(b) for b in df_ber],
        "af_ber": [float(b) for b in af_ber],
        "cells": {f"w{w}_L{L}": v for (w, L), v in cells.items()},
        "min_params_any": best_any[1]["params"] if best_any else None,
        "min_config_any": ({"window": best_any[0][0], "depth": best_any[0][1],
                            "width": best_any[1]["width"],
                            "arch": best_any[1]["arch"],
                            "monotone": best_any[1]["monotone"]}
                           if best_any else None),
        "min_params": best[1]["params"] if best else None,
        "min_window": best[0][0] if best else None,
        "min_depth": best[0][1] if best else None,
        "min_width": best[1]["width"] if best else None,
        "min_arch": best[1]["arch"] if best else None,
        "probes": [v for v in cache.values()],
        "n_probes": len(cache),
    }


def main():
    only = sys.argv[1:] or list(base.CHANNELS)
    print("Minimum MLP size by bisection on hidden width")
    print(f"windows {WINDOWS} x depths {DEPTHS} swept exhaustively; "
          f"width bisected in 1..{H_MAX}")
    print(f"{len(TRAIN_SEEDS)} inits per probe; a size matches only if every "
          f"init passes both criteria")

    path = "results/mlp_min_size_bisect.json"
    out = {"windows": WINDOWS, "depths": DEPTHS, "h_max": H_MAX,
           "train_seeds": TRAIN_SEEDS,
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
    print(f"  {'channel':<14} {'base':>5} {'valid':>12} {'any':>6} "
          f"{'w':>2}  {'architecture':<24} {'thresh':>7} {'probes':>7}")
    for n, r in out["channels"].items():
        a = r.get("min_config_any")
        atag = "" if not a else ("" if a["monotone"] else "!")
        arch = a["arch"] if a else "--"
        win = str(a["window"]) if a else "--"
        print(f"  {n:<14} {r['baseline']:>5} "
              f"{r['baseline_diagnostics']['verdict']:>12} "
              f"{str(r['min_params_any']) + atag:>6} {win:>2}  {arch:<24} "
              f"{str(r['min_params']):>7} {r['n_probes']:>7}")
    print("\n  any    = smallest configuration that matched at all")
    print("  thresh = smallest whose (window, depth) cell also matched at every")
    print("           wider width probed -- i.e. a threshold, not just a point")
    print("  !      = the 'any' winner is non-monotone: wider nets stop matching")
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
