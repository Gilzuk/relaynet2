"""SNR penalty in dB at a target BER -- the metric this study should have used.

A relative BER penalty, (ber_relay - ber_base) / ber_base, divides by a
denominator that shrinks with SNR, so the same physical gap reads as a
wildly different number depending where on the curve it is measured. This
study produced several figures that are arithmetically correct and
practically meaningless because of it:

  awgn          "no configuration matches DF, +11% to +28%" on a worst
                absolute gap of 0.0004 BER, against an ML-optimal baseline
  isi           "+12470%" at window 1, because MLSE drives BER to zero and
                anything finite divided by ~0 is enormous
  Table 5.2     the thesis's own published MLP-169 scores +2.06% at 20 dB on
                an absolute gap of 0.0002, marginally failing a 2% bar

The standard communications metric does not have this failure mode: how much
extra SNR does the relay need to reach the same BER. It is scale-free, stays
finite and interpretable wherever the curves are defined, and is what a
reader can act on -- "costs 0.4 dB" means something, "+2.06%" does not.

  snr_at_ber   invert a BER curve: the SNR at which it crosses a target,
               by linear interpolation in log(BER) against SNR, which is
               close to straight for the curves here
  db_penalty   snr_at_ber(relay) - snr_at_ber(baseline), positive meaning
               the relay needs more SNR

WHAT IT CANNOT DO. A target below the floor of either curve, or above its
start, is not bracketed by the data and no amount of interpolation invents
it. Those cases return NaN with a reason rather than an extrapolated number,
and callers are expected to report the NaN rather than silently drop the
point -- a target that only some configurations reach would otherwise
quietly compare different subsets.
"""

import numpy as np

DEFAULT_TARGETS = (1e-1, 1e-2, 1e-3)


def snr_at_ber(snr_db, ber, target):
    """SNR (dB) at which a BER curve crosses `target`, or NaN if unbracketed.

    Interpolates linearly in log(BER) versus SNR between the two points that
    bracket the target. Uses the *last* crossing, so a curve that is not
    perfectly monotone (Monte Carlo noise, or a genuinely non-monotone
    baseline) resolves to its high-SNR behaviour rather than an early dip.
    """
    snr = np.asarray(snr_db, dtype=float)
    b = np.asarray(ber, dtype=float)
    if snr.size != b.size or snr.size < 2:
        return float("nan")

    cross = None
    for i in range(len(b) - 1):
        lo, hi = b[i], b[i + 1]
        if lo <= 0 or hi <= 0:
            # a zero-BER point cannot be interpolated in log space; if the
            # target sits above it the crossing is still bracketed, so fall
            # back to the last finite point below the target
            if hi <= 0 and lo > target:
                cross = snr[i + 1]
            continue
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            t = (np.log(target) - np.log(lo)) / (np.log(hi) - np.log(lo))
            cross = snr[i] + t * (snr[i + 1] - snr[i])
    return float(cross) if cross is not None else float("nan")


def db_penalty(snr_db, ber_relay, ber_base, target):
    """Extra SNR (dB) the relay needs to reach `target`. NaN if either curve
    does not bracket it. Positive means the relay is worse."""
    a = snr_at_ber(snr_db, ber_relay, target)
    b = snr_at_ber(snr_db, ber_base, target)
    if a != a or b != b:
        return float("nan")
    return a - b


def penalty_table(snr_db, ber_relay, ber_base, targets=DEFAULT_TARGETS):
    """dB penalty at each target, plus which targets were reachable."""
    out = {}
    for t in targets:
        out[t] = {
            "snr_relay": snr_at_ber(snr_db, ber_relay, t),
            "snr_base": snr_at_ber(snr_db, ber_base, t),
            "db_penalty": db_penalty(snr_db, ber_relay, ber_base, t),
        }
    reached = [t for t, v in out.items() if v["db_penalty"] == v["db_penalty"]]
    return {"per_target": out, "targets_reached": reached,
            "worst_db_penalty": (max(out[t]["db_penalty"] for t in reached)
                                 if reached else float("nan"))}


def penalty_common(snr_db, ber_relay, ber_ref, targets=DEFAULT_TARGETS):
    """Worst penalty over the targets the *reference* reaches: comparable sets.

    `penalty_table` keeps only the targets both curves bracket, so a relay that
    never reaches 1e-3 is scored on 1e-1 and 1e-2 alone and can look better than
    one that reaches every target. Here the target set is fixed by the
    reference curve, so every relay compared against it is held to the same
    targets. For each target t the reference reaches inside the tested range:

      ok           the relay reaches t too: penalty = snr_relay - snr_ref;
      not_reached  the relay is still above t at the last SNR, so it needs more
                   than snr_max: the penalty is only bounded below, by
                   snr_max - snr_ref;
      below_range  the relay is already below t at the first SNR: the penalty
                   is bounded above by snr_min - snr_ref.

    `worst_db` is the largest exact penalty or lower bound over the common set;
    `lower_bound` is True when any target is not reached, in which case the
    true worst penalty is at least `worst_db`.
    """
    snr = np.asarray(snr_db, dtype=float)
    br = np.asarray(ber_relay, dtype=float)
    per, exact, lbs = {}, [], []
    common = []
    for t in targets:
        b = snr_at_ber(snr, ber_ref, t)
        if b != b:
            continue
        common.append(t)
        a = snr_at_ber(snr, br, t)
        if a == a:
            per[t] = {"status": "ok", "db": a - b}
            exact.append(a - b)
        elif br[-1] > t:
            per[t] = {"status": "not_reached", "db": float(snr[-1] - b)}
            lbs.append(float(snr[-1] - b))
        elif br[0] < t:
            per[t] = {"status": "below_range", "db": float(snr[0] - b)}
            exact.append(float(snr[0] - b))
        else:
            per[t] = {"status": "unbracketed", "db": float("nan")}
    vals = exact + lbs
    return {"common_targets": common, "per_target": per,
            "unreached": [t for t, v in per.items() if v["status"] == "not_reached"],
            "worst_db": float(max(vals)) if vals else float("nan"),
            "lower_bound": bool(lbs)}


def best_of_seeds(results):
    """Best (smallest) worst-case penalty over seeds, honouring lower bounds.

    `results` is a list of `penalty_common` outputs. Returns (value, flag):
    flag is "" when exact, ">" when the best is itself only bounded below
    (every seed misses a target, or a missing seed's bound is the smallest),
    "nan" when nothing is defined.
    """
    ex = [r["worst_db"] for r in results if r["worst_db"] == r["worst_db"] and not r["lower_bound"]]
    lb = [r["worst_db"] for r in results if r["worst_db"] == r["worst_db"] and r["lower_bound"]]
    if ex and (not lb or min(ex) <= min(lb)):
        return float(min(ex)), ""
    if lb:
        return float(min(lb)), ">"
    return float("nan"), "nan"


def spread_of_seeds(results):
    """Largest minus smallest worst-case penalty over seeds, honouring bounds.

    Returns (value, flag): "" exact; ">" when some seed misses a target, so the
    spread is at least the value (the true smallest is at most the smallest
    exact value, the true largest at least the largest value or bound); "nan"
    when no seed is exact (the spread is then not determined by the data).
    """
    ex = [r["worst_db"] for r in results if r["worst_db"] == r["worst_db"] and not r["lower_bound"]]
    lb = [r["worst_db"] for r in results if r["worst_db"] == r["worst_db"] and r["lower_bound"]]
    if not ex:
        return float("nan"), "nan"
    if lb:
        return float(max(ex + lb) - min(ex)), ">"
    return float(max(ex) - min(ex)), ""


def hierarchical_ci(samples, n_seeds, n_trials, conf=0.95):
    """95% CI for a mean over `n_seeds` trained networks x `n_trials` runs each.

    The obvious thing -- 1.96 * std(all 30 samples) / sqrt(30) -- is wrong here,
    and wrong in the direction that flatters the result. The 10 inference trials
    inside one seed share a single trained network: they resample Monte Carlo
    noise but not training variance, so they are not 30 independent draws from
    the population the claim is about ("a relay of this architecture, trained
    this way"). Treating them as independent divides by sqrt(30) when the
    effective sample size is closer to the 3 seeds, and the interval comes out
    several times too narrow.

    The correct unit of replication is the seed. Average within each seed, then
    put a Student-t interval on those `n_seeds` means. With 3 seeds that is
    t(2, 0.975) = 4.303 rather than 1.96 -- deliberately wide, because three
    points genuinely say little about a spread.

    Parameters
    ----------
    samples : array of shape (n_seeds * n_trials,), seed-major
        Column `s * n_trials + t` is trial `t` of seed `s`, matching the layout
        the E6 runners produce.
    n_seeds, n_trials : int
    conf : float, the two-sided confidence level.

    Returns
    -------
    dict with `mean`, `ci` (half-width, hierarchical), `seed_means`,
    `between_seed_std`, and `ci_pooled` -- the naive interval, kept so the two
    can be reported side by side rather than one silently replacing the other.
    """
    from scipy import stats

    x = np.asarray(samples, dtype=float).ravel()
    if x.size != n_seeds * n_trials:
        raise ValueError(f"expected {n_seeds * n_trials} samples, got {x.size}")
    seed_means = x.reshape(n_seeds, n_trials).mean(axis=1)
    mean = float(seed_means.mean())

    if n_seeds < 2:
        ci = float("nan")
        sd = float("nan")
    else:
        sd = float(seed_means.std(ddof=1))
        tcrit = float(stats.t.ppf(0.5 + conf / 2.0, n_seeds - 1))
        ci = tcrit * sd / np.sqrt(n_seeds)

    zcrit = float(stats.norm.ppf(0.5 + conf / 2.0))
    ci_pooled = float(zcrit * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")

    return {"mean": mean, "ci": float(ci), "seed_means": seed_means.tolist(),
            "between_seed_std": sd, "ci_pooled": ci_pooled,
            "n_seeds": int(n_seeds), "n_trials": int(n_trials)}


def _selftest():
    snr = [0, 4, 8, 12, 16, 20]

    # an exactly-known shift: shifting a curve right by 4 dB must read as 4 dB
    base = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    shifted = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]     # base delayed one step
    for t in (1e-1, 1e-2, 1e-3):
        d = db_penalty(snr, shifted, base, t)
        assert abs(d - 4.0) < 1e-6, f"expected 4 dB at {t}, got {d}"
    print("  known 4 dB shift recovered at every target: OK")

    # identical curves must read as zero
    assert abs(db_penalty(snr, base, base, 1e-2)) < 1e-12
    print("  identical curves give 0 dB: OK")

    # unreachable target returns NaN rather than an extrapolation
    d = db_penalty(snr, base, base, 1e-9)
    assert d != d, f"expected NaN for an unreachable target, got {d}"
    print("  unreachable target returns NaN: OK")

    # the real case this metric exists for: the published Table 5.2 numbers,
    # where a 2.06% relative penalty at 20 dB should be a fraction of a dB
    df = [0.3337, 0.2219, 0.1218, 0.0580, 0.0245, 0.0097]
    mlp = [0.3349, 0.2233, 0.1229, 0.0586, 0.0247, 0.0099]
    p = penalty_table(snr, mlp, df)
    for t in p["targets_reached"]:
        d = p["per_target"][t]["db_penalty"]
        print(f"  Table 5.2 MLP-169 vs DF at BER {t:g}: {d:+.3f} dB")

    # hierarchical CI: with no between-seed spread the two intervals should be
    # close; with real spread the hierarchical one must be substantially wider,
    # since that spread is exactly what pooling hides.
    rng = np.random.default_rng(0)
    tight = np.concatenate([rng.normal(0.0064, 1e-5, 10) for _ in range(3)])
    h = hierarchical_ci(tight, 3, 10)
    assert h["ci"] < 5 * h["ci_pooled"], "no real spread should not blow up the CI"
    print(f"  hierarchical CI, no between-seed spread: "
          f"{h['ci']:.2e} vs pooled {h['ci_pooled']:.2e}")

    # the S1 8 dB case that prompted this: seed means 0.0061 / 0.0064 / 0.0073
    spread = np.concatenate([rng.normal(m, 1e-5, 10)
                             for m in (0.0061, 0.0064, 0.0073)])
    h = hierarchical_ci(spread, 3, 10)
    assert h["ci"] > 5 * h["ci_pooled"], "real between-seed spread must widen it"
    print(f"  hierarchical CI, S1 8 dB spread: {h['ci']:.2e} vs pooled "
          f"{h['ci_pooled']:.2e} ({h['ci'] / h['ci_pooled']:.1f}x wider)")

    # the seed means must be recovered exactly, not approximately
    assert np.allclose(h["seed_means"],
                       spread.reshape(3, 10).mean(axis=1)), "seed means wrong"
    print("  seed means recovered from the seed-major layout: OK")
    return True


if __name__ == "__main__":
    _selftest()
