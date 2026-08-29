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
    return True


if __name__ == "__main__":
    _selftest()
