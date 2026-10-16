"""Comparable-target SNR penalties (ber_metrics.penalty_common and helpers)."""
import math

from ber_metrics import penalty_table, penalty_common, best_of_seeds, spread_of_seeds

SNR = [0, 4, 8, 12, 16, 20]
REF = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]           # reaches 1e-1, 1e-2 and 1e-3
SHIFTED = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]       # REF delayed by exactly 4 dB
FLOORED = [3e-1, 1e-1, 3e-2, 1e-2, 5e-3, 4e-3]       # never reaches 1e-3


def test_exact_shift_matches_published_rule():
    p = penalty_common(SNR, SHIFTED, REF)
    assert not p["lower_bound"] and p["unreached"] == []
    assert math.isclose(p["worst_db"], 4.0, abs_tol=1e-9)
    assert math.isclose(p["worst_db"], penalty_table(SNR, SHIFTED, REF)["worst_db_penalty"], abs_tol=1e-12)


def test_missed_target_is_kept_as_a_lower_bound():
    old = penalty_table(SNR, FLOORED, REF)
    assert 1e-3 not in old["targets_reached"]          # the published rule drops it
    p = penalty_common(SNR, FLOORED, REF)
    assert p["common_targets"] == [1e-1, 1e-2, 1e-3]
    assert p["unreached"] == [1e-3] and p["lower_bound"]
    # it needs more than 20 dB where REF needs 16, so the bound is 4 dB
    assert math.isclose(p["per_target"][1e-3]["db"], 4.0, abs_tol=1e-9)
    assert p["worst_db"] >= old["worst_db_penalty"]


def test_targets_the_reference_misses_are_not_imposed():
    ref = [3e-1, 2e-1, 1.5e-1, 1.2e-1, 1.1e-1, 5e-2]   # reaches 1e-1 only
    p = penalty_common(SNR, SHIFTED, ref)
    assert p["common_targets"] == [1e-1] and not p["lower_bound"]


def test_best_and_spread_honour_bounds():
    exact = penalty_common(SNR, SHIFTED, REF)          # 4.0, exact
    bounded = penalty_common(SNR, FLOORED, REF)        # >= 4.0 at 1e-3
    best, flag = best_of_seeds([bounded, exact])
    assert flag == "" and math.isclose(best, exact["worst_db"])
    spread, sflag = spread_of_seeds([exact, bounded])
    assert sflag == ">" and spread >= 0.0
    assert spread_of_seeds([bounded, bounded])[1] == "nan"
    assert spread_of_seeds([exact, exact]) == (0.0, "")
