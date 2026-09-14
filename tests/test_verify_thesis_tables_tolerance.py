"""Regression tests for the table checker's tolerance and cell parsing.

A manuscript audit on 2026-09-14 found that ``tol_for`` counted decimals in
the mantissa and ignored the exponent, so a cell printed as ``1.2\\times10^-7``
was checked against an absolute tolerance of 0.05. Eighteen published cells
were being "verified" that way, the worst inflated by a factor of 1e10. None
of them was in fact wrong, which is exactly why the defect survived: the
checker reported 527 cells and zero inconsistencies while a large fraction of
its rare-event coverage proved nothing.

These tests pin both the scaling and the two parsing forms so that a
regression shows up as a failing test rather than as a green audit.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verify_thesis_tables import Report, clean_cell, tol_for


@pytest.mark.parametrize("text,expected", [
    # plain decimals: half a unit in the last place, unchanged behaviour
    ("0.0065", 5e-5),
    ("0.0580", 5e-5),
    ("2.3003", 5e-5),
    # LaTeX scientific notation, unbraced (what clean_cell produces)
    (r"1.20\times10^-7", 5e-10),
    (r"3.20\times10^-5", 5e-8),
    (r"<3.0\times10^-10", 5e-12),
    # braced, as written in the source before clean_cell strips them
    (r"1.2\times10^{-7}", 5e-9),
    (r"1.46\times10^{-4}", 5e-7),
    # plain e-notation
    ("3.3e-3", 5e-5),
    ("<5e-5", 5e-6),
])
def test_tolerance_scales_with_the_exponent(text, expected):
    # rel is loose enough to absorb the tie-guard epsilon, which is bounded
    # by min(1e-12, unit*1e-6) and is never the thing under test here.
    assert tol_for(text) == pytest.approx(expected, rel=1e-4)


def test_the_epsilon_never_dominates_the_unit_it_protects():
    """A tiny cell's tolerance must stay proportional, not bottom out at 1e-12."""
    assert tol_for(r"1.0\times10^-20") < 1e-19


@pytest.mark.parametrize("text,expected", [
    (r"1.20\times10^-7", 1.2e-7),
    (r"1.2\times10^{-7}", 1.2e-7),
    # a leading word used to defeat the anchored match, so the bound was read
    # off the mantissa alone and 3e-10 became 3
    (r"exploratory <3.0\times10^-10^\ddagger", "<3e-10"),
    (r"<3.0\times10^{-10}", "<3e-10"),
])
def test_scientific_cells_parse_to_their_full_value(text, expected):
    assert clean_cell(text)[1] == expected


@pytest.mark.parametrize("text,published,source", [
    (r"1.2\times10^{-7}", 1.2e-7, 0.01),        # 5 orders of magnitude out
    (r"1.46\times10^{-4}", 1.46e-4, 0.005),     # 34x out
    (r"3.20\times10^-5", 3.2e-5, 3.9e-5),       # 22% out
])
def test_a_wrong_scientific_cell_is_flagged(text, published, source):
    rep = Report()
    rep.cell("tbl:audit", "regression", text, published, source)
    assert len(rep.flags) == 1, "a materially wrong cell passed the check"


@pytest.mark.parametrize("text,published,source", [
    (r"1.20\times10^-7", 1.2e-7, 1.2e-7),
    (r"3.20\times10^-5", 3.2e-5, 3.2e-5),
    (r"1.456\times10^-4", 1.456e-4, 1.4562e-4),  # rounds to the printed digits
])
def test_a_correct_scientific_cell_still_passes(text, published, source):
    rep = Report()
    rep.cell("tbl:audit", "regression", text, published, source)
    assert rep.flags == []


def test_an_upper_bound_is_checked_against_the_bound_not_the_mantissa():
    """src=1.0 must fail a <3e-10 bound; before the fix the bound read as 3."""
    rep = Report()
    rep.cell("tbl:audit", "regression", r"exploratory <3.0\times10^-10", "<3e-10", 1.0)
    assert len(rep.flags) == 1
