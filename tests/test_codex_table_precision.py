import pytest

from verify_thesis_tables import Report, tol_for


@pytest.mark.parametrize("text,expected", [
    (r"$1.2\times10^{-7}$", 5e-9),
    (r"$1.46\times10^{-4}$", 5e-7),
    ("1.20e-7", 5e-10),
    ("5e-5", 5e-6),
    ("0.1234", 5e-5),
    ("170", .5),
])
def test_rounding_tolerance_includes_exponent(text, expected):
    assert tol_for(text) == pytest.approx(expected, rel=1e-10, abs=0)


def test_large_scientific_notation_error_rejected():
    report = Report()
    report.cell("tbl:tableE6", "regression", r"1.2\times10^{-7}", 1.2e-7, .01)
    assert len(report.flags) == 1


@pytest.mark.parametrize("source,passes", [(4.9e-5, True), (5e-5, False), (5.1e-5, False)])
def test_upper_bound_has_no_rounding_slack(source, passes):
    report = Report()
    report.cell("tbl:tableE6", "bound", "<5e-5", "<5e-5", source)
    assert (not report.flags) == passes
