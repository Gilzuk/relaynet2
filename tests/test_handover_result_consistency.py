import pytest

from scripts.plot_e6_unknown_channel_awgn import matched_points
from verify_thesis_tables import Report, clean_cell, matched_viterbi_counts, tol_for


@pytest.mark.parametrize("name", ["VIT-genie", "VIT-est"])
def test_original_figure_uses_same_counts_as_table(name):
    snrs, rates, budgets = matched_points(name)
    for snr, rate, budget in zip(snrs, rates, budgets):
        errors, bits = matched_viterbi_counts(name, int(snr))
        assert budget == bits
        assert rate == errors / bits


@pytest.mark.parametrize("text,valid", [
    (r"$<3.0\times10^{-8}$", True),
    (r"$<3.0\times10^{-10}$", False),
    (r"$<3.0\times10^{-6}$", False),
    ("0", False),
    (r"$3.0\times10^{-8}$", False),
])
def test_zero_error_bound_requires_matching_exposure(text, valid):
    report = Report()
    raw, value = clean_cell(text)
    report.zero_error_bound("test", "16dB", raw, value, 100_000_000)
    assert (not report.flags) == valid


def test_rare_event_scientific_precision_is_not_mantissa_precision():
    assert tol_for("1.20e-7") == pytest.approx(5e-10)
    report = Report()
    report.cell("tbl:layers", "MLP16", "1.20e-7", 1.2e-7, 9.2e-7)
    assert report.flags
