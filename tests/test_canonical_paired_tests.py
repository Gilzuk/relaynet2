"""Regression checks for ties in discrete-error BER signed-rank tests."""
import numpy as np
import pytest

from canonical_paired_tests import paired


def test_native_s6_ties_match_exhaustive_signed_rank_probability():
    # Archived native S6 minus DF, 16 dB: tied +/-1 and +/-2 counts must
    # share ranks. Direct floating BER subtraction produced p=0.03515625.
    differences = np.array([9, 2, 8, 5, -1, 29, -2, 1, 4, -1])
    df_counts = np.array([250, 251, 252, 253, 254, 255, 256, 257, 258, 259])
    for budget in (10000, 100000):
        row = paired([(df_counts + differences) / budget], [df_counts / budget],
                     [16], bits_per_trial=budget)[0]
        # Independent enumeration: 56 of 1024 sign assignments are as
        # extreme in the two tails, after assigning tied midranks.
        assert row['p'] == 56 / 1024
        assert not row['resolved']
        assert row['mean_diff'] == pytest.approx(differences.mean() / budget)


def test_identical_trials_are_unresolved():
    row = paired([[0.01, 0.02]], [[0.01, 0.02]], [0], bits_per_trial=10000)[0]
    assert row['p'] == 1
    assert not row['resolved']


def test_coded_budget_cannot_silently_use_canonical_denominator():
    with pytest.raises(ValueError, match='integer counts'):
        paired([[0.00001]], [[0.0]], [20], bits_per_trial=10000)
    row = paired([[0.00001]], [[0.0]], [20], bits_per_trial=100000)[0]
    assert row['mean_diff'] == 0.00001


def test_all_saved_ten_pair_pvalues_against_independent_sign_enumeration():
    """Audit all 418 canonical contrasts without using SciPy for the oracle."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def load(path):
        return json.loads((root / path).read_text())

    def exact(a, b):
        d = np.rint(np.array(a) * 10000).astype(int) - np.rint(np.array(b) * 10000).astype(int)
        d = d[d != 0]
        if not len(d):
            return 1.0
        absolute = np.abs(d)
        # Twice each midrank keeps the enumeration integer-valued.
        ranks = [2 * np.count_nonzero(absolute < x) + np.count_nonzero(absolute == x) + 1
                 for x in absolute]
        observed = sum(rank for rank, difference in zip(ranks, d) if difference > 0)
        counts = {0: 1}
        for rank in ranks:
            updated = counts.copy()
            for subtotal, count in counts.items():
                updated[subtotal + int(rank)] = updated.get(subtotal + int(rank), 0) + count
            counts = updated
        lower = sum(count for subtotal, count in counts.items() if subtotal <= observed)
        upper = sum(count for subtotal, count in counts.items() if subtotal >= observed)
        return min(1.0, 2 * min(lower, upper) / 2 ** len(d))

    native = load('results/modulation/qpsk_rayleigh.json')['results']
    mlp169 = native['MLP (169p)']['ber_trials']
    saved = load('results/canonical_paired_tests.json')['tables']
    checked = 0
    for table, results in [('table_5_2', native),
                           ('table_5_3', load('results/normalized_3k/3k_rayleigh.json')['results'])]:
        for group in ('vs_DF', 'vs_MLP169', 'feedforward_pairs'):
            for model, rows in saved[table].get(group, {}).items():
                if group == 'feedforward_pairs':
                    a, b = (results[k]['ber_trials'] for k in model.split(' - '))
                else:
                    a = results[model]['ber_trials']
                    b = results['DF']['ber_trials'] if group == 'vs_DF' else mlp169
                for av, bv, row in zip(a, b, rows):
                    assert row['p'] == exact(av, bv), (table, group, model, row)
                    checked += 1
    saved = load('results/round2_rerun_assessment.json')['canonical']
    for name in ('canonical_native', 'canonical_3k'):
        directory = Path('results/reproduction') / f'gha-39024026594-1-{name}' / name
        job = load(directory / 'job.json')
        results = load(directory / 'outputs' / job['outputs'][0])['results']
        for model, rows in saved[name]['paired_vs_DF'].items():
            for av, bv, row in zip(results[model]['ber_trials'], results['DF']['ber_trials'], rows):
                assert row['p'] == exact(av, bv), (name, model, row)
                checked += 1
    assert checked == 418
