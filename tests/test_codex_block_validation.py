import json
import numpy as np
import pytest

from relaynet.codex_block_statistics import summarize_blocks, summarize_seed_means
from codex_fixed_budget_validation import main, evaluate_block


def test_zero_errors_do_not_produce_zero_width_certainty():
    s = summarize_blocks([0] * 10, 10000)
    assert s['bits'] == 100000
    assert s['block_t_ci95_approx'] is None
    assert s['zero_error_block_upper95'] == pytest.approx(1 - 0.05 ** 0.1)


def test_sampling_unit_is_block_then_seed():
    s = summarize_blocks([0, 100, 0, 100], 100)
    assert s['ber'] == 0.5
    assert s['block_t_ci95_approx'] == [0, 1]
    assert summarize_seed_means([0.1, 0.2, 0.3])['training_seeds'] == 3
    with pytest.raises(ValueError):
        summarize_blocks([1.5], 10)


def test_plan_exposure_and_no_overwrite(tmp_path):
    out = tmp_path / 'codex_smoke'
    args = ['--output', str(out), '--snrs', '8', '--seeds', '3',
            '--bits-per-seed', '60', '--block-bits', '20', '--train-samples', '30', '--epochs', '1']
    main(args)
    records = [json.loads(line) for line in (out / 'counts.jsonl').read_text().splitlines()]
    assert len(records) == 3 and sum(r['bits'] for r in records) == 60
    assert json.loads((out / 'summary.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        main(args)


def test_evaluation_is_reproducible_and_paired():
    class DF:
        def process(self, y):
            return np.where(y < 0, -1., 1.)
    first = evaluate_block(DF(), 0, 0, 8, 0, 1000)
    assert first == evaluate_block(DF(), 0, 0, 8, 0, 1000)
    assert first['mlp'] == first['df']
