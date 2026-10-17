"""Regression checks for planning, fresh evidence, failure handling and RNG state.

Synthetic committed repositories exercise the real detached-worktree runner.
No full-budget training or rare-event study is launched by these tests.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import reproduce_all as RA  # noqa: E402
from provenance_audit import REGISTRY  # noqa: E402


def test_plan_covers_registry_without_executing_historical_archives():
    plan = RA.build_plan()
    planned = {o for s in plan for o in s['outputs']}
    assert {o for _, outputs, _ in REGISTRY.values() for o in outputs} <= planned
    archives = [s for s in plan if s['kind'] == 'archive_reference']
    assert {s['script'] for s in archives} == set(RA.ARCHIVE_SCRIPTS)
    assert all(not s['after'] for s in archives)
    assert all((ROOT / s['script']).is_file() for s in plan)
    positions = {s['name']: k for k, s in enumerate(plan)}
    assert all(positions[dep] < k for k, s in enumerate(plan) for dep in s['after'])


def test_rare_event_replication_has_distinct_command_and_source_seed():
    runs = [s for s in RA.build_plan() if s['script'] == 'rare_event_fixed_budget.py']
    assert len(runs) == 2
    for s in runs:
        seed = '7000' if 'replicate' in s['outputs'][0] else '1000'
        assert s['args'] == ['--src-base', seed, '--out', s['outputs'][0]]


def test_shared_writers_are_ordered_and_matched_configuration_is_explicit():
    plan = RA.build_plan()
    previous = None
    for s in plan:
        if 'results/coded_df_experiment.json' in s['outputs']:
            if previous:
                assert previous in s['after']
            previous = s['name']
        if any(o.endswith('_matched.json') for o in s['outputs']):
            if 'matched' not in s['script']:
                assert s['env'] == RA.MATCHED_ENV
            elif 'CODED_METRIC' in RA.source(s['script']):
                assert s['env']['CODED_METRIC'] == 'matched'


def test_selection_discloses_omitted_dependencies():
    plan = RA.build_plan()
    step = next(s for s in plan if s['after'])
    selected = RA.select(plan, only='^' + __import__('re').escape(step['name']) + ' ')
    assert len(selected) == 1
    assert selected[0]['archived_dependencies'] == step['after']


def test_default_invocation_runs_nothing():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_all.py'), '--only', 'MAC accounting'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Nothing was run' in result.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    (root / 'results').mkdir()
    shutil.copy2(ROOT / 'scripts/reproduce_all_worker.py', root / 'scripts')
    (root / 'results/value.json').write_text('{"value": 1}\n')
    (root / 'noop.py').write_text('pass\n')
    (root / 'write.py').write_text("from pathlib import Path\nPath('results/value.json').write_text('{\"value\": 1}\\n')\n")
    (root / 'second.py').write_text("from pathlib import Path\np=Path('results/value.json')\np.write_text(p.read_text().replace('1', '2'))\n")
    (root / 'fail.py').write_text('raise SystemExit(7)\n')
    (root / 'partial.py').write_text((root / 'write.py').read_text() + 'raise SystemExit(7)\n')
    (root / 'random.py').write_text("""import json, os, random, numpy as np
from pathlib import Path
Path('results/value.json').write_text(json.dumps({'python': random.random(), 'numpy': np.random.random(), 'metric': os.getenv('CODED_METRIC')}))
""")
    # Avoid shadowing Python's random module in the isolated repository.
    (root / 'random.py').rename(root / 'random_driver.py')
    (root / 'verify_thesis_tables.py').write_text('raise SystemExit(0)\n')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    monkeypatch.setattr(RA, 'ROOT', str(root))
    RA.git('config', 'user.name', 'Fixture')
    RA.git('config', 'user.email', 'fixture@example.invalid')
    RA.git('add', '.')
    RA.git('commit', '-qm', 'committed fixture')
    return root


def step(script='write.py', *, name='producer', after=None, kind='experiment', needs_torch=False):
    return dict(name=name, script=script, args=[], env={}, outputs=['results/value.json'],
                kind=kind, after=after or [], needs_torch=needs_torch, reason='historical reference')


def execute(repo, tmp_path, plan=None, *, verify=False, suffix='one', **kwargs):
    ev, wt = tmp_path / ('evidence-' + suffix), tmp_path / ('wt-' + suffix)
    before = (repo / 'results/value.json').read_bytes()
    result = RA.run(plan or [step()], str(ev), str(wt), verify=verify, **kwargs)
    assert (repo / 'results/value.json').read_bytes() == before
    assert not wt.exists()
    return result, ev


def test_fresh_output_matches_and_does_not_touch_published_results(repo, tmp_path):
    (summary, comparison, meta), ev = execute(repo, tmp_path)
    assert summary[0]['status'] == 'COMPLETED'
    assert comparison[0]['status'] == 'BYTE_MATCH'
    assert meta['exit_code'] == 0
    assert (ev / comparison[0]['evidence']).is_file()
    assert json.loads((ev / summary[0]['evidence'] / 'rng_before.json').read_text())['bootstrap_seed'] == 42


def test_noop_cannot_match_an_untouched_committed_output(repo, tmp_path):
    (summary, comparison, meta), _ = execute(repo, tmp_path, [step('noop.py')])
    assert summary[0]['returncode'] == 0
    assert summary[0]['status'] == 'FAILED'
    assert comparison[0]['status'] == 'MISSING'
    assert meta['exit_code'] == 1


def test_shared_writer_snapshots_are_preserved_and_noop_is_rejected(repo, tmp_path):
    plan = [step(), step('second.py', name='second', after=['producer']),
            step('noop.py', name='third', after=['second'])]
    (summary, comparison, meta), ev = execute(repo, tmp_path, plan)
    assert [c['status'] for c in comparison] == ['BYTE_MATCH', 'BYTES_DIFFER_REVIEW', 'NOT_REGENERATED']
    assert [json.loads((ev / c['evidence']).read_text())['value'] for c in comparison] == [1, 2, 2]
    assert all(hashlib.sha256((ev / c['evidence']).read_bytes()).hexdigest() == c['observed_sha256'] for c in comparison)
    assert meta['exit_code'] == 1


def test_failure_blocks_dependents_and_final_verifier(repo, tmp_path):
    (summary, _, meta), _ = execute(repo, tmp_path, [step('fail.py'), step(name='reader', after=['producer'])], verify=True)
    assert [s['status'] for s in summary] == ['FAILED', 'BLOCKED']
    assert meta['verify_status'] == 'BLOCKED_BY_FAILED_STEPS'
    assert meta['exit_code'] == 1


def test_partial_output_after_error_is_not_a_successful_match(repo, tmp_path):
    (summary, comparison, meta), _ = execute(repo, tmp_path, [step('partial.py')])
    assert summary[0]['returncode'] == 7
    assert comparison[0]['status'] == 'PARTIAL_OUTPUT'
    assert meta['exit_code'] == 1


def test_verifier_failure_is_an_error(repo, tmp_path):
    (repo / 'verify_thesis_tables.py').write_text('raise SystemExit(7)\n')
    RA.git('add', '.')
    RA.git('commit', '-qm', 'failing verifier')
    (_, comparison, meta), _ = execute(repo, tmp_path, verify=True)
    assert comparison[0]['status'] == 'BYTE_MATCH'
    assert meta['verify_returncode'] == 7
    assert meta['verify_status'] == 'FAILED'
    assert meta['exit_code'] == 1


def test_archives_are_not_executed_or_reported_as_reproduced(repo, tmp_path):
    (summary, comparison, meta), _ = execute(repo, tmp_path, [step('fail.py', kind='archive_reference')])
    assert summary[0]['returncode'] is None
    assert comparison[0]['status'] == 'ARCHIVE_REFERENCE'
    assert meta['exit_code'] == 0


def test_same_seed_replays_global_rng_and_clears_inherited_study_settings(repo, tmp_path, monkeypatch):
    monkeypatch.setenv('CODED_METRIC', 'matched')
    first, ev1 = execute(repo, tmp_path, [step('random_driver.py')], suffix='a')
    second, ev2 = execute(repo, tmp_path, [step('random_driver.py')], suffix='b')
    other, _ = execute(repo, tmp_path, [step('random_driver.py')], suffix='c', seed=43)
    assert first[1][0]['observed_sha256'] == second[1][0]['observed_sha256']
    assert other[1][0]['observed_sha256'] != first[1][0]['observed_sha256']
    result = json.loads((ev1 / first[1][0]['evidence']).read_text())
    assert result['metric'] is None
    assert first[2]['exit_code'] == 2
    for label in ('before', 'after'):
        a = json.loads((ev1 / first[0][0]['evidence'] / f'rng_{label}.json').read_text())
        b = json.loads((ev2 / second[0][0]['evidence'] / f'rng_{label}.json').read_text())
        assert a['python_state'] == b['python_state']
        assert a['numpy_legacy_state'] == b['numpy_legacy_state']


def test_torch_seed_state_and_output_repeat(repo, tmp_path):
    pytest.importorskip('torch')
    (repo / 'torch_driver.py').write_text("import torch, json\nfrom pathlib import Path\nPath('results/value.json').write_text(json.dumps(torch.nn.Linear(3,2).weight.detach().tolist()))\n")
    RA.git('add', '.')
    RA.git('commit', '-qm', 'torch fixture')
    first, ev1 = execute(repo, tmp_path, [step('torch_driver.py', needs_torch=True)], suffix='torch-a')
    second, ev2 = execute(repo, tmp_path, [step('torch_driver.py', needs_torch=True)], suffix='torch-b')
    assert first[1][0]['observed_sha256'] == second[1][0]['observed_sha256']
    a = json.loads((ev1 / first[0][0]['evidence'] / 'rng_before.json').read_text())
    b = json.loads((ev2 / second[0][0]['evidence'] / 'rng_before.json').read_text())
    assert a['torch']['initial_seed'] == 42
    assert a['torch']['cpu_state_hex'] == b['torch']['cpu_state_hex']
    assert a['torch']['deterministic_algorithms'] is True


def test_evidence_overwrite_and_dirty_source_are_rejected(repo, tmp_path):
    _, ev = execute(repo, tmp_path)
    with pytest.raises(FileExistsError):
        RA.run([step()], str(ev), str(tmp_path / 'wt-unused'), verify=False)
    (repo / 'write.py').write_text('pass\n')
    with pytest.raises(RuntimeError, match='Commit tracked changes'):
        RA.run([step()], str(tmp_path / 'new'), str(tmp_path / 'other'), verify=False)


def test_unsafe_output_path_is_rejected(repo, tmp_path):
    bad = step()
    bad['outputs'] = ['../escape.json']
    with pytest.raises(ValueError, match='unsafe artifact path'):
        RA.run([bad], str(tmp_path / 'new'), str(tmp_path / 'other'), verify=False)


def test_saved_checkpoints_and_extra_results_are_retained(repo, tmp_path):
    (repo / 'write.py').write_text((repo / 'write.py').read_text() +
        "Path('weights').mkdir(exist_ok=True)\nPath('weights/model.pt').write_bytes(b'checkpoint fixture')\n"
        "Path('results/extra.json').write_text('{}')\n")
    RA.git('add', '.')
    RA.git('commit', '-qm', 'checkpoint fixture')
    (summary, _, _), ev = execute(repo, tmp_path)
    extra = summary[0]['extra_outputs']
    assert {x['path'] for x in extra} == {'weights/model.pt', 'results/extra.json'}
    assert all(hashlib.sha256((ev / x['evidence']).read_bytes()).hexdigest() == x['sha256'] for x in extra)


def test_transitive_optional_torch_import_is_conservatively_detected(repo):
    (repo / 'helper.py').write_text('try:\n import torch\nexcept ImportError:\n pass\n')
    (repo / 'outer.py').write_text('import helper\n')
    assert RA.needs_torch('outer.py')
    (repo / 'outer.py').write_text('"""torch is mentioned only in documentation"""\npass\n')
    assert not RA.needs_torch('outer.py')


def test_cli_propagates_verifier_failure(repo, tmp_path, monkeypatch):
    (repo / 'verify_thesis_tables.py').write_text('raise SystemExit(7)\n')
    RA.git('add', '.')
    RA.git('commit', '-qm', 'CLI verifier failure')
    monkeypatch.setattr(RA, 'build_plan', lambda: [step()])
    monkeypatch.setattr(sys, 'argv', ['reproduce_all.py', '--run', '--evidence-dir', str(tmp_path / 'cli'),
                                    '--worktree-dir', str(tmp_path / 'cli-wt')])
    assert RA.main() == 1
