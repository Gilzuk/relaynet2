#!/usr/bin/env python3
"""Reanalyse archived round-2 runs; never retrain or overwrite thesis inputs.

Run from any directory: python scripts/assess_round2_reruns.py [--check].
The committed archive and original references are read together. The output
records fresh paired tests and comparisons, not historical p-values reused
for new checkpoints. --check rejects stale or edited assessment output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from canonical_paired_tests import paired
from penalty_common_targets import seeds_summary
from scripts.reproduce_round2 import compare

ARCHIVE = 'dc7c02c00cbebdfb566362a810ab4b45cb6abbe0'
SOURCE = 'c0394f0da5dac449594b89a9794fe251fad75e76'
RUN = 'gha-39024026594-1'
OUTPUT = ROOT / 'results/round2_rerun_assessment.json'


def load(path):
    return json.loads(path.read_text())


def assess():
    evidence = []
    def record(path, expected=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f'Artifact hash mismatch: {path}')
        evidence.append({'path': str(path.relative_to(ROOT)), 'sha256': digest,
                         'url': f'https://github.com/Gilzuk/relaynet2/blob/{ARCHIVE}/{path.relative_to(ROOT)}'})
    out = {'archive_commit': ARCHIVE, 'execution_source': SOURCE,
           'analysis_environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                    'scipy': scipy.__version__},
           'paired_test': 'Two-sided Wilcoxon signed-rank, uncorrected; conditional on each trained checkpoint. No equivalence test.',
           'jobs': {}, 'canonical': {}, 'sequence': {}, 'coded': {}, 'evidence': evidence}
    for name in ('canonical_native', 'canonical_3k', 'sequence_memory', 'coded_matched'):
        directory = ROOT / 'results/reproduction' / f'{RUN}-{name}' / name
        job = load(directory / 'job.json')
        if job['source_commit'] != SOURCE or job['returncode'] != 0:
            raise ValueError(f'Unexpected source or failed worker: {name}')
        for f in ('job.json', 'comparison.json'):
            record(directory / f)
        record(directory / 'run.log', job['log_sha256'])
        for a in job['artifacts']:
            record(directory / a['path'], a['sha256'])
        for a in job['reference_files']:
            record(directory / 'references' / a['path'], a['sha256'])
        recomputed = compare(job, directory)
        saved = load(directory / 'comparison.json')
        for k in ('status', 'difference_count', 'classical_control_differences'):
            if recomputed[k] != saved[k]:
                raise ValueError(f'Comparator disagrees for {name}: {k}')
        out['jobs'][name] = {k: job[k] for k in ('finished_at', 'elapsed_seconds', 'returncode')}
        out['jobs'][name].update({k: recomputed[k] for k in ('status', 'difference_count', 'classical_control_differences')})
        new = load(directory / 'outputs' / job['outputs'][0])
        old = load(directory / 'references' / job['reference_files'][0]['path'])
        if name.startswith('canonical_'):
            for model in ('AF', 'DF'):
                if not np.array_equal(old['results'][model]['ber_trials'], new['results'][model]['ber_trials']):
                    raise ValueError(f'Classical trials differ: {name}/{model}')
            snrs = new['snr_range']
            pairs = {model: paired(v['ber_trials'], new['results']['DF']['ber_trials'], snrs)
                     for model, v in new['results'].items() if model not in ('AF', 'DF')}
            reversals = []
            models = list(new['results'])
            for i, snr in enumerate(snrs):
                for aidx, a in enumerate(models):
                    for b in models[aidx + 1:]:
                        before = old['results'][a]['ber_mean'][i] - old['results'][b]['ber_mean'][i]
                        after = new['results'][a]['ber_mean'][i] - new['results'][b]['ber_mean'][i]
                        if abs(before) > 1e-14 and abs(after) > 1e-14 and before * after < 0:
                            reversals.append({'snr_db': snr, 'a': a, 'b': b,
                                              'historical_mean_difference': before, 'rerun_mean_difference': after})
            gaps = {}
            for model, rows in pairs.items():
                row = max(rows, key=lambda r: abs(r['mean_diff']))
                gaps[model] = {'snr_db': row['snr_db'], 'abs_mean_ber_gap': abs(row['mean_diff'])}
            out['canonical'][name] = {'paired_vs_DF': pairs, 'max_gap_vs_DF': gaps,
                                     'strict_pointwise_reversals': reversals}
        elif name == 'sequence_memory':
            published = load(directory / 'references/results/penalty_common_targets.json')['table_6_7']
            for channel, current in new['channels'].items():
                if not np.array_equal(old['channels'][channel]['baseline_ber'], current['baseline_ber']):
                    raise ValueError(f'Classical baseline differs: {channel}')
                summaries = {}
                for arch, rec in current['archs'].items():
                    before = seeds_summary(old['snr_db'], [r['ber'] for r in old['channels'][channel]['archs'][arch]['runs']], old['channels'][channel]['baseline_ber'])
                    after = seeds_summary(new['snr_db'], [r['ber'] for r in rec['runs']], current['baseline_ber'])
                    for k in ('new_best', 'new_spread', 'new_best_flag', 'new_spread_flag', 'unreached'):
                        if before[k] != published[channel][arch][k]:
                            raise ValueError(f'Historical summary mismatch: {channel}/{arch}/{k}')
                    summaries[arch] = {'historical': before, 'rerun': after}
                ranks = {label: sorted(summaries, key=lambda a: summaries[a][label]['new_best'])
                         for label in ('historical', 'rerun')}
                out['sequence'][channel] = {'summaries': summaries, 'best_of_three_order': ranks,
                                           'order_unchanged': ranks['historical'] == ranks['rerun']}
        else:
            for snr, rows in new['per_trial'].items():
                for model, values in rows.items():
                    if not model.startswith('mlp_') and not np.array_equal(old['per_trial'][snr][model], values):
                        raise ValueError(f'Classical coded trials differ: {snr}/{model}')
                out['coded'][snr] = {}
                for model in ('mlp_hard|matched', 'mlp_soft|matched', 'mlp_hard|plain', 'mlp_soft|plain'):
                    df, values = np.asarray(rows['coded_df_m|matched']), np.asarray(rows[model])
                    out['coded'][snr][model] = {
                        'historical_mean': float(np.mean(old['per_trial'][snr][model])),
                        'rerun_mean': float(values.mean()), 'matched_DF_mean': float(df.mean()),
                        'DF_wins': int(np.sum(df < values)), 'ties': int(np.sum(df == values)),
                        'paired_vs_DF': paired([values], [df], [int(snr)])[0]}
    seed_dir = ROOT / 'results/reproduction/20261017-rng-seed42'
    manifest = load(seed_dir / 'manifest.json')
    for artifact in manifest['artifacts']:
        path = seed_dir / artifact['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError(f'Seed-control hash mismatch: {path}')
    first, second = (load(seed_dir / f'all_seeded_{i}/result.json') for i in (1, 2))
    out['same_host_control'] = {
        'seed': first['torch_initial_seed'],
        'trial_ber_count': int(np.asarray(first['ber_trials']).size),
        'different_trial_ber_values': int(np.count_nonzero(np.asarray(first['ber_trials']) != np.asarray(second['ber_trials']))),
        'initial_weights_equal': first['initial_weights'] == second['initial_weights'],
        'final_weights_equal': first['final_weights'] == second['final_weights'],
        'scope': 'Native MLP only; full original budget; same host/environment; new declared seed42.'}
    out['limits'] = ('Historical canonical Torch states/checkpoints are absent. Seeded sequence/coded discrepancies remain unexplained; '
                     'they cannot be attributed to missing canonical Torch seeding alone. Fixed-checkpoint paired tests do not measure '
                     'training-population uncertainty. Minimum-size training grid, pilot studies, adaptive tails and matched Mamba were not rerun.')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    result = assess()
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.check:
        # Package versions are recorded for the generating analysis; numerical
        # comparisons must still agree when checked in another environment.
        saved = load(OUTPUT)
        saved.pop('analysis_environment', None)
        result.pop('analysis_environment', None)
        if saved != result:
            raise SystemExit('Assessment differs: inspect differences before replacing it')
        print('Archived hashes, fresh comparisons and saved assessment agree')
    else:
        OUTPUT.write_text(text)
        print(OUTPUT)


if __name__ == '__main__':
    main()
