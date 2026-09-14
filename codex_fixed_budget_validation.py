"""Fresh, fixed-budget Chapter 7 BPSK/ISI evaluation; never alters old results.

Run e.g. python codex_fixed_budget_validation.py --output results/codex_isi_run
The output directory must be new. A plan is saved before training or testing,
and each completed independent block is appended to counts.jsonl. No exposure
is chosen from observed errors. Three independently trained MLPs are the default.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from relaynet.channels.e6_channels import ISIChannel
from relaynet.relays.mlp import MLPRelay
from relaynet.codex_block_statistics import summarize_blocks, summarize_seed_means

TAPS = [1.0, 0.7, 0.5]
TRAIN_SNRS = [5, 10, 15]


def train(seed, samples, epochs):
    streams = np.random.SeedSequence([20260914, 0, seed]).spawn(2)
    rng = np.random.default_rng(streams[0])
    channel = ISIChannel(TAPS, rng=np.random.default_rng(streams[1]))
    relay = MLPRelay(11, 13, window_size=11, seed=seed)
    xs, ys = [], []
    for snr in TRAIN_SNRS:
        clean = 1.0 - 2.0 * rng.integers(0, 2, samples // len(TRAIN_SNRS))
        noisy = channel(clean, snr)
        xs.append(relay._extract_windows(noisy))
        ys.append(clean)
    relay.train_on_data(np.vstack(xs), np.concatenate(ys), epochs=epochs,
                        batch_size=256, lr=3e-3)
    return relay


def evaluate_block(relay, seed, snr_index, snr, block_index, bits):
    # Separate training/evaluation streams; common random draws across relays.
    streams = np.random.SeedSequence([20260914, 1, seed, snr_index, block_index]).spawn(3)
    rng = np.random.default_rng(streams[0])
    clean = 1.0 - 2.0 * rng.integers(0, 2, bits)
    channel = ISIChannel(TAPS, rng=np.random.default_rng(streams[1]))
    y = channel(clean, snr)
    z = np.random.default_rng(streams[2]).standard_normal(bits)
    outputs = {"af": y / np.sqrt(np.mean(y * y)),
               "df": np.where(y < 0, -1.0, 1.0), "mlp": relay.process(y)}
    counts = {}
    for name, x in outputs.items():
        received = x + z * np.sqrt(np.mean(x * x) / (2 * 10 ** (snr / 10)))
        counts[name] = int(np.count_nonzero((received < 0) != (clean < 0)))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--snrs', nargs='+', type=float, default=[12, 14, 16])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--bits-per-seed', type=int, default=10_000_000)
    ap.add_argument('--block-bits', type=int, default=100_000)
    ap.add_argument('--train-samples', type=int, default=120_000)
    ap.add_argument('--epochs', type=int, default=25)
    a = ap.parse_args(argv)
    if not a.output.name.startswith('codex_'):
        ap.error('output directory must have a codex_ prefix')
    if (a.block_bits <= 0 or a.bits_per_seed <= 0 or a.bits_per_seed % a.block_bits
            or a.train_samples < 3 or a.train_samples % 3 or a.epochs <= 0
            or any(s < 0 for s in a.seeds) or len(set(a.seeds)) != len(a.seeds)
            or not np.all(np.isfinite(a.snrs)) or len(set(a.snrs)) != len(a.snrs)):
        ap.error('use positive, divisible budgets, unique nonnegative seeds and finite unique SNRs')
    a.output.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    plan = {**vars(a), 'output': str(a.output), 'git_commit': git.stdout.strip(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'taps': TAPS, 'train_snrs': TRAIN_SNRS, 'window': 11, 'hidden': 13,
            'normalization': 'per independent block; ISI and window padding reset each block',
            'stopping_rule': 'fixed bits per seed per SNR; no early stopping',
            'status': 'planned; inspect summary.json for completion'}
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    results = {}
    with (a.output / 'counts.jsonl').open('x', encoding='utf-8') as log:
        for seed in a.seeds:
            relay = train(seed, a.train_samples, a.epochs)
            np.savez(a.output / f'codex_seed_{seed}.npz', W1=relay.W1, b1=relay.b1, W2=relay.W2, b2=relay.b2)
            results[str(seed)] = {}
            for si, snr in enumerate(a.snrs):
                counts = {k: [] for k in ('af', 'df', 'mlp')}
                for block in range(a.bits_per_seed // a.block_bits):
                    got = evaluate_block(relay, seed, si, snr, block, a.block_bits)
                    log.write(json.dumps({'seed': seed, 'snr_db': snr, 'block': block,
                                          'bits': a.block_bits, 'errors': got}) + '\n')
                    log.flush()
                    for k, errors in got.items():
                        counts[k].append(errors)
                cell = {k: summarize_blocks(v, a.block_bits) for k, v in counts.items()}
                results[str(seed)][str(snr)] = cell
                print(f'seed={seed} snr={snr}: ' + str({k: v['errors'] for k, v in cell.items()}), flush=True)
    aggregate = {str(snr): {k: summarize_seed_means([
        results[str(seed)][str(snr)][k]['ber'] for seed in a.seeds])
        for k in ('af', 'df', 'mlp')} for snr in a.snrs}
    summary = {'status': 'complete', 'per_seed': results, 'across_seeds': aggregate,
               'historical_results_replaced': False,
               'warning': 'Low/zero counts cannot resolve a precise rare-event BER; no iid-bit CI is claimed.'}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
