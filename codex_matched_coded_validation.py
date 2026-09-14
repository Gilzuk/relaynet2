"""Paired matched/mismatched QPSK coded-relay ablation with fresh artifacts.

Changes relay and destination metrics separately. Destination weighting is
matched to a QPSK AWGN hop, NOT to the full distribution induced by an erroneous
or soft relay. No end-to-end optimality is claimed. Defaults use the historical
MLP training budget; choose smaller explicit budgets only for pilot validation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.stats import t

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.coding.codex_matched import CodexMatchedViterbiDecoder, qpsk_soft_observations
from relaynet.relays.codex_coded_df import CodexMatchedCodedRelay
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.modulation.qpsk import qpsk_modulate
from coded_learned_relay import generate_coded_training_data


def interval(values):
    a = np.asarray(values, dtype=float)
    half = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
    # Identical observations, especially all-zero errors, don't establish certainty.
    if half == 0:
        half = None
    return {'mean': float(a.mean()), 't_halfwidth95_approx': half, 'replicates': len(a)}


def channel_draw(rng, size):
    h = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    z = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return h, z


def transmit(tx, h, z, snr):
    n0 = float(np.mean(np.abs(tx) ** 2)) / 10 ** (snr / 10)
    if n0 <= 0:
        raise ValueError('zero-power relay output cannot define the specified SNR')
    return (h * tx + np.sqrt(n0) * z) / h, n0 / (2 * np.abs(h) ** 2)


def run_trial(relays, snr, seed, frames, frame_bits=200):
    rng = np.random.default_rng(seed)
    encoder = ConvolutionalEncoder()
    info = rng.integers(0, 2, (frames, frame_bits))
    tx = np.concatenate([qpsk_modulate(encoder.encode(b)) for b in info])
    h1, z1 = channel_draw(rng, len(tx))
    h2, z2 = channel_draw(rng, len(tx))
    y1, var1 = transmit(tx, h1, z1, snr)
    fs = frame_bits + encoder.num_tail
    old, new = ViterbiCodeDecoder(), CodexMatchedViterbiDecoder()
    records = {}
    for name, relay in relays.items():
        if isinstance(relay, CodexMatchedCodedRelay):
            out = relay.process(y1, axis_noise_var=var1)
        else:
            if isinstance(relay, SoftCodedDecodeAndForwardRelay):
                relay.set_snr_db(snr)
            out = relay.process(y1)
        y2, var2 = transmit(out, h2, z2, snr)
        obs, variance = qpsk_soft_observations(y2, var2)
        for destination in ('legacy', 'weighted'):
            errors = []
            for f in range(frames):
                sl = slice(2 * f * fs, 2 * (f + 1) * fs)
                decoded = (old.decode(obs[sl]) if destination == 'legacy' else
                           new.decode(obs[sl], variance[sl], symbol_amplitude=1 / np.sqrt(2)))
                errors.append(int(np.count_nonzero(decoded != info[f])))
            records[name + '/' + destination] = {
                'frame_errors': errors, 'errors': sum(errors), 'bits': frames * frame_bits,
                'ber': sum(errors) / (frames * frame_bits),
                'fer': sum(e > 0 for e in errors) / frames}
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--snrs', type=float, nargs='+', default=[16, 20])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--train-frames', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=25)
    a = ap.parse_args(argv)
    if (not a.output.name.startswith('codex_') or min(a.trials, a.frames, a.train_frames, a.epochs) < 1
            or any(s < 0 for s in a.seeds) or len(set(a.seeds)) != len(a.seeds)
            or not np.all(np.isfinite(a.snrs)) or len(set(a.snrs)) != len(a.snrs)):
        ap.error('use codex_ output, positive budgets, unique nonnegative seeds and finite unique SNRs')
    a.output.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    plan = {**vars(a), 'output': str(a.output), 'git_commit': git.stdout.strip(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'frame_info_bits': 200,
            'constraint_length': 3, 'train_snrs': [5, 10, 15], 'window': 21, 'hidden': 16,
            'snr_convention': 'Es/N0 per hop, unit-power QPSK, actual transmitted power used',
            'paired': 'same information bits and both-hop fading/noise for every method within a trial',
            'uncertainty_unit': 'trial at fixed checkpoint; then independent training seeds',
            'scope': 'weighted destination metrics model a QPSK hop, not the entire relay-induced channel'}
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    summaries = {}
    with (a.output / 'counts.jsonl').open('x', encoding='utf-8') as log:
        for seed in a.seeds:
            encoder = ConvolutionalEncoder()
            mlp = MLPQPSKClassifierRelay(window_size=21, hidden_size=16, seed=seed)
            x, targets = generate_coded_training_data(encoder, mlp, a.train_frames, seed=seed)
            mlp.train_on_data(x, targets, epochs=a.epochs, batch_size=512, lr=3e-3)
            np.savez(a.output / f'codex_mlp_seed_{seed}.npz', W1=mlp.W1, b1=mlp.b1, W2=mlp.W2, b2=mlp.b2)
            relays = {'legacy_df': CodedDecodeAndForwardRelay(frame_info_bits=200),
                      'weighted_df': CodexMatchedCodedRelay(),
                      'legacy_soft': SoftCodedDecodeAndForwardRelay(),
                      'weighted_soft': CodexMatchedCodedRelay(soft=True), 'mlp': mlp}
            summaries[str(seed)] = {}
            for si, snr in enumerate(a.snrs):
                cells = []
                for trial in range(a.trials):
                    ss = np.random.SeedSequence([20260914, 2, seed, si, trial])
                    got = run_trial(relays, snr, ss, a.frames)
                    log.write(json.dumps({'seed': seed, 'snr_db': snr, 'trial': trial, 'results': got}) + '\n')
                    log.flush()
                    cells.append(got)
                metrics = {k: interval([c[k]['ber'] for c in cells]) for k in cells[0]}
                # Preserve pairing when estimating the uncertainty of differences.
                pairs = [('weighted_df/weighted', 'legacy_df/legacy'),
                         ('mlp/weighted', 'weighted_df/weighted'),
                         ('weighted_soft/weighted', 'weighted_df/weighted')]
                differences = {left + ' minus ' + right: interval([
                    c[left]['ber'] - c[right]['ber'] for c in cells]) for left, right in pairs}
                summaries[str(seed)][str(snr)] = {'ber': metrics, 'paired_differences': differences}
                print(f'seed={seed} snr={snr}: ' + str({k: v['mean'] for k, v in metrics.items()}), flush=True)
    across = {str(snr): {k: interval([summaries[str(seed)][str(snr)]['ber'][k]['mean']
                                    for seed in a.seeds])
                        for k in summaries[str(a.seeds[0])][str(snr)]['ber']} for snr in a.snrs}
    (a.output / 'summary.json').write_text(json.dumps(
        {'status': 'complete', 'per_seed': summaries, 'across_training_seeds': across,
         'historical_results_replaced': False, 'comparisons': 'exploratory; no equivalence claim'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
