#!/usr/bin/env python3
"""Plot 1-seed vs 3-seed comparison for E6_SIM (S1: unknown ISI → AWGN).

Shows that training on 3 independent seeds produces tighter confidence intervals
and more stable mean BER, addressing the reviewer concern about single-seed results.

Outputs:
  e6_multi_training_results/e6_seed_comparison.png
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'e6_multi_training_results')
sys.path.insert(0, ROOT)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'figure.dpi': 150,
})

BER_FLOOR = 5e-5


def _load_e6sim():
    """Load e6_sim_ported as a module without running __main__."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_e6sim_cmp", os.path.join(ROOT, "e6_sim_ported.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = "_e6sim_cmp"   # avoids __main__ guard
    spec.loader.exec_module(mod)
    return mod


def run_one_seed(mod, seed):
    """Train 1 MLP with given seed, evaluate on S1: ISI→AWGN, return (snrs, mu, ci)."""
    np.random.seed(seed % (2 ** 31))
    hop1_ch = mod.create_channel('isi', seed=1)
    mlp, _ = mod.train_mlp(hop1_ch, seed=seed)

    # Patch N_TRAIN=1 so run_experiment pools only N_TRIALS columns
    orig = mod.N_TRAIN
    mod.N_TRAIN = 1
    res = mod.run_experiment('isi', 'awgn', [mlp])
    mod.N_TRAIN = orig

    mu, ci = res['MLP']
    return mod.SNRS, mu, ci


def _semilogy_band(ax, x, mu, ci, label, color, marker, ls='-', alpha=0.18):
    mu2 = np.maximum(mu, BER_FLOOR)
    lo  = np.maximum(mu - ci, BER_FLOOR)
    hi  = mu + ci
    ax.semilogy(x, mu2, ls=ls, marker=marker, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)


def main():
    # ── 3-seed summary from saved file ───────────────────────────────────
    d3       = np.load(os.path.join(DATA_DIR, 'e6_sim_ported_results.npy'),
                       allow_pickle=True).item()
    snrs     = d3['snrs']
    n_train  = d3['n_train']
    n_trials = d3['n_trials']
    mu3, ci3 = d3['results']['S1: unknown ISI -> AWGN']['MLP']

    # ── 1-seed live run ───────────────────────────────────────────────────
    print('Training single-seed MLP (seed=42) on S1 for comparison…', flush=True)
    mod = _load_e6sim()
    snrs1, mu1, ci1 = run_one_seed(mod, seed=42)
    print('  Done.')

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    _semilogy_band(ax, snrs,  mu3, ci3,
                   label=f'MLP — 3 training seeds × {n_trials} trials  (CI ∝ 1/√30)',
                   color='#2ca02c', marker='^', ls='-')

    _semilogy_band(ax, snrs1, mu1, ci1,
                   label=f'MLP — 1 training seed × {n_trials} trials  (CI ∝ 1/√10)',
                   color='#d62728', marker='o', ls='--', alpha=0.12)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('Seed robustness: 1 vs 3 independent training seeds\n'
                 'S1 — Unknown ISI relay → AWGN channel, MLP-170')
    ax.set_ylim(BER_FLOOR * 0.5, 1.0)
    ax.legend(loc='lower left')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()

    out = os.path.join(DATA_DIR, 'e6_seed_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ── summary table ─────────────────────────────────────────────────────
    print('\n   SNR  | 1-seed BER ± CI      | 3-seed BER ± CI')
    print('  ------+---------------------+--------------------')
    for i, snr in enumerate(snrs):
        print(f'  {snr:5.1f} | {mu1[i]:.6f} ± {ci1[i]:.6f}  | '
              f'{mu3[i]:.6f} ± {ci3[i]:.6f}')


if __name__ == '__main__':
    main()

