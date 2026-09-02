#!/usr/bin/env python3
"""Regenerate thesis figures for Chapter 7 (unknown-channel experiments)
from the N_TRAIN=3 pooled result files in e6_multi_training_results/.

Outputs (matching the filenames referenced in ch07_unknown_and_mismatch_channels.tex):
  results/e6_unknown_channel.png   — S1 ISI+AWGN BER with Viterbi benchmarks
  results/e6_composite.png         — composite channel relay comparison
  results/e6_blind.png             — blind/posterior-free relay comparison
  results/e6_partial_pilot_budget_sweep.png — panel (a): pilot sweep
  results/e6_partial_short_blocks_overhead.png — panel (b): block-length sweep
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'e6_multi_training_results')

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'figure.dpi': 150,
})

BER_FLOOR = 5e-5   # display floor for semilogy


def _semilogy_ber(ax, x, mu, ci, label, color, marker, ls='-', alpha_fill=0.15):
    mu = np.maximum(mu, BER_FLOOR)
    lo = np.maximum(mu - ci, BER_FLOOR)
    hi = mu + ci
    ax.semilogy(x, mu, ls=ls, marker=marker, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_fill)


# ── 1. e6_unknown_channel.png  (S1: ISI→AWGN) ─────────────────────────────
def plot_unknown_channel():
    d = np.load(os.path.join(DATA_DIR, 'e6_sim_ported_results.npy'),
                allow_pickle=True).item()
    snrs = d['snrs']
    s1 = d['results']['S1: unknown ISI -> AWGN']
    n_train = d['n_train']
    n_trials = d['n_trials']
    total = n_train * n_trials

    fig, ax = plt.subplots(figsize=(6, 4))

    palette = {'AF': '#1f77b4', 'DF': '#ff7f0e', 'MLP': '#2ca02c',
               'Vit-genie': '#d62728', 'Vit-LS': '#9467bd'}

    for relay, col, mk in [('AF', palette['AF'], 'o'),
                             ('DF', palette['DF'], 's'),
                             ('MLP', palette['MLP'], '^')]:
        mu, ci = s1[relay]
        _semilogy_ber(ax, snrs, mu, ci,
                      label=f'{relay}' if relay != 'MLP' else 'MLP-170',
                      color=col, marker=mk)

    # Viterbi benchmarks (from the original verified run — fixed, no training seed)
    vit_genie = np.array([0.2467, 0.1822, 0.1233, 0.0741, 0.0375,
                           0.0149, 0.0044, 0.0009, 0.0001, 0.0000, 0.0000])
    vit_ls    = np.array([0.2462, 0.1823, 0.1237, 0.0743, 0.0377,
                           0.0152, 0.0046, 0.0009, 0.0001, 0.0000, 0.0000])
    ax.semilogy(snrs, np.maximum(vit_genie, BER_FLOOR), '--', marker='D',
                color=palette['Vit-genie'], label='Viterbi (genie CSI)')
    ax.semilogy(snrs, np.maximum(vit_ls, BER_FLOOR),    ':', marker='v',
                color=palette['Vit-LS'],   label='Viterbi (200-pilot LS)')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'Unknown ISI → AWGN  '
                 f'(mean ± 95% CI, {n_train} seeds × {n_trials} trials)')
    ax.legend(loc='lower left')
    ax.set_ylim(BER_FLOOR * 0.5, 1.0)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'e6_unknown_channel.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── 2. e6_composite.png ────────────────────────────────────────────────────
def plot_composite():
    d = np.load(os.path.join(DATA_DIR, 'e6_composite_ported_results.npy'),
                allow_pickle=True).item()
    snrs = d['snrs']
    summary = d['summary']
    n_train = d['n_train']
    n_trials = d['n_trials']

    colors = {'AF': '#1f77b4', 'DF-diff': '#ff7f0e',
              'Viterbi-diff': '#d62728',
              'MLP-169': '#2ca02c', 'MLP-large': '#9467bd'}
    markers = {'AF': 'o', 'DF-diff': 's', 'Viterbi-diff': 'D',
               'MLP-169': '^', 'MLP-large': 'v'}
    styles  = {'AF': '-', 'DF-diff': '-', 'Viterbi-diff': '--',
               'MLP-169': '-', 'MLP-large': ':'}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ('AF', 'DF-diff', 'Viterbi-diff', 'MLP-169', 'MLP-large'):
        mu, ci = summary[name]
        _semilogy_ber(ax, snrs, mu, ci, label=name,
                      color=colors[name], marker=markers[name],
                      ls=styles[name])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'Composite channel (ISI × PA × unknown phase)  '
                 f'({n_train} seeds × {n_trials} trials)')
    ax.legend(loc='upper right')
    ax.set_ylim(BER_FLOOR * 0.5, 1.0)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'e6_composite.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── 3. e6_blind.png ────────────────────────────────────────────────────────
def plot_blind():
    d = np.load(os.path.join(DATA_DIR, 'e6_blind_ported_results.npy'),
                allow_pickle=True).item()
    snrs = d['snrs']
    summary = d['summary']
    n_train = d['n_train']
    n_trials = d['n_trials']

    colors  = {'DF-diff': '#ff7f0e', 'CMA-blind': '#1f77b4',
               'Viterbi-blind': '#d62728', 'MLP-169': '#2ca02c'}
    markers = {'DF-diff': 's', 'CMA-blind': 'o',
               'Viterbi-blind': 'D', 'MLP-169': '^'}
    styles  = {'DF-diff': '-', 'CMA-blind': '-',
               'Viterbi-blind': '--', 'MLP-169': '-'}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ('DF-diff', 'CMA-blind', 'Viterbi-blind', 'MLP-169'):
        mu, ci = summary[name]
        _semilogy_ber(ax, snrs, mu, ci, label=name,
                      color=colors[name], marker=markers[name],
                      ls=styles[name])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'Blind (posterior-free) regime  '
                 f'({n_train} seeds × {n_trials} trials per seed)')
    ax.legend(loc='upper right')
    ax.set_ylim(BER_FLOOR * 0.5, 1.0)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'e6_blind.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── 4 & 5. e6_partial_*.png ────────────────────────────────────────────────
def plot_partial():
    d = np.load(os.path.join(DATA_DIR, 'e6_partial_ported_results.npy'),
                allow_pickle=True).item()
    n_train = d['n_train']
    n_trials = d['n_trials']

    # panel (a): pilot sweep
    pilots   = d['pilots']
    panel_a  = d['panel_a']
    mlp_ref  = d['mlp_ref']
    cma_ref  = d['cma_ref']

    fig, ax = plt.subplots(figsize=(5.5, 4))
    mus = [panel_a[p][0] for p in pilots]
    cis = [panel_a[p][1] for p in pilots]
    ax.errorbar(pilots, mus, yerr=cis, fmt='o-', color='#d62728',
                capsize=4, label='Viterbi (pilot-aided)')
    ax.axhline(mlp_ref[0], color='#2ca02c', ls='--',
               label=f'MLP-169 (0 pilots)  BER={mlp_ref[0]:.4f}')
    ax.fill_between([pilots[0], pilots[-1]],
                    mlp_ref[0] - mlp_ref[1], mlp_ref[0] + mlp_ref[1],
                    color='#2ca02c', alpha=0.15)
    ax.axhline(cma_ref[0], color='#1f77b4', ls=':',
               label=f'CMA-blind (0 pilots)  BER={cma_ref[0]:.4f}')

    ax.set_xlabel('Number of pilot symbols')
    ax.set_ylabel('Payload BER')
    ax.set_title(f'Pilot-count sweep at 10 dB  '
                 f'({n_train} seeds × {n_trials} trials)')
    ax.set_xscale('log')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'e6_partial_pilot_budget_sweep.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')

    # panel (b): block-length sweep
    block_lengths = d['block_lengths']
    panel_b       = d['panel_b']
    panel_b_cma   = d['panel_b_cma']

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bls   = block_lengths
    vit_mu = [panel_b[l][0] for l in bls]
    vit_ci = [panel_b[l][1] for l in bls]
    vit_oh = [panel_b[l][2] * 100 for l in bls]
    cma_mu = [panel_b_cma[l][0] for l in bls]
    cma_ci = [panel_b_cma[l][1] for l in bls]

    ax.errorbar(bls, vit_mu, yerr=vit_ci, fmt='o-', color='#d62728',
                capsize=4, label='Viterbi (10 pilots)')
    ax.errorbar(bls, cma_mu, yerr=cma_ci, fmt='s:', color='#1f77b4',
                capsize=4, label='CMA-blind (0 pilots)')
    ax.axhline(mlp_ref[0], color='#2ca02c', ls='--',
               label=f'MLP-169 (0 pilots, 0% overhead)')

    # annotate overhead on top axis
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(bls)
    ax2.set_xticklabels([f'{o:.0f}%' for o in vit_oh], fontsize=8)
    ax2.set_xlabel('Pilot overhead (10 pilots / L)', fontsize=9)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel('Block length L (symbols)')
    ax.set_ylabel('Payload BER')
    ax.set_title(f'Block-length sweep at 10 dB, 10 pilots fixed  '
                 f'({n_train} seeds × {n_trials} trials)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'e6_partial_short_blocks_overhead.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


if __name__ == '__main__':
    print('Regenerating thesis figures from N_TRAIN=3 results...')
    plot_unknown_channel()
    plot_composite()
    plot_blind()
    plot_partial()
    print('Done.')
