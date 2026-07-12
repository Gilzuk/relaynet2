#!/usr/bin/env python3
"""QPSK BER vs number of ISI taps, under symmetric ISI+Rayleigh+AWGN hops.

Extends the symmetric-hop relay comparison (e6_relay_comparison_symmetric.py)
along a new axis: channel memory length L. Longer channel impulse responses
mean more inter-symbol interference to resolve, and for ViterbiMLSEQPSKRelay
specifically, an exponentially growing trellis (num_states = 4^(L-1)).

Taps use a fixed geometric decay profile h_k = 0.7^k for k=0..L-1 (same
shape as the rest of the E6 port's 3-tap default [1.0, 0.7, ~0.5], extended
consistently to more taps), normalized to unit energy as usual. Both hops
use the identical L-tap profile (independent fading/noise realizations per
hop), continuing the symmetric-hop methodology.

Scale reduced from the project default (5 trials) to 3 trials to keep the
L=5 case (256-state trellis) tractable -- Viterbi decode cost scales ~4x
per additional tap (benchmarked: L=3 ~1.8s, L=4 ~6.1s, L=5 ~23.6s per
50k-symbol block). L=6+ not attempted here (would be ~100s/block).
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ComplexISIRayleighChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay

SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 3, 50_000
TAP_LENGTHS = [3, 4, 5]
MODULATION = 'qpsk'


def taps_for(L):
    return np.array([0.7 ** k for k in range(L)])


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0]


def run_for_taplen(L):
    taps = taps_for(L)
    channel_h1 = ComplexISIRayleighChannel(taps, seed=1)
    channel_h2 = ComplexISIRayleighChannel(taps, seed=2)

    relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay(MODULATION),
        'DF-Soft': DFSoftRelay(),
        'Viterbi-Genie': ViterbiMLSEQPSKRelay(channel_taps=taps),
    }

    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in relays}
    print(f"\nL={L} taps (states={relays['Viterbi-Genie'].num_states}) -- SNR (dB): "
          + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for name, relay in relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber

    summary = {}
    for name in relays:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>13}: " + " ".join(f"{m:7.4f}" for m in mu))
    return summary


def main():
    print("=" * 80)
    print("QPSK BER vs ISI TAP COUNT — symmetric ISI+Rayleigh+AWGN hops")
    print("=" * 80)

    all_summaries = {L: run_for_taplen(L) for L in TAP_LENGTHS}

    print("\n" + "=" * 80)
    print("Viterbi-Genie floor @ 20dB SNR, by tap count:")
    for L in TAP_LENGTHS:
        mu, ci = all_summaries[L]['Viterbi-Genie']
        print(f"  L={L}: {mu[-1]:.4f} +/- {ci[-1]:.4f}")

    plot_results(all_summaries)
    return all_summaries


def plot_results(all_summaries):
    fig, axes = plt.subplots(1, len(TAP_LENGTHS) + 1, figsize=(6 * (len(TAP_LENGTHS) + 1), 6))

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728', 'Viterbi-Genie': '#8c564b'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'Viterbi-Genie': 'p'}

    for ax, L in zip(axes[:len(TAP_LENGTHS)], TAP_LENGTHS):
        summary = all_summaries[L]
        for name, (mu, ci) in summary.items():
            ax.semilogy(SNRS, mu, color=color_map[name], marker=marker_map[name],
                        label=name, linewidth=2.5, markersize=7)
            ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])
        ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title(f'L={L} taps', fontsize=12, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_ylim([1e-3, 0.6])
        ax.legend(loc='upper right', fontsize=9)

    axes[0].set_ylabel('BER', fontsize=12, fontweight='bold')

    # Summary panel: Viterbi-Genie only, across all tap lengths
    ax = axes[-1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(TAP_LENGTHS)))
    for L, c in zip(TAP_LENGTHS, cmap):
        mu, ci = all_summaries[L]['Viterbi-Genie']
        ax.semilogy(SNRS, mu, color=c, marker='p', label=f'L={L}', linewidth=2.5, markersize=7)
        ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=c)
    ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Viterbi-Genie only, by tap count', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-3, 0.6])
    ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('QPSK BER vs SNR, increasing ISI channel memory (symmetric Rayleigh+ISI hops)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/e6_viterbi_qpsk_tap_sweep.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_viterbi_qpsk_tap_sweep.png")

    np.save('/tmp/e6_viterbi_qpsk_tap_sweep_results.npy',
            {'snrs': SNRS, 'tap_lengths': TAP_LENGTHS, 'summaries': all_summaries}, allow_pickle=True)
    print("  Saved: /tmp/e6_viterbi_qpsk_tap_sweep_results.npy")


if __name__ == '__main__':
    main()
