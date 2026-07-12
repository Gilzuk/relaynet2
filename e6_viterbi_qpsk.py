#!/usr/bin/env python3
"""E6_VITERBI (QPSK): Viterbi-Genie MLSE vs classical DF hard/soft, unknown ISI -> AWGN.

Extends the E6_VITERBI comparison (previously BPSK-only, see e6_viterbi_ported.py)
to QPSK, using the new ViterbiMLSEQPSKRelay (4-symbol Gray-coded alphabet,
16-state trellis for the 3-tap ISI channel). Shows what the actual optimal
(memory-aware, genie-CSI) decoder achieves versus the memoryless AF/DF-Hard/
DF-Soft relays compared in e6_sim_enhanced_multimod.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ComplexISIChannel, ComplexAWGNChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay

SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 5, 50_000
MODULATION = 'qpsk'

H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)

    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)[0]


def main():
    print("=" * 80)
    print("E6_VITERBI (QPSK): Viterbi-Genie vs AF / DF-Hard / DF-Soft")
    print("=" * 80)

    channel_h1 = ComplexISIChannel(H_ISI, seed=1)
    channel_h2 = ComplexAWGNChannel(seed=2)

    relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay(MODULATION),
        'DF-Soft': DFSoftRelay(),
        'Viterbi-Genie': ViterbiMLSEQPSKRelay(channel_taps=H_ISI),
    }

    results = {r: np.zeros((len(SNRS), N_TRIALS)) for r in relays}

    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for relay_name, relay in relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[relay_name][si, tr] = ber

    summary = {}
    for relay_name in ('AF', 'DF-Hard', 'DF-Soft', 'Viterbi-Genie'):
        mu = results[relay_name].mean(axis=1)
        ci = 1.96 * results[relay_name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[relay_name] = (mu, ci)
        print(f"  {relay_name:>13}: " + " ".join(f"{m:7.4f}" for m in mu))

    # Report Viterbi-Genie's advantage at 1e-2 BER, matching the BPSK E6_VITERBI convention
    vit_mu = summary['Viterbi-Genie'][0]
    below_1e2 = np.where(vit_mu < 1e-2)[0]
    if len(below_1e2):
        print(f"\n  Viterbi-Genie first reaches BER < 1e-2 at SNR = {SNRS[below_1e2[0]]} dB")

    plot_results(summary)
    return summary


def plot_results(summary):
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728', 'Viterbi-Genie': '#8c564b'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'Viterbi-Genie': 'p'}

    for relay_name, (mu, ci) in summary.items():
        ax.semilogy(SNRS, mu, color=color_map[relay_name], marker=marker_map[relay_name],
                    label=relay_name, linewidth=2.5, markersize=8)
        ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[relay_name])

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title('QPSK: Viterbi-Genie MLSE vs Classical DF (Hard/Soft) — Unknown ISI -> AWGN',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-5, 0.6])

    plt.tight_layout()
    plt.savefig('/tmp/e6_viterbi_qpsk_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_viterbi_qpsk_comparison.png")

    np.save('/tmp/e6_viterbi_qpsk_results.npy', {'snrs': SNRS, 'summary': summary}, allow_pickle=True)
    print("  Saved: /tmp/e6_viterbi_qpsk_results.npy")


if __name__ == '__main__':
    main()
