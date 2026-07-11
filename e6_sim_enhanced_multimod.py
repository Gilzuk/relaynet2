#!/usr/bin/env python3
"""E6_SIM Enhanced (multi-modulation): DF hard vs soft decision boundaries.

Extends e6_sim_enhanced.py's classical relay comparison (AF, DF-Hard, DF-Soft)
from BPSK to QPSK and 16-QAM, on the flagship E6_SIM scenario: unknown 3-tap
ISI (hop 1) -> AWGN (hop 2).

DF Decision Types (per modulation):
  - Hard: demodulate to bits (nearest constellation point), then re-modulate
    a clean unit-power symbol -- NOT a bare sign() once modulation order > 2.
  - Soft: forward the received signal with power normalization only, no
    per-symbol quantization.

AI relays (MLP/Viterbi) are intentionally out of scope here: MLPRelay's
single tanh output only regresses one real value per window, which is a
correct target for BPSK but not for QPSK (2 bits/symbol) or 16-QAM
(4 bits/symbol) without a multi-output redesign. See memory-bank/progress.md.
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay
from relaynet.channels import ComplexISIChannel, ISIChannel, ComplexAWGNChannel
from relaynet.modulation import get_modulation_functions, calculate_ber
from relaynet.nodes import Source, Destination

# Configuration
SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 5, 50_000
MODULATIONS = ['bpsk', 'qpsk', 'qam16']

H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)


class DFHardRelay:
    """DF Hard Decision: nearest-constellation-point quantization (modulation-aware)."""

    def __init__(self, modulation):
        self.modulate, self.demodulate, _ = get_modulation_functions(modulation)

    def process(self, received_signal):
        bits = self.demodulate(received_signal)
        return self.modulate(bits)


class DFSoftRelay:
    """DF Soft Decision: power normalization, no quantization."""

    def process(self, received_signal):
        power = np.sqrt(np.mean(np.abs(received_signal) ** 2)) + 1e-12
        return received_signal / power


def create_hop1(modulation, seed):
    """Unknown 3-tap ISI channel, complex-aware for QPSK/16-QAM."""
    if modulation == 'bpsk':
        return ISIChannel(H_ISI, seed=seed)
    return ComplexISIChannel(H_ISI, seed=seed)


def run_ber_trial(relay, channel_h1, channel_h2, modulation, num_bits, snr_db, seed):
    """Run a single BER trial for one modulation/relay/SNR combination."""
    source = Source(seed=seed, modulation=modulation)
    dest = Destination(modulation=modulation)

    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)[0]


def run_modulation(modulation):
    """Run the AF / DF-Hard / DF-Soft comparison for one modulation scheme."""
    channel_h1 = create_hop1(modulation, seed=1)
    # ComplexAWGNChannel handles both real (BPSK) and complex (QPSK/16-QAM) signals,
    # applying real or circularly-symmetric complex noise as appropriate.
    channel_h2 = ComplexAWGNChannel(seed=2)

    relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay(modulation),
        'DF-Soft': DFSoftRelay(),
    }

    results = {r: np.zeros((len(SNRS), N_TRIALS)) for r in relays}

    print(f"\n{modulation.upper()}: unknown ISI -> AWGN")
    print(f"  SNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))

    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for relay_name, relay in relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, modulation, N_BITS, snr, seed=seed_base)
                results[relay_name][si, tr] = ber

    for relay_name in ('AF', 'DF-Hard', 'DF-Soft'):
        mu = results[relay_name].mean(axis=1)
        print(f"  {relay_name:>8}: " + " ".join(f"{m:7.4f}" for m in mu))

    return {r: (v.mean(axis=1), 1.96 * v.std(axis=1) / np.sqrt(N_TRIALS)) for r, v in results.items()}


def plot_results(all_results):
    """Three-panel BER comparison, one panel per modulation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^'}

    for ax, modulation in zip(axes, MODULATIONS):
        results = all_results[modulation]
        for relay_name in ('AF', 'DF-Hard', 'DF-Soft'):
            mu, ci = results[relay_name]
            ax.semilogy(SNRS, mu, color=color_map[relay_name], marker=marker_map[relay_name],
                        label=relay_name, linewidth=2.5, markersize=7)
            ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci,
                             alpha=0.15, color=color_map[relay_name])
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title(modulation.upper(), fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_ylim([1e-4, 0.6])

    axes[0].set_ylabel('BER', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)

    plt.suptitle('DF Hard vs Soft Decision Boundary — Unknown ISI -> AWGN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/e6_sim_enhanced_multimod_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_sim_enhanced_multimod_comparison.png")

    np.save('/tmp/e6_sim_enhanced_multimod_results.npy',
            {'snrs': SNRS, 'results': all_results}, allow_pickle=True)
    print("  Saved: /tmp/e6_sim_enhanced_multimod_results.npy")


def main():
    print("=" * 80)
    print("E6_SIM ENHANCED (multi-modulation): DF Hard vs Soft — BPSK / QPSK / 16-QAM")
    print("=" * 80)

    all_results = {m: run_modulation(m) for m in MODULATIONS}
    plot_results(all_results)
    return all_results


if __name__ == '__main__':
    main()
