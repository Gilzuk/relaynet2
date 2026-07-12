#!/usr/bin/env python3
"""Relay architecture comparison under SYMMETRIC hop impairments.

Every prior E6 comparison in this repo made hop 2 easier than hop 1 (clean
AWGN, or Rayleigh-only with no ISI) -- so a relay that "fixed" hop 1 got a
free ride on hop 2. That conflates relay quality with channel asymmetry.

Here BOTH hops use the identical channel model -- unknown 3-tap ISI +
coherently-compensated Rayleigh fading + AWGN (`ISIRayleighChannel` /
`ComplexISIRayleighChannel`, relaynet/channels/e6_channels.py) -- each hop
drawing its own independent fading/ISI-convolution/noise realization, but
from the same statistical model, agnostic to which side is transmitting.
This isolates relay-architecture differences: since the destination does
plain hard-decision demodulation (no hop-2 equalization), even a "perfect"
relay forwards a signal that hop 2 will re-corrupt with its own unequalized
ISI + fading -- so there is an unavoidable shared floor, and what we're
really comparing is how gracefully each relay's output degrades under a
second dose of the same impairment.

Note on Viterbi-Genie: "genie" here means perfect knowledge of the (static)
ISI taps only, exactly as in every other E6 Viterbi comparison in this repo.
The per-symbol Rayleigh fading is NOT part of the genie CSI and is not
explicitly corrected by the trellis -- it's an additional impairment on top
of what Viterbi models, deliberately, to see how a relay built for "ISI
only" degrades once fading is added on both hops.
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, MLPRelay, ViterbiMLSERelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ISIRayleighChannel, ComplexISIRayleighChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay

W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 10, 100_000  # project-standard scale for thesis integration

rng = np.random.default_rng(42)
H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)


def train_mlp170(channel, seed=0, n_train=120_000):
    mlp = MLPRelay(input_size=W, hidden_size=13, output_size=1, window_size=W, seed=seed)
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        bits = rng.integers(0, 2, per_snr)
        x = 1.0 - 2.0 * bits
        y = channel(x, snr_db)
        pad = W // 2
        yp = np.pad(y, (pad, pad), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(yp, W)
        X_list.append(windows)
        T_list.append(x)
    X = np.vstack(X_list)
    T = np.concatenate(T_list)
    mlp.train_on_data(X, T, epochs=25, batch_size=256, lr=3e-3)
    return mlp


def run_ber_trial(relay, channel_h1, channel_h2, modulation, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=modulation)
    dest = Destination(modulation=modulation)
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0]


def run_suite(modulation, relays, channel_h1, channel_h2):
    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in relays}
    print(f"\n{modulation.upper()} -- SNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for name, relay in relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, modulation, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber
    summary = {}
    for name in relays:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>15}: " + " ".join(f"{m:7.4f}" for m in mu))
    return summary


def main():
    print("=" * 80)
    print("RELAY COMPARISON — SYMMETRIC HOPS (unknown ISI + Rayleigh + AWGN, both hops)")
    print("=" * 80)

    # BPSK
    bpsk_h1 = ISIRayleighChannel(H_ISI, seed=1)
    bpsk_h2 = ISIRayleighChannel(H_ISI, seed=2)

    print("\nTraining MLP-170 (BPSK) on the symmetric-hop channel...")
    mlp170 = train_mlp170(ISIRayleighChannel(H_ISI, seed=3), seed=0)

    bpsk_relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay('bpsk'),
        'DF-Soft': DFSoftRelay(),
        'MLP-170': mlp170,
        'Viterbi-Genie': ViterbiMLSERelay(channel_taps=H_ISI),
    }
    bpsk_summary = run_suite('bpsk', bpsk_relays, bpsk_h1, bpsk_h2)

    # QPSK
    qpsk_h1 = ComplexISIRayleighChannel(H_ISI, seed=1)
    qpsk_h2 = ComplexISIRayleighChannel(H_ISI, seed=2)

    qpsk_relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay('qpsk'),
        'DF-Soft': DFSoftRelay(),
        'Viterbi-Genie': ViterbiMLSEQPSKRelay(channel_taps=H_ISI),
    }
    qpsk_summary = run_suite('qpsk', qpsk_relays, qpsk_h1, qpsk_h2)

    plot_results(bpsk_summary, qpsk_summary)
    return bpsk_summary, qpsk_summary


def plot_results(bpsk_summary, qpsk_summary):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728',
                 'MLP-170': '#2ca02c', 'Viterbi-Genie': '#8c564b'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'MLP-170': 'd', 'Viterbi-Genie': 'p'}

    for ax, (title, summary) in zip(axes, [('BPSK', bpsk_summary), ('QPSK', qpsk_summary)]):
        for name, (mu, ci) in summary.items():
            ax.semilogy(SNRS, mu, color=color_map[name], marker=marker_map[name],
                        label=name, linewidth=2.5, markersize=8)
            ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} — Symmetric Hops (ISI + Rayleigh + AWGN, both hops)',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_ylim([1e-4, 0.6])

    axes[0].set_ylabel('BER', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/tmp/e6_relay_comparison_symmetric.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_relay_comparison_symmetric.png")

    np.save('/tmp/e6_relay_comparison_symmetric_results.npy',
            {'snrs': SNRS, 'bpsk': bpsk_summary, 'qpsk': qpsk_summary}, allow_pickle=True)
    print("  Saved: /tmp/e6_relay_comparison_symmetric_results.npy")


if __name__ == '__main__':
    main()
