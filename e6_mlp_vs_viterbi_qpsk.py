#!/usr/bin/env python3
"""Fair comparison: MLP-170 (BPSK) vs Viterbi-Genie (BPSK) vs Viterbi-Genie (QPSK).

Earlier ad-hoc numbers were NOT comparable: MLP-170's BER came from a run
using RayleighChannel for hop 2 (e6_sim_enhanced.py), while Viterbi-Genie
(QPSK)'s came from a run using plain ComplexAWGNChannel for hop 2
(e6_viterbi_qpsk.py) -- Rayleigh fading alone caps high-SNR BER around
~0.005 regardless of relay, which would make any "QPSK wins" conclusion
spurious. This script re-runs all three under the IDENTICAL scenario
(unknown 3-tap ISI hop 1 -> plain AWGN hop 2), isolating the actual
variables of interest: relay architecture and modulation order.

Also included as a sanity check: for coherent Gray-coded detection over
AWGN, BER-vs-SNR_dB should be provably invariant between BPSK and QPSK
(splitting into I/Q halves both signal energy and noise variance equally).
Any large QPSK-vs-BPSK gap that shows up here for Viterbi-Genie is real
and needs an explanation, not a modulation-order artifact.
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import MLPRelay, ViterbiMLSERelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ISIChannel, ComplexISIChannel, ComplexAWGNChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination

W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000

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


def main():
    print("=" * 80)
    print("FAIR COMPARISON: MLP-170 (BPSK) vs Viterbi-Genie (BPSK) vs Viterbi-Genie (QPSK)")
    print("Identical scenario for all three: unknown 3-tap ISI -> plain AWGN")
    print("=" * 80)

    bpsk_h1 = ISIChannel(H_ISI, seed=1)
    bpsk_h2 = ComplexAWGNChannel(seed=2)
    qpsk_h1 = ComplexISIChannel(H_ISI, seed=1)
    qpsk_h2 = ComplexAWGNChannel(seed=2)

    print("\nTraining MLP-170 (BPSK)...")
    mlp170 = train_mlp170(bpsk_h1, seed=0)

    relays = {
        'MLP-170 (BPSK)': ('bpsk', mlp170, bpsk_h1, bpsk_h2),
        'Viterbi-Genie (BPSK)': ('bpsk', ViterbiMLSERelay(channel_taps=H_ISI), bpsk_h1, bpsk_h2),
        'Viterbi-Genie (QPSK)': ('qpsk', ViterbiMLSEQPSKRelay(channel_taps=H_ISI), qpsk_h1, qpsk_h2),
    }

    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in relays}

    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for name, (modulation, relay, ch1, ch2) in relays.items():
                ber = run_ber_trial(relay, ch1, ch2, modulation, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber

    summary = {}
    for name in relays:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>22}: " + " ".join(f"{m:7.4f}" for m in mu))

    plot_results(summary)
    return summary


def plot_results(summary):
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {
        'MLP-170 (BPSK)': '#2ca02c',
        'Viterbi-Genie (BPSK)': '#8c564b',
        'Viterbi-Genie (QPSK)': '#9467bd',
    }
    marker_map = {
        'MLP-170 (BPSK)': 'd',
        'Viterbi-Genie (BPSK)': 'p',
        'Viterbi-Genie (QPSK)': 'X',
    }

    for name, (mu, ci) in summary.items():
        ax.semilogy(SNRS, mu, color=color_map[name], marker=marker_map[name],
                    label=name, linewidth=2.5, markersize=8)
        ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title('MLP-170 vs Viterbi-Genie, BPSK vs QPSK — Unknown ISI -> AWGN (matched hop 2)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-5, 0.6])

    plt.tight_layout()
    plt.savefig('/tmp/e6_mlp_vs_viterbi_qpsk_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_mlp_vs_viterbi_qpsk_comparison.png")
    np.save('/tmp/e6_mlp_vs_viterbi_qpsk_results.npy', {'snrs': SNRS, 'summary': summary}, allow_pickle=True)
    print("  Saved: /tmp/e6_mlp_vs_viterbi_qpsk_results.npy")


if __name__ == '__main__':
    main()
