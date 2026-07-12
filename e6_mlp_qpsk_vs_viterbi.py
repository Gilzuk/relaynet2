#!/usr/bin/env python3
"""QPSK relay comparison including a proper 4-class MLP classifier, L=3 taps.

Adds MLPQPSKClassifierRelay (relaynet/relays/mlp.py) to the QPSK relay
suite -- the earlier BPSK-only MLPRelay used a single tanh regression
output, which is not a valid target for QPSK's 4-symbol alphabet. This
classifier instead predicts one of the 4 Gray-coded constellation points
per windowed observation via softmax + cross-entropy, matching
ViterbiMLSEQPSKRelay's alphabet index-for-index so outputs are directly
comparable.

Scenario: symmetric hops (identical ISIRayleighChannel/ComplexISIRayleighChannel
model on both hops, unknown ISI + Rayleigh + AWGN), L=3 taps only for now
(geometric decay h_k = 0.7^k, same profile as the L-sweep experiment).

Also reports wall-clock latency (MLP-QPSK forward pass vs Viterbi-Genie
decode) on a matched block size, extending the earlier per-tap latency
benchmark to include the classifier.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, MLPQPSKClassifierRelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ComplexISIRayleighChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay

W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000
MODULATION = 'qpsk'
L = 3

rng = np.random.default_rng(42)


def taps_for(L):
    return np.array([0.7 ** k for k in range(L)])


def train_mlp_qpsk(channel, seed=0, n_train=150_000, hidden_size=7):
    mlp = MLPQPSKClassifierRelay(window_size=W, hidden_size=hidden_size, seed=seed)
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        n_bits = (per_snr // 2) * 2
        bits = rng.integers(0, 2, n_bits)
        idx = bits.reshape(-1, 2)[:, 0] * 2 + bits.reshape(-1, 2)[:, 1]  # 0..3 class index
        I = 1.0 - 2.0 * bits.reshape(-1, 2)[:, 0]
        Q = 1.0 - 2.0 * bits.reshape(-1, 2)[:, 1]
        x = (I + 1j * Q) / np.sqrt(2)
        y = channel(x, snr_db)
        windows = mlp._extract_windows(y)
        X_list.append(windows)
        T_list.append(idx)
    X = np.vstack(X_list)
    T = np.concatenate(T_list)
    mlp.train_on_data(X, T, epochs=25, batch_size=256, lr=3e-3)
    return mlp


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0]


def measure_latency(relay, block_size=50_000, repeats=3):
    rng_l = np.random.default_rng(0)
    y = (rng_l.standard_normal(block_size) + 1j * rng_l.standard_normal(block_size)) / np.sqrt(2)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        relay.process(y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    print("=" * 80)
    print(f"QPSK RELAY COMPARISON incl. 4-class MLP classifier — L={L} taps, symmetric hops")
    print("=" * 80)

    taps = taps_for(L)
    channel_h1 = ComplexISIRayleighChannel(taps, seed=1)
    channel_h2 = ComplexISIRayleighChannel(taps, seed=2)

    print("\nTraining MLP-QPSK classifier...")
    mlp_qpsk = train_mlp_qpsk(ComplexISIRayleighChannel(taps, seed=3), seed=0)
    print(f"  MLP-QPSK: {mlp_qpsk.n_params()} params (window={W}, hidden={mlp_qpsk.hidden_size})")

    relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay(MODULATION),
        'DF-Soft': DFSoftRelay(),
        'MLP-QPSK': mlp_qpsk,
        'Viterbi-Genie': ViterbiMLSEQPSKRelay(channel_taps=taps),
    }

    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in relays}
    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
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

    print("\nLatency (median of 3 runs, 50k-symbol block):")
    lat_mlp = measure_latency(mlp_qpsk)
    lat_vit = measure_latency(relays['Viterbi-Genie'])
    print(f"  MLP-QPSK      : {lat_mlp*1000:8.2f} ms  ({lat_mlp/50_000*1e6:.3f} us/symbol)")
    print(f"  Viterbi-Genie : {lat_vit*1000:8.2f} ms  ({lat_vit/50_000*1e6:.3f} us/symbol)")
    print(f"  Viterbi is {lat_vit/lat_mlp:.1f}x slower than MLP-QPSK at L={L}")

    plot_results(summary, lat_mlp, lat_vit)
    return summary, lat_mlp, lat_vit


def plot_results(summary, lat_mlp, lat_vit):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728',
                 'MLP-QPSK': '#2ca02c', 'Viterbi-Genie': '#8c564b'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'MLP-QPSK': 'd', 'Viterbi-Genie': 'p'}

    ax = axes[0]
    for name, (mu, ci) in summary.items():
        ax.semilogy(SNRS, mu, color=color_map[name], marker=marker_map[name],
                    label=name, linewidth=2.5, markersize=8)
        ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title(f'QPSK, L={L} taps, symmetric ISI+Rayleigh hops', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-3, 0.6])

    ax = axes[1]
    names = ['MLP-QPSK', 'Viterbi-Genie']
    times_ms = [lat_mlp * 1000, lat_vit * 1000]
    bars = ax.bar(names, times_ms, color=[color_map['MLP-QPSK'], color_map['Viterbi-Genie']])
    ax.set_ylabel('Latency, ms per 50k-symbol block (log scale)', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title(f'Decode latency, L={L} taps', fontsize=12, fontweight='bold')
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f'{t:.1f} ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('/tmp/e6_mlp_qpsk_vs_viterbi.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_mlp_qpsk_vs_viterbi.png")

    np.save('/tmp/e6_mlp_qpsk_vs_viterbi_results.npy',
            {'snrs': SNRS, 'summary': summary, 'lat_mlp': lat_mlp, 'lat_vit': lat_vit},
            allow_pickle=True)
    print("  Saved: /tmp/e6_mlp_qpsk_vs_viterbi_results.npy")


if __name__ == '__main__':
    main()
