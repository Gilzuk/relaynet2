#!/usr/bin/env python3
"""E6_SIM Enhanced: Multi-Architecture Relay Comparison.

Compares:
  - Classical: AF, DF (hard decision), DF (soft decision)
  - AI-based: MLP-170, MLPLarge, ViterbiMLSE

DF Decision Types:
  - Hard: Nearest constellation point (discrete quantization)
  - Soft: No quantization (continuous signal pass-through)
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import (
    AmplifyAndForwardRelay,
    DecodeAndForwardRelay,
    MLPRelay,
    ViterbiMLSERelay,
)
from relaynet.channels import ISIChannel, RayleighChannel
from relaynet.modulation.bpsk import calculate_ber
from relaynet.nodes import Source, Destination

# Configuration
W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000

rng = np.random.default_rng(42)
H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)


class DFSoftRelay:
    """DF Soft Decision: Pass-through with power normalization (no hard quantization)."""

    def process(self, received_signal):
        """Normalize power without hard quantization."""
        power = np.sqrt(np.mean(received_signal ** 2)) + 1e-12
        return received_signal / power


def create_isi_channel(seed=None):
    """Create ISI channel."""
    return ISIChannel(H_ISI, seed=seed)


def train_mlp(channel, hidden_size=13, seed=0, n_train=120_000):
    """Train MLP relay."""
    mlp = MLPRelay(
        input_size=W,
        hidden_size=hidden_size,
        output_size=1,
        window_size=W,
        seed=seed
    )

    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []

    for snr_db in TRAIN_SNRS:
        bits = rng.integers(0, 2, per_snr)
        x = 1.0 - 2.0 * bits
        y = channel(x, snr_db)
        pad_size = W // 2
        yp = np.pad(y, (pad_size, pad_size), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(yp, W)
        X_list.append(windows)
        T_list.append(x)

    X = np.vstack(X_list)
    T = np.concatenate(T_list)
    mlp.train_on_data(X, T, epochs=25, batch_size=256, lr=3e-3)
    return mlp


def train_viterbi_genie(channel):
    """Create Viterbi MLSE with genie CSI."""
    return ViterbiMLSERelay(channel_taps=H_ISI)


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed=None):
    """Run single BER trial."""
    if seed is None:
        seed = 42

    source = Source(seed=seed, modulation='bpsk')
    dest = Destination(modulation='bpsk')

    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)

    # Relay processing
    relay_out = relay.process(rx_relay)

    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)[0]


def main():
    """Main entry point."""
    print("=" * 80)
    print("E6_SIM ENHANCED: 3 AI Architectures + Classical DF (Hard/Soft)")
    print("=" * 80)

    channel_h1 = create_isi_channel(seed=1)
    channel_h2 = RayleighChannel(seed=2)

    print("\nTraining relays...")

    # Classical relays
    af_relay = AmplifyAndForwardRelay(target_power=1.0)
    df_hard_relay = DecodeAndForwardRelay(target_power=1.0)
    df_soft_relay = DFSoftRelay()

    # AI-based relays
    print("  Training MLP-170 (170 params)...")
    mlp_small = train_mlp(channel_h1, hidden_size=13, seed=0)

    print("  Training MLP-Large (512 params)...")
    mlp_large = train_mlp(channel_h1, hidden_size=45, seed=1)

    print("  Creating Viterbi-Genie (MLSE with perfect CSI)...")
    viterbi_relay = train_viterbi_genie(channel_h1)

    relays = {
        'AF': af_relay,
        'DF-Hard': df_hard_relay,
        'DF-Soft': df_soft_relay,
        'MLP-170': mlp_small,
        'MLP-512': mlp_large,
        'Viterbi-Genie': viterbi_relay,
    }

    # Run experiment
    print("\nRunning BER simulations (5 trials × 50k bits)...")
    print(f"SNR (dB): {' '.join(f'{s:>7d}' for s in SNRS)}")

    results = {r: np.zeros((len(SNRS), N_TRIALS)) for r in relays.keys()}

    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for relay_name, relay in relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[relay_name][si, tr] = ber

            if tr == 0:
                print(f"  SNR {snr:2d} dB: ", end="")
                for r in ['AF', 'DF-Hard', 'DF-Soft', 'MLP-170']:
                    print(f"{r}={results[r][si, 0]:.4f} ", end="")
                print("...")

    # Print results
    print("\n" + "=" * 80)
    print("Results Summary (5 trials, mean ± 95% CI):")
    print("=" * 80)
    print(f"SNR (dB): {' '.join(f'{s:>8d}' for s in SNRS)}")
    for relay_name in sorted(results.keys()):
        mu = results[relay_name].mean(axis=1)
        ci = 1.96 * results[relay_name].std(axis=1) / np.sqrt(N_TRIALS)
        print(f"{relay_name:>15}: " + " ".join(f"{m:8.5f}" for m in mu))

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, relays, SNRS)

    return results, relays


def plot_results(results, relays, snrs):
    """Plot BER curves with all relays."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors and markers
    color_map = {
        'AF': '#1f77b4',           # blue
        'DF-Hard': '#ff7f0e',      # orange
        'DF-Soft': '#d62728',      # red
        'MLP-170': '#2ca02c',      # green
        'MLP-512': '#9467bd',      # purple
        'Viterbi-Genie': '#8c564b', # brown
    }

    marker_map = {
        'AF': 'o',
        'DF-Hard': 's',
        'DF-Soft': '^',
        'MLP-170': 'd',
        'MLP-512': 'v',
        'Viterbi-Genie': 'p',
    }

    # Plot 1: All relays
    ax = axes[0]
    for relay_name in sorted(results.keys()):
        mu = results[relay_name].mean(axis=1)
        ci = 1.96 * results[relay_name].std(axis=1) / np.sqrt(N_TRIALS)

        ax.semilogy(snrs, mu,
                   color=color_map.get(relay_name, 'gray'),
                   marker=marker_map.get(relay_name, 'o'),
                   label=relay_name,
                   linewidth=2.5,
                   markersize=8)
        ax.fill_between(snrs,
                       np.maximum(mu - ci, 1e-6),
                       mu + ci,
                       alpha=0.15,
                       color=color_map.get(relay_name, 'gray'))

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title('All Relays (Classical + AI)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-5, 0.5])

    # Plot 2: AI relays only (zoomed in)
    ax = axes[1]
    for relay_name in ['MLP-170', 'MLP-512', 'Viterbi-Genie']:
        mu = results[relay_name].mean(axis=1)
        ci = 1.96 * results[relay_name].std(axis=1) / np.sqrt(N_TRIALS)

        ax.semilogy(snrs, mu,
                   color=color_map.get(relay_name, 'gray'),
                   marker=marker_map.get(relay_name, 'o'),
                   label=relay_name,
                   linewidth=2.5,
                   markersize=8)
        ax.fill_between(snrs,
                       np.maximum(mu - ci, 1e-6),
                       mu + ci,
                       alpha=0.15,
                       color=color_map.get(relay_name, 'gray'))

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title('AI Relays Comparison (Zoom)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-5, 0.5])

    plt.tight_layout()
    plt.savefig('/tmp/e6_sim_enhanced_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: /tmp/e6_sim_enhanced_comparison.png")

    # Save data
    np.save('/tmp/e6_sim_enhanced_results.npy',
            {'snrs': snrs, 'results': results},
            allow_pickle=True)
    print("  ✓ Saved: /tmp/e6_sim_enhanced_results.npy")


if __name__ == '__main__':
    results, relays = main()
