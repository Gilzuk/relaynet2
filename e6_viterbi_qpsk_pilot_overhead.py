#!/usr/bin/env python3
"""QPSK: Viterbi with realistic 1%-pilot-overhead channel estimation vs genie CSI.

Every prior Viterbi-Genie comparison in this repo assumed the relay has
perfect, free knowledge of the ISI taps. Real relays have to estimate the
channel from pilot symbols, which costs airtime (overhead) and gives an
imperfect (noisy) estimate. This adds "Viterbi-Est-1pct": a per-block LS
channel estimate from a pilot preamble equal to 1% of the data block's
symbol count (250 pilot symbols for a 25,000-symbol QPSK data payload at
N_BITS=50,000), re-estimated fresh every trial/SNR since the pilot
observation noise differs each time. Pilot symbols are transmitted through
the SAME channel_h1 instance immediately before the data (continuous RNG
stream), so the estimate reflects a source realistically close to what the
data itself experiences (short of the shared-fading-per-symbol effects,
which no static-tap estimate can capture -- consistent with how
Viterbi-Genie also only knows the static ISI taps, not the random fading,
throughout this repo's E6 work).

L=3 taps, symmetric ISI+Rayleigh+AWGN hops (ComplexISIRayleighChannel),
same methodology as e6_mlp_qpsk_vs_viterbi.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, MLPQPSKClassifierRelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ComplexISIRayleighChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay
from e6_mlp_qpsk_vs_viterbi import train_mlp_qpsk, taps_for, W, TRAIN_SNRS

SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 5, 50_000
MODULATION = 'qpsk'
L = 3
PILOT_FRAC = 0.01

rng = np.random.default_rng(42)


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0]


def run_ber_trial_pilot_est(channel_h1, channel_h2, num_bits, snr_db, seed, taps):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)
    n_data_syms = tx_symbols.size
    n_pilot_syms = max(int(round(PILOT_FRAC * n_data_syms)), taps.size)

    pilot_source = Source(seed=seed + 500_000, modulation=MODULATION)
    _, pilot_symbols = pilot_source.transmit(2 * n_pilot_syms)

    full_tx = np.concatenate([pilot_symbols, tx_symbols])
    full_rx = channel_h1(full_tx, snr_db)
    y_pilot = full_rx[:n_pilot_syms]
    rx_relay = full_rx[n_pilot_syms:]

    relay = ViterbiMLSEQPSKRelay(pilot_symbols=(y_pilot, pilot_symbols), channel_len=taps.size)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0], n_pilot_syms, n_data_syms


def main():
    print("=" * 80)
    print(f"QPSK: Viterbi with {PILOT_FRAC*100:.0f}% pilot-overhead LS estimation vs genie CSI")
    print(f"L={L} taps, symmetric ISI+Rayleigh+AWGN hops")
    print("=" * 80)

    taps = taps_for(L)
    channel_h1 = ComplexISIRayleighChannel(taps, seed=1)
    channel_h2 = ComplexISIRayleighChannel(taps, seed=2)

    print("\nTraining MLP-QPSK classifier (no pilots needed)...")
    mlp_qpsk = train_mlp_qpsk(ComplexISIRayleighChannel(taps, seed=3), seed=0)
    print(f"  MLP-QPSK: {mlp_qpsk.n_params()} params")

    static_relays = {
        'AF': AmplifyAndForwardRelay(target_power=1.0),
        'DF-Hard': DFHardRelay(MODULATION),
        'DF-Soft': DFSoftRelay(),
        'MLP-QPSK': mlp_qpsk,
        'Viterbi-Genie': ViterbiMLSEQPSKRelay(channel_taps=taps),
    }
    all_names = list(static_relays.keys()) + [f'Viterbi-Est-{PILOT_FRAC*100:.0f}pct']
    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in all_names}

    n_pilot_report, n_data_report = None, None
    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for name, relay in static_relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber
            ber_est, n_pilot_report, n_data_report = run_ber_trial_pilot_est(
                channel_h1, channel_h2, N_BITS, snr, seed=seed_base, taps=taps)
            results[f'Viterbi-Est-{PILOT_FRAC*100:.0f}pct'][si, tr] = ber_est

    summary = {}
    for name in all_names:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>20}: " + " ".join(f"{m:7.4f}" for m in mu))

    overhead_pct = 100.0 * n_pilot_report / (n_pilot_report + n_data_report)
    print(f"\n  Pilot preamble: {n_pilot_report} symbols per {n_data_report}-symbol data block "
          f"= {overhead_pct:.2f}% airtime overhead")

    plot_results(summary)
    return summary


def plot_results(summary):
    fig, ax = plt.subplots(figsize=(9, 6.5))

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728',
                 'MLP-QPSK': '#2ca02c', 'Viterbi-Genie': '#8c564b',
                 f'Viterbi-Est-{PILOT_FRAC*100:.0f}pct': '#e377c2'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'MLP-QPSK': 'd',
                  'Viterbi-Genie': 'p', f'Viterbi-Est-{PILOT_FRAC*100:.0f}pct': 'X'}

    for name, (mu, ci) in summary.items():
        ax.semilogy(SNRS, mu, color=color_map[name], marker=marker_map[name],
                    label=name, linewidth=2.5, markersize=8)
        ax.fill_between(SNRS, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title(f'QPSK, L={L} taps — Viterbi-Genie vs {PILOT_FRAC*100:.0f}%-pilot-overhead LS estimate',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-3, 0.6])

    plt.tight_layout()
    plt.savefig('/tmp/e6_viterbi_qpsk_pilot_overhead.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_viterbi_qpsk_pilot_overhead.png")

    np.save('/tmp/e6_viterbi_qpsk_pilot_overhead_results.npy',
            {'snrs': SNRS, 'summary': summary}, allow_pickle=True)
    print("  Saved: /tmp/e6_viterbi_qpsk_pilot_overhead_results.npy")


if __name__ == '__main__':
    main()
