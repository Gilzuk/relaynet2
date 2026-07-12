#!/usr/bin/env python3
"""QPSK Viterbi: worst-case vs medium vs ideal channel knowledge.

Three CSI tiers for ViterbiMLSEQPSKRelay, all on the same L=3-tap
symmetric ISI+Rayleigh+AWGN channel used throughout this line of work:

  Worst case  : 5-pilot LS estimate  -- just above the L=3 identifiability
                floor (3 unknowns, 5 equations), heavily noise-limited.
  Medium case : 20-pilot LS estimate -- a realistic partial/imperfect
                estimate, well short of the ~250-pilot (1% overhead)
                near-ideal case used previously.
  Ideal case  : Viterbi-Genie-EhScaled -- perfect knowledge of both the
                ISI taps AND the average Rayleigh fading gain E[|h|]
                (see e6_viterbi_qpsk_pilot_overhead.py for why plain
                "Viterbi-Genie" is NOT actually the ideal case on this
                channel -- it's fading-blind).

Classical relays (AF/DF-Hard/DF-Soft, zero CSI) and MLP-QPSK (no pilots,
learned offline) included for context. N_TRIALS reverted to 5 (the
project's standard iteration scale used throughout most of this session,
before the N=20 confirmation run).
"""

import numpy as np
import matplotlib.pyplot as plt
from relaynet.relays import AmplifyAndForwardRelay, ViterbiMLSEQPSKRelay
from relaynet.channels import ComplexISIRayleighChannel
from relaynet.modulation import calculate_ber
from relaynet.nodes import Source, Destination
from e6_sim_enhanced_multimod import DFHardRelay, DFSoftRelay
from e6_mlp_qpsk_vs_viterbi import train_mlp_qpsk, taps_for

SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS = 10, 100_000  # project-standard scale for thesis integration
MODULATION = 'qpsk'
L = 3
E_H = np.sqrt(np.pi) / 2

PILOT_TIERS = {
    'Viterbi-Worst-5pilots': 5,
    'Viterbi-Medium-20pilots': 20,
}


def run_ber_trial(relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_h1(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_h2(relay_out, snr_db)
    rx_bits = dest.receive(rx_dest)
    return calculate_ber(tx_bits, rx_bits)[0]


def run_ber_trial_pilot_est(channel_h1, channel_h2, num_bits, snr_db, seed, taps, n_pilot_syms):
    source = Source(seed=seed, modulation=MODULATION)
    dest = Destination(modulation=MODULATION)
    tx_bits, tx_symbols = source.transmit(num_bits)

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
    return calculate_ber(tx_bits, rx_bits)[0]


def main():
    print("=" * 80)
    print("QPSK Viterbi: worst-case vs medium vs ideal channel knowledge")
    print(f"L={L} taps, symmetric ISI+Rayleigh+AWGN hops, N_TRIALS={N_TRIALS}")
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
        'Viterbi-Ideal-EhScaled': ViterbiMLSEQPSKRelay(channel_taps=taps * E_H),
    }
    all_names = list(static_relays.keys()) + list(PILOT_TIERS.keys())
    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in all_names}

    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr
            for name, relay in static_relays.items():
                ber = run_ber_trial(relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber
            for tier_name, n_pilots in PILOT_TIERS.items():
                ber_est = run_ber_trial_pilot_est(
                    channel_h1, channel_h2, N_BITS, snr, seed=seed_base, taps=taps, n_pilot_syms=n_pilots)
                results[tier_name][si, tr] = ber_est

    summary = {}
    for name in all_names:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>24}: " + " ".join(f"{m:7.4f}" for m in mu))

    n_data_syms = N_BITS // 2  # qpsk: 2 bits/symbol
    print("\n  Pilot overhead by tier:")
    for tier_name, n_pilots in PILOT_TIERS.items():
        pct = 100.0 * n_pilots / (n_pilots + n_data_syms)
        print(f"    {tier_name}: {n_pilots} pilots / {n_data_syms}-symbol block = {pct:.3f}% overhead")

    plot_results(summary)
    return summary


ZOOM_SNR_MIN = 12  # dB, start of the high-SNR zoom panel


def plot_results(summary, snrs=None):
    if snrs is None:
        snrs = SNRS

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    color_map = {'AF': '#1f77b4', 'DF-Hard': '#ff7f0e', 'DF-Soft': '#d62728',
                 'MLP-QPSK': '#2ca02c', 'Viterbi-Ideal-EhScaled': '#8c564b',
                 'Viterbi-Medium-20pilots': '#e377c2', 'Viterbi-Worst-5pilots': '#7f7f7f'}
    marker_map = {'AF': 'o', 'DF-Hard': 's', 'DF-Soft': '^', 'MLP-QPSK': 'd',
                  'Viterbi-Ideal-EhScaled': 'p', 'Viterbi-Medium-20pilots': 'X',
                  'Viterbi-Worst-5pilots': 'v'}

    # Panel 1: full SNR range
    ax = axes[0]
    for name, (mu, ci) in summary.items():
        ax.semilogy(snrs, mu, color=color_map[name], marker=marker_map[name],
                    label=name, linewidth=2.5, markersize=8)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-6), mu + ci, alpha=0.15, color=color_map[name])
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax.set_title(f'QPSK, L={L} taps — full range', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_ylim([1e-3, 0.6])

    # Panel 2: zoom on the high-SNR regime, where CSI-quality differences matter most
    ax = axes[1]
    zoom_mask = snrs >= ZOOM_SNR_MIN
    for name, (mu, ci) in summary.items():
        ax.semilogy(snrs[zoom_mask], mu[zoom_mask], color=color_map[name], marker=marker_map[name],
                    label=name, linewidth=2.5, markersize=9)
        ax.fill_between(snrs[zoom_mask], np.maximum(mu[zoom_mask] - ci[zoom_mask], 1e-4),
                        mu[zoom_mask] + ci[zoom_mask], alpha=0.15, color=color_map[name])
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title(f'Zoom: SNR ≥ {ZOOM_SNR_MIN}dB', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    zoom_mu_all = np.concatenate([mu[zoom_mask] for mu, ci in summary.values()])
    zoom_ci_all = np.concatenate([ci[zoom_mask] for mu, ci in summary.values()])
    ax.set_ylim([max((zoom_mu_all - zoom_ci_all).min() * 0.7, 1e-4),
                 (zoom_mu_all + zoom_ci_all).max() * 1.3])

    fig.suptitle(f'QPSK Viterbi CSI quality: worst (5 pilots) / medium (20 pilots) / ideal',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/e6_viterbi_qpsk_partial_csi.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: /tmp/e6_viterbi_qpsk_partial_csi.png")

    np.save('/tmp/e6_viterbi_qpsk_partial_csi_results.npy',
            {'snrs': snrs, 'summary': summary}, allow_pickle=True)
    print("  Saved: /tmp/e6_viterbi_qpsk_partial_csi_results.npy")


if __name__ == '__main__':
    main()
