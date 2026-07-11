#!/usr/bin/env python3
"""E6 addendum: Viterbi (MLSE) relay baselines — ported to relaynet.

Viterbi-genie: MLSE with perfect channel knowledge (upper bound).
Viterbi-est:   200 pilot symbols -> LS channel estimate -> MLSE (practical).

Following PORTING.md section 1 acceptance criteria.
"""

import numpy as np
from relaynet.relays import ViterbiMLSERelay
from relaynet.channels import ISIChannel, awgn_channel, RayleighChannel
from relaynet.modulation.bpsk import calculate_ber
from relaynet.nodes import Source, Destination

# Configuration
H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)
SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS, N_PILOT = 5, 50_000, 200

# Global RNG
rng = np.random.default_rng(43)


def create_channels():
    """Create ISI channel (hop 1) and AWGN/Rayleigh (hop 2)."""
    isi_channel = ISIChannel(H_ISI, seed=1)
    awgn_ch = awgn_channel
    rayleigh_ch = RayleighChannel(seed=2)
    return isi_channel, awgn_ch, rayleigh_ch


def run_ber_trial(relay, hop1_channel, hop2_channel, num_bits, snr_db, seed=None):
    """Run a single BER trial.

    Parameters
    ----------
    relay : Relay
        Relay strategy (Viterbi).
    hop1_channel : callable
        Hop 1 channel (ISI).
    hop2_channel : callable
        Hop 2 channel (AWGN or Rayleigh).
    num_bits : int
        Number of bits to transmit.
    snr_db : float
        SNR in dB.
    seed : int, optional
        Random seed.

    Returns
    -------
    ber : float
        Bit Error Rate.
    """
    if seed is None:
        seed = 42

    source = Source(seed=seed, modulation='bpsk')
    destination = Destination(modulation='bpsk')

    # Transmit
    tx_bits, tx_symbols = source.transmit(num_bits)

    # Hop 1 (ISI + AWGN)
    rx_relay = hop1_channel(tx_symbols, snr_db)

    # Relay (Viterbi decoder)
    relay_out = relay.process(rx_relay)

    # Hop 2 (AWGN or Rayleigh)
    rx_dest = hop2_channel(relay_out, snr_db)

    # Receive
    rx_bits = destination.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)[0]


def run_experiment(hop2_kind, use_genie_csi=True):
    """Run Viterbi experiment.

    Parameters
    ----------
    hop2_kind : str
        'awgn' or 'rayleigh'.
    use_genie_csi : bool
        If True, use perfect CSI. If False, estimate from pilots.

    Returns
    -------
    results : ndarray
        BER at each SNR point.
    """
    isi_channel, awgn_ch, rayleigh_ch = create_channels()
    hop2_channel = rayleigh_ch if hop2_kind == 'rayleigh' else awgn_ch

    results = np.zeros((len(SNRS), N_TRIALS))

    for si, snr_db in enumerate(SNRS):
        for tr in range(N_TRIALS):
            # Create fresh RNG for this trial
            seed = 1000 * si + tr + (0 if use_genie_csi else 5000)
            trial_rng = np.random.default_rng(seed)

            # Generate symbols
            bits = trial_rng.integers(0, 2, N_BITS)
            symbols = 1.0 - 2.0 * bits

            # Hop 1
            y_received = isi_channel(symbols, snr_db)

            # Channel estimation or CSI
            if use_genie_csi:
                # Genie CSI: use true channel
                relay = ViterbiMLSERelay(channel_taps=H_ISI)
            else:
                # Estimate from pilots
                y_pilot = y_received[:N_PILOT]
                x_pilot = symbols[:N_PILOT]
                relay = ViterbiMLSERelay(pilot_symbols=(y_pilot, x_pilot))

            # Relay (Viterbi decode)
            relay_out = relay.process(y_received)

            # Hop 2
            y_dest = hop2_channel(relay_out, snr_db)

            # Bit decisions
            bit_decisions = (y_dest < 0).astype(int)

            # BER
            results[si, tr] = np.mean(bit_decisions != bits)

            if tr == 0:
                csi_tag = 'genie' if use_genie_csi else 'est'
                print(f"  SNR {snr_db:2d} dB, Viterbi-{csi_tag}, trial {tr}: BER={results[si, tr]:.4f}")

    return results.mean(axis=1)


def main():
    """Main entry point."""
    print("=" * 70)
    print("E6_VITERBI: MLSE Baselines for Unknown ISI (Ported to relaynet)")
    print("=" * 70)

    setups = [
        ('awgn', 'S1: unknown ISI -> AWGN'),
        ('rayleigh', 'S2: unknown ISI -> Rayleigh'),
    ]

    for hop2_kind, tag in setups:
        print(f"\n{tag}")
        print(f"  SNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))

        # Run Viterbi-genie
        print("  Running Viterbi-genie...")
        res_genie = run_experiment(hop2_kind, use_genie_csi=True)

        # Run Viterbi-est
        print("  Running Viterbi-est...")
        res_est = run_experiment(hop2_kind, use_genie_csi=False)

        # Print results
        print(f"  VIT-genie: " + " ".join(f"{m:7.4f}" for m in res_genie))
        print(f"  VIT-est  : " + " ".join(f"{m:7.4f}" for m in res_est))

        # Save
        output = {'VIT-genie': res_genie, 'VIT-est': res_est}
        output_path = f'/tmp/e6_viterbi_{hop2_kind}.npy'
        np.save(output_path, output, allow_pickle=True)
        print(f"  Saved to {output_path}")

    print("\n" + "=" * 70)
    print("E6_VITERBI: Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
