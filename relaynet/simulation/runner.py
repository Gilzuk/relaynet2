"""Simulation runner for two-hop relay systems."""

import numpy as np

from relaynet.nodes import Source, Destination
from relaynet.modulation.bpsk import calculate_ber
from relaynet.channels.awgn import awgn_channel


def simulate_transmission(relay, num_bits, snr_db, seed=None,
                           channel_fn=None):
    """Simulate a single two-hop transmission through a relay.

    Parameters
    ----------
    relay : Relay
        Any relay implementing ``process(received_signal)``.
    num_bits : int
        Number of bits to transmit.
    snr_db : float
        SNR in dB for both hops.
    seed : int, optional
        Random seed for reproducibility.
    channel_fn : callable, optional
        Channel function ``f(signal, snr_db) -> noisy_signal``.
        Defaults to :func:`relaynet.channels.awgn.awgn_channel`.

    Returns
    -------
    ber : float
        Bit Error Rate.
    num_errors : int
        Number of bit errors.
    """
    if channel_fn is None:
        channel_fn = awgn_channel

    source = Source(seed=seed)
    destination = Destination()

    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_fn(tx_symbols, snr_db)
    relay_out = relay.process(rx_relay)
    rx_dest = channel_fn(relay_out, snr_db)
    rx_bits = destination.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)


def run_monte_carlo(relay, snr_range, num_bits_per_trial=10000,
                    num_trials=10, channel_fn=None, seed_offset=0):
    """Run a Monte Carlo BER simulation over a range of SNR values.

    Parameters
    ----------
    relay : Relay
        Any relay implementing ``process(received_signal)``.
    snr_range : array-like
        SNR values in dB.
    num_bits_per_trial : int
        Number of bits per trial.
    num_trials : int
        Number of independent trials per SNR point.
    channel_fn : callable, optional
        Channel function. Defaults to AWGN.
    seed_offset : int
        Base seed offset.

    Returns
    -------
    snr_values : numpy.ndarray
    ber_values : numpy.ndarray
        Mean BER at each SNR.
    ber_trials : numpy.ndarray, shape (len(snr_range), num_trials)
        Per-trial BER values (useful for confidence intervals).
    """
    snr_values = np.array(snr_range)
    ber_values = np.zeros(len(snr_values))
    ber_trials = np.zeros((len(snr_values), num_trials))

    for i, snr_db in enumerate(snr_values):
        trial_bers = []
        for trial in range(num_trials):
            ber, _ = simulate_transmission(
                relay, num_bits_per_trial, snr_db,
                seed=seed_offset + trial,
                channel_fn=channel_fn,
            )
            trial_bers.append(ber)
        ber_trials[i] = trial_bers
        ber_values[i] = np.mean(trial_bers)

    return snr_values, ber_values, ber_trials
