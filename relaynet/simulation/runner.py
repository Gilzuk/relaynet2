"""Simulation runner for two-hop relay systems."""

import numpy as np

from relaynet.nodes import Source, Destination
from relaynet.modulation.bpsk import calculate_ber
from relaynet.channels.awgn import awgn_channel


# ── Modulation-aware relay processing helpers ───────────────────────

def _df_constellation_detect(received_signal, modulation, target_power=1.0):
    """DF-like processing for complex signals: nearest constellation point.

    For QPSK the I and Q decisions are independent sign decisions.
    For 16-QAM each component is quantised to {−3, −1, +1, +3}/√10.
    """
    if modulation == "qpsk":
        I_clean = np.sign(received_signal.real)
        Q_clean = np.sign(received_signal.imag)
        clean = (I_clean + 1j * Q_clean) / np.sqrt(2)
    elif modulation == "qam16":
        norm = np.sqrt(10.0)
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        I = received_signal.real * norm
        Q = received_signal.imag * norm
        I_idx = np.argmin(np.abs(I[:, None] - levels[None, :]), axis=1)
        Q_idx = np.argmin(np.abs(Q[:, None] - levels[None, :]), axis=1)
        clean = (levels[I_idx] + 1j * levels[Q_idx]) / norm
    elif modulation == "psk16":
        # Nearest of 16 uniformly-spaced unit-circle points
        angles = np.angle(received_signal)
        angles = np.where(angles < 0, angles + 2.0 * np.pi, angles)
        ref_angles = 2.0 * np.pi * np.arange(16) / 16.0
        idx = np.round(angles * 16.0 / (2.0 * np.pi)).astype(int) % 16
        clean = np.exp(1j * ref_angles[idx])
    else:
        raise ValueError(f"Unsupported modulation for complex DF: {modulation}")

    pwr = np.mean(np.abs(clean) ** 2)
    if pwr > 0:
        clean = clean * np.sqrt(target_power / pwr)
    return clean


def _process_relay(relay, received_signal, modulation="bpsk"):
    """Route signal through *relay* with modulation awareness.

    * **BPSK** (real signal) — delegates directly to ``relay.process()``.
    * **QPSK / QAM16** (complex signal after a fading or AWGN channel):
      - **AF**: amplifies the complex signal as-is (power normalisation
        handles complex magnitudes correctly).
      - **DF**: nearest-constellation-point detection + re-transmission.
      - **All other relays** (MLP, Hybrid, VAE, CGAN, Transformer,
        Mamba, …): process the I and Q components independently through
        the real-valued neural network, then recombine.

    Parameters
    ----------
    relay : Relay
    received_signal : numpy.ndarray
    modulation : str
    """
    from relaynet.relays.af import AmplifyAndForwardRelay
    from relaynet.relays.df import DecodeAndForwardRelay

    # ── Handle CSI tuple if provided ───────────────────────────────
    if isinstance(received_signal, tuple):
        y, h_csi = received_signal
    else:
        y = received_signal
        h_csi = None

    # Real signal → vanilla relay processing (BPSK or already I/Q split)
    if not np.iscomplexobj(y):
        if h_csi is not None:
            return relay.process((y, np.abs(h_csi)))
        return relay.process(y)

    # ── Complex signal (QPSK / QAM16) ──────────────────────────────

    if isinstance(relay, AmplifyAndForwardRelay):
        # AF: analogue amplification — works natively on complex
        return relay.process(y)

    if isinstance(relay, DecodeAndForwardRelay):
        # DF: ML constellation-point detection
        return _df_constellation_detect(
            y, modulation,
            target_power=relay.target_power,
        )

    # AI-based relays: check for 2D classification mode first
    if getattr(relay, 'classify_2d', False):
        # 16-class 2D: relay handles complex signal natively
        return relay.process(y)

    # AI-based relays: process I and Q independently
    if h_csi is not None:
        h_mag = np.abs(h_csi).copy()
        out_i = relay.process((y.real.copy(), h_mag))
        out_q = relay.process((y.imag.copy(), h_mag))
    else:
        out_i = relay.process(y.real.copy())
        out_q = relay.process(y.imag.copy())
        
    return out_i + 1j * out_q


# ── Public simulation API ──────────────────────────────────────────

def simulate_transmission(relay, num_bits, snr_db, seed=None,
                           channel_fn=None, modulation="bpsk"):
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
    modulation : str, optional
        Modulation scheme: ``'bpsk'``, ``'qpsk'``, or ``'qam16'``.
        Default ``'bpsk'``.

    Returns
    -------
    ber : float
        Bit Error Rate.
    num_errors : int
        Number of bit errors.
    """
    if channel_fn is None:
        channel_fn = awgn_channel

    source = Source(seed=seed, modulation=modulation)
    destination = Destination(modulation=modulation)

    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_relay = channel_fn(tx_symbols, snr_db)

    relay_out = _process_relay(relay, rx_relay, modulation)

    rx_dest = channel_fn(relay_out, snr_db)
    
    # Destination only cares about the signal, not CSI, assuming perfectly equalized
    if isinstance(rx_dest, tuple):
        rx_dest_signal = rx_dest[0]
    else:
        rx_dest_signal = rx_dest
        
    rx_bits = destination.receive(rx_dest_signal)

    return calculate_ber(tx_bits, rx_bits)


def run_monte_carlo(relay, snr_range, num_bits_per_trial=10000,
                    num_trials=10, channel_fn=None, seed_offset=0,
                    modulation="bpsk"):
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
    modulation : str, optional
        Modulation scheme: ``'bpsk'``, ``'qpsk'``, or ``'qam16'``.
        Default ``'bpsk'``.

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
                modulation=modulation,
            )
            trial_bers.append(ber)
        ber_trials[i] = trial_bers
        ber_values[i] = np.mean(trial_bers)

    return snr_values, ber_values, ber_trials
