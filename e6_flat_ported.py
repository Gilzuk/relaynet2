#!/usr/bin/env python3
"""E6_FLAT: Flat (memoryless) unknown channels — ported to relaynet.

Three falsification control tests showing classical relays don't fail on unknownness
per se, but on memory (ISI). All these channels have NO memory.

Cases:
  F1 - Unknown Phase: y = e^{jθ}·s + n, DBPSK source, complex I/Q → MLP-169 (22→7)
  F2 - Unknown Gain: y = g·x + n, real BPSK → MLP-170 (11→13→1)
  F3 - I/Q Imbalance: y = a+ if x>0 else -a- + n, real BPSK → MLP-170

Classical baseline for phase: differential detection sign(Re{y[i]·conj(y[i-1])})
Classical DF for gain/asymmetry: sign threshold.

Expected result: **MLP gap ≤ 0.0036 everywhere** — classical does NOT fail.

Following PORTING.md section 2 acceptance criteria.
"""

import numpy as np
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, MLPRelay
from relaynet.channels import FlatPhaseChannel, FlatGainChannel, BranchAsymmetryChannel, RayleighChannel
from relaynet.modulation.bpsk import calculate_ber
from relaynet.nodes import Source, Destination

# Configuration
W = 11          # window length for real signals
W_COMPLEX = 2*W # I/Q pairs for complex signals
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 10, 100_000

# Global RNG
rng = np.random.default_rng(21)


def diff_encode(x):
    """DBPSK differential encoding: running product."""
    s = np.empty_like(x, dtype=float)
    s[0] = 1.0
    for i in range(1, len(x)):
        s[i] = s[i-1] * x[i]
    return s


def diff_detect(y_complex):
    """Differential detection: sign(Re{y[i] · conj(y[i-1])}).

    Returns array of same length as input, with first element undefined (set to +1).
    """
    product = y_complex[1:] * np.conj(y_complex[:-1])
    decisions = np.sign(product.real) + (product.real == 0)
    # Prepend 1.0 for first bit (boundary condition)
    return np.r_[1.0, decisions]


def create_channel(kind, seed=None):
    """Create a flat channel.

    Parameters
    ----------
    kind : str
        One of: 'phase', 'gain', 'iqimb'.

    Returns
    -------
    callable
        Channel function f(signal, snr_db) -> noisy_signal.
    """
    if kind == 'phase':
        return FlatPhaseChannel(seed=seed)
    elif kind == 'gain':
        return FlatGainChannel(seed=seed)
    elif kind == 'iqimb':
        return BranchAsymmetryChannel(seed=seed)
    else:
        raise ValueError(f"Unknown channel kind: {kind}")


def extract_windows(y, window_size=W):
    """Extract sliding windows from signal.

    Handles both real and complex signals.

    Parameters
    ----------
    y : ndarray
        Input signal (real or complex).
    window_size : int
        Window size.

    Returns
    -------
    windows : ndarray
        Windows of shape (n, window_size) or (n, 2*window_size) for complex.
    """
    if np.iscomplexobj(y):
        # Complex: extract I and Q separately, then concatenate
        pad_size = window_size // 2
        yp_real = np.pad(y.real, (pad_size, pad_size), mode='constant')
        yp_imag = np.pad(y.imag, (pad_size, pad_size), mode='constant')

        wins_real = np.lib.stride_tricks.sliding_window_view(yp_real, window_size)
        wins_imag = np.lib.stride_tricks.sliding_window_view(yp_imag, window_size)

        # Concatenate I and Q: (n_samples, 2*window_size)
        return np.concatenate([wins_real, wins_imag], axis=1)
    else:
        # Real: standard sliding window
        pad_size = window_size // 2
        yp = np.pad(y, (pad_size, pad_size), mode='constant')
        return np.lib.stride_tricks.sliding_window_view(yp, window_size)


def train_mlp(kind, seed=2, n_train=150_000, epochs=25, batch_size=256):
    """Train MLP on synthetic data for a flat channel type.

    Parameters
    ----------
    kind : str
        'phase', 'gain', or 'iqimb'.
    seed : int
    n_train : int
        Total training samples.
    epochs : int
    batch_size : int

    Returns
    -------
    mlp : MLPRelay
        Trained relay.
    n_params : int
        Number of parameters.
    """
    channel = create_channel(kind, seed=1)

    # Determine MLP input/hidden sizes
    if kind == 'phase':
        input_size = 2 * W  # I/Q pairs
        hidden_size = 7
        window_size = None  # No windowing, already extracted in extract_windows
    else:
        input_size = W
        hidden_size = 13
        window_size = W  # Use windowing in process()

    mlp = MLPRelay(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        window_size=window_size,
        seed=seed
    )

    # Collect training data
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []

    for snr_db in TRAIN_SNRS:
        # Generate random symbols
        bits = rng.integers(0, 2, per_snr)

        if kind == 'phase':
            # DBPSK: differential binary phase shift keying
            x = 1.0 - 2.0 * bits  # BPSK: {-1, +1}
            x_dbpsk = diff_encode(x)  # differential encoding
            # Pass through phase channel
            y = channel(x_dbpsk, snr_db)
            # Extract windows (complex)
            windows = extract_windows(y, window_size=W)
            X_list.append(windows)
            T_list.append(x)  # target is original BPSK
        else:
            # Standard BPSK
            x = 1.0 - 2.0 * bits
            y = channel(x, snr_db)
            # Extract windows (real)
            windows = extract_windows(y, window_size=W)
            X_list.append(windows)
            T_list.append(x)

    X = np.vstack(X_list)
    T = np.concatenate(T_list)

    # Train
    mlp.train_on_data(X, T, epochs=epochs, batch_size=batch_size, lr=3e-3)

    n_params = sum(p.size for p in mlp.params)
    return mlp, n_params


def run_ber_trial_phase(relay_name, relay, y_received, bits, snr_db, hop2_seed):
    """Run a single BER trial for phase channel.

    `y_received` (hop 1's output) and `bits` are computed ONCE per trial by the
    caller and shared across AF/DF/MLP, so all three relays are compared on the
    identical unknown-phase realization -- only `hop2_seed` (also shared across
    relays, but distinct from the hop-1/bits stream) determines hop 2's fading
    and noise, again identical across relays for a paired, low-variance
    comparison.

    Parameters
    ----------
    relay_name : str
        'AF', 'DF', 'MLP'.
    relay : Relay
    y_received : ndarray
        Hop 1's complex output (shared across relays for this trial).
    bits : ndarray
        The trial's transmitted bits (shared across relays).
    snr_db : float
    hop2_seed : int
        Seed for hop 2's fading/noise draw (shared across relays this trial).

    Returns
    -------
    ber : float
    """
    rng2 = np.random.default_rng(hop2_seed)

    # Relay processing
    if relay_name == 'AF':
        # AF: normalize, then do differential detection at destination
        gain = np.sqrt(1.0 / (np.mean(np.abs(y_received) ** 2) + 1e-12))
        relay_out = gain * y_received
        # Hop 2
        h = np.abs((rng2.standard_normal(N_BITS) + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2))
        sigma = 10 ** (-snr_db / 20.0)
        n = sigma * (rng2.standard_normal(N_BITS) + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2)
        y_dest = h * relay_out + n
        # Differential detection (full-length, with first element = 1.0)
        bit_out = diff_detect(y_dest)
        bit_out = (bit_out < 0).astype(int)
        # Skip first bit (differential encoding boundary)
        return np.mean(bit_out[1:] != bits[1:])

    elif relay_name == 'DF':
        # DF: differential detection at relay
        bit_out = diff_detect(y_received)
        relay_out = 1.0 - 2.0 * (bit_out < 0).astype(int)  # {-1, +1}
        # Hop 2
        h = np.abs((rng2.standard_normal(N_BITS) + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2))
        sigma = 10 ** (-snr_db / 20.0)
        n = sigma * rng2.standard_normal(N_BITS)
        y_dest = h * relay_out + n
        # Bit detection
        bit_out_2 = (y_dest < 0).astype(int)
        # Skip first bit (differential encoding boundary)
        return np.mean(bit_out_2[1:] != bits[1:])

    elif relay_name == 'MLP':
        # MLP: window-based neural relay (complex, pre-extracted windows)
        windows = extract_windows(y_received, window_size=W)
        relay_out = relay.fwd(windows)
        # Normalize output power
        relay_out_norm = relay_out / (np.sqrt(np.mean(relay_out ** 2)) + 1e-12)
        # Hop 2
        h = np.abs((rng2.standard_normal(N_BITS) + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2))
        sigma = 10 ** (-snr_db / 20.0)
        n = sigma * rng2.standard_normal(N_BITS)
        y_dest = h * relay_out_norm + n
        # Bit detection
        bit_out = (y_dest < 0).astype(int)
        # Skip first bit (differential encoding boundary)
        return np.mean(bit_out[1:] != bits[1:])

    raise ValueError(f"Unknown relay: {relay_name}")


def run_ber_trial_real(relay_name, relay, y_received, bits, snr_db, hop2_seed):
    """Run a single BER trial for real-valued channels (gain, iqimb).

    `y_received` (hop 1's output) and `bits` are computed ONCE per trial by the
    caller and shared across AF/DF/MLP -- see `run_ber_trial_phase`'s docstring
    for why (paired comparison: same unknown-channel + hop-2 realization for
    all three relays, so gaps reflect the relay, not sampling noise).

    Parameters
    ----------
    relay_name : str
        'AF', 'DF', 'MLP'.
    relay : Relay
    y_received : ndarray
        Hop 1's real output (shared across relays for this trial).
    bits : ndarray
        The trial's transmitted bits (shared across relays).
    snr_db : float
    hop2_seed : int
        Seed for hop 2's fading/noise draw (shared across relays this trial).

    Returns
    -------
    ber : float
    """
    # Relay processing
    if relay_name == 'AF':
        gain = np.sqrt(1.0 / (np.mean(y_received ** 2) + 1e-12))
        relay_out = gain * y_received
    elif relay_name == 'DF':
        relay_out = np.sign(y_received) + (y_received == 0)
    elif relay_name == 'MLP':
        relay_out = relay.process(y_received)
    else:
        raise ValueError(f"Unknown relay: {relay_name}")

    # Hop 2 (Rayleigh, coherently compensated)
    rng2 = np.random.default_rng(hop2_seed)
    h = np.abs((rng2.standard_normal(N_BITS) + 1j * rng2.standard_normal(N_BITS)) / np.sqrt(2))
    sigma = 10 ** (-snr_db / 20.0)
    n = sigma * rng2.standard_normal(N_BITS)
    y_dest = h * relay_out + n

    # Bit detection
    bit_out = (y_dest < 0).astype(int)
    return np.mean(bit_out != bits)


def run_experiment(kind, mlp_relay):
    """Run flat channel experiment.

    Parameters
    ----------
    kind : str
        'phase', 'gain', or 'iqimb'.
    mlp_relay : MLPRelay
        Trained relay.

    Returns
    -------
    results : dict
        Dictionary with keys 'AF', 'DF', 'MLP'.
    """
    channel = create_channel(kind, seed=2)
    is_phase = (kind == 'phase')

    results = {r: np.zeros((len(SNRS), N_TRIALS)) for r in ('AF', 'DF', 'MLP')}

    af_relay = AmplifyAndForwardRelay(target_power=1.0)
    df_relay = DecodeAndForwardRelay(target_power=1.0)

    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 9000 * si + tr

            # Draw bits + hop-1 channel ONCE per trial, shared across AF/DF/MLP
            # (paired comparison -- same unknown-channel realization for all
            # three relays; only hop2_seed differs from seed_base so hop 2's
            # fading/noise draw is independent of the bits/hop1 stream, but is
            # itself likewise shared across the three relays this trial).
            rng_trial = np.random.default_rng(seed_base)
            bits = rng_trial.integers(0, 2, N_BITS)
            x_bpsk = 1.0 - 2.0 * bits
            hop2_seed = seed_base + 5_000_000

            if is_phase:
                run_fn = run_ber_trial_phase
                relays = [('AF', None), ('DF', None), ('MLP', mlp_relay)]
                y_received = channel(diff_encode(x_bpsk), snr)
            else:
                run_fn = run_ber_trial_real
                relays = [('AF', af_relay), ('DF', df_relay), ('MLP', mlp_relay)]
                y_received = channel(x_bpsk, snr)

            for name, relay_obj in relays:
                ber = run_fn(name, relay_obj, y_received, bits, snr, hop2_seed)
                results[name][si, tr] = ber

            if tr == 0:
                print(f"  SNR {snr:2d} dB: AF={results['AF'][si, tr]:.4f}, " +
                      f"DF={results['DF'][si, tr]:.4f}, MLP={results['MLP'][si, tr]:.4f}")

    return {
        r: (v.mean(1), 1.96 * v.std(1) / np.sqrt(N_TRIALS))
        for r, v in results.items()
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("E6_FLAT: Memoryless Unknown Channels (Ported to relaynet)")
    print("=" * 70)

    setups = [
        ('phase', 'F1 unknown phase (DBPSK)'),
        ('gain', 'F2 unknown gain (control)'),
        ('iqimb', 'F3 per-branch gain asymmetry'),
    ]

    all_results = {}

    for kind, tag in setups:
        print(f"\nTraining MLP for {kind}...")
        mlp, n_params = train_mlp(kind, seed=2)
        print(f"  {kind}: {n_params} params")

        print(f"\n{tag}")
        print(f"  SNR (dB): " + " ".join(f"{s:>6d}" for s in SNRS))

        results = run_experiment(kind, mlp)
        all_results[kind] = results

        for relay in ('AF', 'DF', 'MLP'):
            mu, ci = results[relay]
            print(f"  {relay:>4}: " + " ".join(f"{m:6.4f}" for m in mu))

        # Compute max gap (should be ≤ 0.0036 for control)
        ber_mlp = results['MLP'][0]
        ber_df = results['DF'][0]
        gap = np.abs(ber_mlp - ber_df).max()
        print(f"  → Max MLP-DF gap: {gap:.6f} (should be ≤ 0.0036 for control)")

    # Save results
    output_path = '/tmp/e6_flat_ported_results.npy'
    np.save(output_path, {'setups': setups, 'results': all_results, 'snrs': SNRS}, allow_pickle=True)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("E6_FLAT: Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
