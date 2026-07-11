#!/usr/bin/env python3
"""E6: Unknown-channel two-hop relay experiment — ported to relaynet.

Hop 1 = unknown channel (ISI / nonlinear-bias) or control (AWGN / Rayleigh).
Relays: AF, DF (sign), MLP-170 (window 11 -> 13 tanh -> 1 tanh).
Hop 2 = AWGN or coherently-compensated Rayleigh at the same SNR.

Ported to use relaynet's Channel/Relay/runner infrastructure.
SNR convention matches thesis: gamma = 1/sigma^2, single-hop AWGN BER = Q(sqrt(gamma)).

Following PORTING.md section 1 acceptance criteria.
"""

import numpy as np
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, MLPRelay
from relaynet.channels import ISIChannel, NonlinearBiasChannel, RayleighChannel, awgn_channel
from relaynet.channels.awgn import calculate_snr
from relaynet.modulation.bpsk import calculate_ber
from relaynet.nodes import Source, Destination

# Configuration
W = 11          # window length
HID = 13        # hidden units -> params = 11*13+13 + 13*1+1 = 170
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000

# Global RNG (for reproducibility)
rng = np.random.default_rng(42)


def create_channel(kind, seed=None):
    """Create a channel callable based on kind.

    Parameters
    ----------
    kind : str
        One of: 'isi', 'nlbias', 'awgn', 'rayleigh'.
    seed : int, optional

    Returns
    -------
    callable
        Channel function f(signal, snr_db) -> noisy_signal.
    """
    if kind == 'isi':
        H_ISI = np.array([1.0, 0.7, 0.5])
        return ISIChannel(H_ISI, seed=seed)
    elif kind == 'nlbias':
        return NonlinearBiasChannel(saturation=1.5, dc_bias=0.5, seed=seed)
    elif kind == 'awgn':
        # Simple wrapper around awgn_channel
        return awgn_channel
    elif kind == 'rayleigh':
        return RayleighChannel(seed=seed)
    else:
        raise ValueError(f"Unknown channel kind: {kind}")


def train_mlp(hop1_channel, seed=0, n_train=120_000, epochs=25, batch=256):
    """Train MLP relay on synthetic data.

    Parameters
    ----------
    hop1_channel : callable
        Channel function for hop 1.
    seed : int
        Seed for MLP initialization.
    n_train : int
        Total training samples.
    epochs : int
        Training epochs.
    batch : int
        Batch size.

    Returns
    -------
    mlp : MLPRelay
        Trained relay.
    n_params : int
        Number of parameters.
    """
    # Create MLP with windowed input
    mlp = MLPRelay(
        input_size=W,
        hidden_size=HID,
        output_size=1,
        window_size=W,
        seed=seed
    )

    # Collect training data
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []

    for snr_db in TRAIN_SNRS:
        # Generate random BPSK symbols
        bits = rng.integers(0, 2, per_snr)
        x = 1.0 - 2.0 * bits

        # Pass through channel
        y = hop1_channel(x, snr_db)

        # Extract windows (manually, since we're training)
        pad_size = W // 2
        yp = np.pad(y, (pad_size, pad_size), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(yp, W)

        X_list.append(windows)
        T_list.append(x)

    X = np.vstack(X_list)
    T = np.concatenate(T_list)

    # Train
    mlp.train_on_data(X, T, epochs=epochs, batch_size=batch, lr=3e-3)

    n_params = sum(p.size for p in mlp.params)
    return mlp, n_params


def run_ber_trial(relay, hop1_channel, hop2_channel, source, destination, num_bits, snr_db):
    """Run a single BER trial.

    Parameters
    ----------
    relay : Relay
        Relay strategy.
    hop1_channel : callable
        Hop 1 channel.
    hop2_channel : callable
        Hop 2 channel.
    source : Source
        Source node.
    destination : Destination
        Destination node.
    num_bits : int
        Number of bits to transmit.
    snr_db : float
        SNR in dB.

    Returns
    -------
    ber : float
        Bit Error Rate.
    """
    # Transmit
    tx_bits, tx_symbols = source.transmit(num_bits)

    # Hop 1
    rx_relay = hop1_channel(tx_symbols, snr_db)

    # Relay processing
    relay_out = relay.process(rx_relay)

    # Hop 2
    rx_dest = hop2_channel(relay_out, snr_db)

    # Receive
    rx_bits = destination.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)[0]


def run_experiment(hop1_kind, hop2_kind, mlp_relay):
    """Run full BER experiment.

    Parameters
    ----------
    hop1_kind : str
        Hop 1 channel type.
    hop2_kind : str
        Hop 2 channel type.
    mlp_relay : MLPRelay
        Trained MLP relay.

    Returns
    -------
    results : dict
        Dictionary with keys 'AF', 'DF', 'MLP', each containing
        (mean_ber, ci_ber) tuples.
    """
    # Create channels
    hop1_channel = create_channel(hop1_kind, seed=1)
    hop2_channel = create_channel(hop2_kind, seed=2)

    # Create nodes
    source = Source(seed=42, modulation='bpsk')
    destination = Destination(modulation='bpsk')

    # Create relays
    af_relay = AmplifyAndForwardRelay(target_power=1.0)
    df_relay = DecodeAndForwardRelay(target_power=1.0)

    # Run trials
    results = {r: np.zeros((len(SNRS), N_TRIALS)) for r in ('AF', 'DF', 'MLP')}

    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            # AF
            ber_af = run_ber_trial(af_relay, hop1_channel, hop2_channel, source, destination, N_BITS, snr)
            results['AF'][si, tr] = ber_af

            # DF
            ber_df = run_ber_trial(df_relay, hop1_channel, hop2_channel, source, destination, N_BITS, snr)
            results['DF'][si, tr] = ber_df

            # MLP
            ber_mlp = run_ber_trial(mlp_relay, hop1_channel, hop2_channel, source, destination, N_BITS, snr)
            results['MLP'][si, tr] = ber_mlp

            if tr == 0:
                print(f"  SNR {snr:2d} dB, trial {tr}: AF={ber_af:.4f}, DF={ber_df:.4f}, MLP={ber_mlp:.4f}")

    # Compute statistics
    return {
        r: (v.mean(1), 1.96 * v.std(1) / np.sqrt(N_TRIALS))
        for r, v in results.items()
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("E6_SIM: Unknown ISI & Nonlinear Bias Experiments (Ported to relaynet)")
    print("=" * 70)

    # Train MLPs once per channel type
    nets = {}
    for kind in ('isi', 'nlbias'):
        channel = create_channel(kind, seed=1)
        print(f"\nTraining MLP-170 for '{kind}'...")
        net, npar = train_mlp(channel, seed=1)
        nets[kind] = net
        print(f"  Trained: {npar} parameters")

    # Run experiments
    setups = [
        ('S1: unknown ISI -> AWGN',      'isi',      'awgn'),
        ('S2: unknown ISI -> Rayleigh',  'isi',      'rayleigh'),
        ('S3: nonlinear+bias -> AWGN',   'nlbias',   'awgn'),
        ('S4 control: Rayleigh -> Rayleigh (canonical)', 'rayleigh', 'rayleigh'),
    ]

    all_results = {}
    for name, hop1_kind, hop2_kind in setups:
        print(f"\n{name}")
        print(f"  SNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))

        # Choose or train MLP for this hop1 type
        if hop1_kind not in nets:
            channel = create_channel(hop1_kind, seed=1)
            net, npar = train_mlp(channel, seed=1)
            nets[hop1_kind] = net
            print(f"  Trained MLP-170: {npar} parameters")
        else:
            net = nets[hop1_kind]

        results = run_experiment(hop1_kind, hop2_kind, net)
        all_results[name] = results

        # Print results
        for relay in ('AF', 'DF', 'MLP'):
            mu, ci = results[relay]
            print(f"  {relay:>4}: " + " ".join(f"{m:7.4f}" for m in mu))

    # Save results
    output_path = '/tmp/e6_sim_ported_results.npy'
    np.save(output_path, {'setups': setups, 'results': all_results, 'snrs': SNRS}, allow_pickle=True)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    main()
