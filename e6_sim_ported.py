#!/usr/bin/env python3
"""E6: Unknown-channel two-hop relay experiment — ported to relaynet.

Hop 1 = unknown channel (ISI / nonlinear-bias) or control (AWGN / Rayleigh).
Relays: AF, DF (sign), MLP-170 (window 11 -> 13 tanh -> 1 tanh).
Hop 2 = AWGN or coherently-compensated Rayleigh at the same SNR.

Ported to use relaynet's Channel/Relay/runner infrastructure.
SNR convention matches thesis: gamma = 1/sigma^2, single-hop AWGN BER = Q(sqrt(gamma)).

Following PORTING.md section 1 acceptance criteria.
"""

import os
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
N_TRIALS, N_BITS = 10, 100_000
N_TRAIN = 3   # independent training seeds; effective MC columns = N_TRAIN * N_TRIALS

# SNR-adaptive bit budget: at high SNR, MLP BER is very small and 100k bits
# yields too few errors for a reliable estimate.  Scale up so each trial
# contributes at least O(100) expected errors for the MLP relay.
# AF/DF plateau above 0.18 everywhere and are already over-sampled, so
# the larger bit count costs nothing in terms of result quality.
BITS_AT_SNR = {
    0:  100_000,
    2:  100_000,
    4:  100_000,
    6:  100_000,
    8:  100_000,
    10: 1_000_000,
    12: 10_000_000,
    14: 10_000_000,
}

# At 16 dB and above the MLP BER is effectively zero in any realistic
# block-length trial. Instead of pooling N_TRAIN*N_TRIALS small blocks,
# we run a first-error experiment: transmit blocks until the first bit
# error is found, then report BER = 1 / bits_until_first_error.
FIRST_ERROR_SNRS = {16, 18, 20}  # SNR values (dB) to use first-error estimator
# Per reviewer requirement: 18 dB and 20 dB must run to first error or
# timeout at 10G bits.
FIRST_ERROR_MAX_BITS_BY_SNR = {
    16: 1_000_000_000,
    18: 10_000_000_000,
    20: 10_000_000_000,
}
FIRST_ERROR_DEFAULT_MAX_BITS = 100_000_000
FIRST_ERROR_BLOCK = 100_000        # transmit in 100k-bit blocks for memory efficiency
# After the first error at N1 bits, keep transmitting to this multiple of N1
# so the estimate rests on ~10 errors rather than 1 (capped by max_bits).
FIRST_ERROR_EXTEND_FACTOR = 10

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


def run_ber_first_error(relay, hop1_channel, hop2_channel, source, destination,
                        snr_db, max_bits=FIRST_ERROR_DEFAULT_MAX_BITS,
                        block_size=FIRST_ERROR_BLOCK,
                        extend_factor=FIRST_ERROR_EXTEND_FACTOR):
    """Rare-event BER estimator: locate the first error, then keep going.

    Stopping at the first error and reporting 1/N is a one-sample estimator:
    the waiting time to a single event has a standard deviation equal to its
    own mean, so the figure carries no usable precision. It is also badly
    behaved when applied to a relay whose BER is not small -- an AF relay
    near 0.2 hits its first error after a couple of bits, and 1/2 is then
    reported as a BER.

    So the search runs in two phases. Phase one finds the first error at
    N1 bits. Phase two continues to `extend_factor * N1` bits (capped by
    max_bits), counting every error, and the estimate is the ordinary
    errors/bits ratio over the whole run. At the default factor of ten this
    accumulates about ten errors, giving roughly +/-30% relative precision
    instead of a single sample, and it costs almost nothing precisely where
    the old estimator was worst: a relay erring every few bits reaches its
    quota in a few dozen bits.

    Where the BER is genuinely tiny the cap binds before the quota is met.
    That case is reported honestly rather than papered over: `n_errors`
    tells the caller how many events the estimate rests on, so a figure
    resting on one or two errors can be labelled as such.

    Returns
    -------
    ber : float
        n_errors / bits_used, or the rule-of-three 95% upper bound 3/max_bits
        if no error was seen at all. (1/N, used previously for that case, is
        not an upper bound of any stated confidence -- it is roughly the
        point at which one error becomes likely, so it understates the true
        bound by about a factor of three.)
    bits_used : int
        Total bits transmitted.
    found_error : bool
        Whether at least one error was observed.
    n_errors : int
        Errors accumulated. 0 means `ber` is an upper bound, not an estimate.
    """
    bits_used = 0
    n_errors = 0
    target_bits = max_bits          # tightened to extend_factor*N1 once found

    while bits_used < target_bits:
        n = min(block_size, target_bits - bits_used)
        tx_bits, tx_symbols = source.transmit(n)
        rx_relay = hop1_channel(tx_symbols, snr_db)
        relay_out = relay.process(rx_relay)
        rx_dest = hop2_channel(relay_out, snr_db)
        rx_bits = destination.receive(rx_dest)

        mismatches = (tx_bits != rx_bits)
        errors = int(np.sum(mismatches))

        if n_errors == 0 and errors > 0:
            # First error located: fix the budget at extend_factor * N1.
            first_idx = int(np.argmax(mismatches))
            bits_to_first = bits_used + first_idx + 1
            target_bits = min(extend_factor * bits_to_first, max_bits)

        n_errors += errors
        bits_used += n

    if n_errors == 0:
        # Rule of three: with zero events in N trials the 95% upper bound on
        # the rate is ~3/N. Flagged by found_error=False and n_errors=0.
        return 3.0 / max_bits, max_bits, False, 0

    return n_errors / bits_used, bits_used, True, n_errors


def run_experiment(hop1_kind, hop2_kind, mlp_relays):
    """Run full BER experiment over multiple independently trained MLPs.

    Parameters
    ----------
    hop1_kind : str
        Hop 1 channel type.
    hop2_kind : str
        Hop 2 channel type.
    mlp_relays : list of MLPRelay
        N_TRAIN independently trained MLP relays.

    Returns
    -------
    results : dict
        Dictionary with keys 'AF', 'DF', 'MLP', each containing
        (mean_ber, ci_ber) tuples pooled over N_TRAIN * N_TRIALS columns.
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

    # Result arrays: (len(SNRS), N_TRAIN * N_TRIALS)
    # For FIRST_ERROR_SNRS we use only 1 seed, 1 trial, stored in col 0;
    # remaining cols are set to the same value so the mean is unchanged.
    total_cols = N_TRAIN * N_TRIALS
    results = {r: np.zeros((len(SNRS), total_cols)) for r in ('AF', 'DF', 'MLP')}
    # first_error_meta[si] = dict with bits_used / found_error for reporting
    first_error_meta = {}

    for ti, mlp_relay in enumerate(mlp_relays):
        col_offset = ti * N_TRIALS
        print(f"  [Training instance {ti + 1}/{N_TRAIN}]")
        for si, snr in enumerate(SNRS):
            if int(snr) in FIRST_ERROR_SNRS:
                # First-error estimator: only run on the first training seed
                if ti > 0:
                    # Copy seed-0 result into this seed's columns so pooled mean is stable
                    results['AF'][si, col_offset:col_offset + N_TRIALS] = results['AF'][si, 0]
                    results['DF'][si, col_offset:col_offset + N_TRIALS] = results['DF'][si, 0]
                    results['MLP'][si, col_offset:col_offset + N_TRIALS] = results['MLP'][si, 0]
                    continue
                # ti == 0: run the actual first-error experiment
                first_error_max_bits = FIRST_ERROR_MAX_BITS_BY_SNR.get(int(snr), FIRST_ERROR_DEFAULT_MAX_BITS)
                print(f"    SNR {snr:2d} dB  [first-error, up to {first_error_max_bits//1_000_000}M bits]")
                ber_af, bits_af, _, nerr_af = run_ber_first_error(af_relay, hop1_channel, hop2_channel, source, destination, snr,
                                                         max_bits=first_error_max_bits)
                ber_df, bits_df, _, nerr_df = run_ber_first_error(df_relay, hop1_channel, hop2_channel, source, destination, snr,
                                                         max_bits=first_error_max_bits)
                ber_mlp, bits_mlp, found, nerr_mlp = run_ber_first_error(mlp_relay, hop1_channel, hop2_channel, source, destination, snr,
                                                                max_bits=first_error_max_bits)
                flag = "" if found else f" [no error in {first_error_max_bits//1_000_000}M bits → 3/N bound]"
                print(f"      AF={ber_af:.2e} ({bits_af:,}b, {nerr_af} err), "
                      f"DF={ber_df:.2e} ({bits_df:,}b, {nerr_df} err), "
                      f"MLP={ber_mlp:.2e} ({bits_mlp:,}b, {nerr_mlp} err){flag}")
                # Fill all columns with this single estimate
                results['AF'][si, :] = ber_af
                results['DF'][si, :] = ber_df
                results['MLP'][si, :] = ber_mlp
                first_error_meta[si] = {
                    'snr': snr, 'bits_af': bits_af, 'bits_df': bits_df,
                    'bits_mlp': bits_mlp, 'found_error': found,
                    # Error counts are what tell a reader whether a cell is an
                    # estimate or a single event; carried through to the .npy.
                    'n_errors_af': nerr_af, 'n_errors_df': nerr_df,
                    'n_errors_mlp': nerr_mlp,
                    'extend_factor': FIRST_ERROR_EXTEND_FACTOR,
                }
            else:
                n_bits_snr = BITS_AT_SNR.get(int(snr), N_BITS)
                for tr in range(N_TRIALS):
                    col = col_offset + tr

                    ber_af = run_ber_trial(af_relay, hop1_channel, hop2_channel, source, destination, n_bits_snr, snr)
                    results['AF'][si, col] = ber_af

                    ber_df = run_ber_trial(df_relay, hop1_channel, hop2_channel, source, destination, n_bits_snr, snr)
                    results['DF'][si, col] = ber_df

                    ber_mlp = run_ber_trial(mlp_relay, hop1_channel, hop2_channel, source, destination, n_bits_snr, snr)
                    results['MLP'][si, col] = ber_mlp

                    if tr == 0:
                        print(f"    SNR {snr:2d} dB [{n_bits_snr//1000}k bits], trial {tr}: AF={ber_af:.4f}, DF={ber_df:.4f}, MLP={ber_mlp:.4f}")

    # Pool statistics over all N_TRAIN * N_TRIALS columns
    # (first-error SNRs have all cols identical → CI = 0, which is correct:
    #  a single first-error measurement has no within-experiment variance)
    return {
        r: (v.mean(1), 1.96 * v.std(1) / np.sqrt(total_cols))
        for r, v in results.items()
    }, first_error_meta


def _save(all_results, setups, complete, rare_event_meta):
    """Write results to the repository, flagging whether the run finished.

    Called after every setup as well as at the end, so a container restart
    costs one setup rather than the whole pass. This is the only writer of
    the .npy: an earlier version also did a bare np.save() at the end of
    main(), which silently dropped the `complete` and `setups_done` tags
    from the finished file.

    `rare_event_meta` carries the bits and error counts behind every 16-20 dB
    cell. Those counts are what tell a reader whether a cell is an estimate
    or a single event, and without them in the file the only record was the
    console log.
    """
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'e6_unknown_channel_results', 'e6_sim_ported_results.npy')
    np.save(out, {'setups': setups, 'results': all_results, 'snrs': SNRS,
                  'n_train': N_TRAIN, 'n_trials': N_TRIALS,
                  'bits_at_snr': BITS_AT_SNR,
                  'first_error_snrs': list(FIRST_ERROR_SNRS),
                  'first_error_max_bits_by_snr': FIRST_ERROR_MAX_BITS_BY_SNR,
                  'first_error_default_max_bits': FIRST_ERROR_DEFAULT_MAX_BITS,
                  'extend_factor': FIRST_ERROR_EXTEND_FACTOR,
                  'rare_event_meta': rare_event_meta,
                  'complete': complete,
                  'setups_done': sorted(all_results)}, allow_pickle=True)
    print(f"  [checkpoint] {len(all_results)}/{len(setups)} setups saved"
          f"{'' if complete else ' (partial)'}", flush=True)


def main():
    """Main entry point."""
    print("=" * 70)
    print("E6_SIM: Unknown ISI & Nonlinear Bias Experiments (Ported to relaynet)")
    print("=" * 70)

    # Train N_TRAIN independent MLPs per channel type
    nets = {}
    for kind in ('isi', 'nlbias'):
        channel = create_channel(kind, seed=1)
        print(f"\nTraining {N_TRAIN}x MLP-170 for '{kind}'...")
        trained = []
        for ti in range(N_TRAIN):
            net, npar = train_mlp(channel, seed=1 + ti)
            trained.append(net)
            print(f"  Seed {1 + ti}: {npar} parameters")
        nets[kind] = trained

    # Run experiments
    setups = [
        ('S1: unknown ISI -> AWGN',      'isi',      'awgn'),
        ('S2: unknown ISI -> Rayleigh',  'isi',      'rayleigh'),
        ('S3: nonlinear+bias -> AWGN',   'nlbias',   'awgn'),
        ('S4 control: Rayleigh -> Rayleigh (canonical)', 'rayleigh', 'rayleigh'),
    ]

    all_results = {}
    rare_event_meta = {}
    for name, hop1_kind, hop2_kind in setups:
        print(f"\n{name}")
        print(f"  SNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))

        # Choose or train MLPs for this hop1 type
        if hop1_kind not in nets:
            channel = create_channel(hop1_kind, seed=1)
            trained = []
            print(f"  Training {N_TRAIN}x MLP-170 for '{hop1_kind}'...")
            for ti in range(N_TRAIN):
                net, npar = train_mlp(channel, seed=1 + ti)
                trained.append(net)
                print(f"    Seed {1 + ti}: {npar} parameters")
            nets[hop1_kind] = trained
        trained_nets = nets[hop1_kind]

        results, fe_meta = run_experiment(hop1_kind, hop2_kind, trained_nets)
        all_results[name] = results
        rare_event_meta[name] = fe_meta

        # Print results
        for relay in ('AF', 'DF', 'MLP'):
            mu, ci = results[relay]
            print(f"  {relay:>4}: " + " ".join(f"{m:7.4f}" for m in mu))

        # Checkpoint after every setup. The 18 and 20 dB first-error searches
        # run to 10 billion bits, so a full pass takes hours; two container
        # restarts have already discarded a complete run that only saved at
        # the end. A partial file is marked so it is never mistaken for one.
        _save(all_results, setups, complete=False,
              rare_event_meta=rare_event_meta)

    # /tmp does not persist between sessions (CLAUDE.md); _save writes straight
    # into the repo, which is what keeps the committed data and the script in
    # step. Nothing else may write this file -- see the note in _save.
    _save(all_results, setups, complete=True, rare_event_meta=rare_event_meta)

    return all_results


if __name__ == '__main__':
    main()
