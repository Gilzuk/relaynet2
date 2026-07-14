#!/usr/bin/env python3
"""E6_COMPOSITE: Composite (mixed) unknown channel -- ported to relaynet.

Hop 1 cascade (all unknown to the relay), following PORTING.md section 3:
    x --(diff-encode)--> s --(3-tap ISI)--> --(Rapp PA nonlinearity)-->
        --(unknown block phase)--> + noise

This mixes three impairments no single classical relay is matched to: memory
(ISI), amplitude nonlinearity (PA), and unknown phase (needs noncoherent
detection). Baselines are each the *best* classical option for ONE
sub-impairment, so the comparison is honest:
    AF            : forward as-is (matched to nothing)
    DF-diff       : conventional differential detector (matched to phase,
                    blind to ISI/PA)
    Viterbi-diff  : pilot-LS complex-channel MLSE then differential decode
                    (matched to ISI+phase via a complex channel estimate,
                    blind to PA nonlinearity)
    MLP-169/large : learned relay on raw I/Q window, no impairment knowledge

Hop 2 = canonical Rayleigh, compensated (relaynet's AdaptiveRayleighChannel).
Uses relaynet's CompositeChannel (ISI -> PA -> phase -> AWGN) for hop 1.

Acceptance targets (PORTING.md): AF/DF floored ~0.25; MLP-169 -> ~5e-3 @20dB;
Viterbi-diff ~2dB ahead of MLP at low-mid SNR; MLP-large =~ MLP-169 (H3: more
params don't help once the task is this hard).
"""

import numpy as np
from relaynet.relays import AmplifyAndForwardRelay, MLPRelay
from relaynet.channels import CompositeChannel, AdaptiveRayleighChannel
from relaynet.modulation import calculate_ber

W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 40_000  # standalone's own dev budget (see PORTING.md conventions table)
H_ISI = np.array([1.0, 0.6, 0.4])
H_ISI = H_ISI / np.linalg.norm(H_ISI)

rng = np.random.default_rng(31)


def diff_encode(x):
    """DBPSK differential encoding: running product."""
    s = np.empty_like(x, dtype=float)
    s[0] = 1.0
    for i in range(1, x.size):
        s[i] = s[i - 1] * x[i]
    return s


def cwin(y, window=W):
    """Extract concatenated I/Q sliding windows from a complex signal."""
    pad = window // 2
    yi = np.pad(y.real, (pad, pad))
    yq = np.pad(y.imag, (pad, pad))
    vi = np.lib.stride_tricks.sliding_window_view(yi, window)
    vq = np.lib.stride_tricks.sliding_window_view(yq, window)
    return np.concatenate([vi, vq], axis=1)


def ls_channel_estimate(y_pilot, s_pilot, taps=3):
    """LS estimate of a (possibly complex-valued) 3-tap channel from pilots."""
    n = s_pilot.size
    X = np.stack([np.r_[np.zeros(k), s_pilot[:n - k]] for k in range(taps)], axis=1)
    h, *_ = np.linalg.lstsq(X, y_pilot, rcond=None)
    return h


class ViterbiDiffCompositeRelay:
    """Pilot-LS complex-channel MLSE + differential decode.

    States are the last two differential symbols; branch metric uses a
    (possibly complex) 3-tap channel estimate obtained via LS from pilots,
    which can absorb a constant per-block phase rotation into the estimate.
    Blind to the PA nonlinearity (a linear channel model can't capture it).
    Ported verbatim from experiments-standalone/e6_composite.py's
    viterbi_diff()/ls_channel(), matching PORTING.md's "port verbatim" note.
    """

    STATES = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]

    def __init__(self, n_pilots=200):
        self.n_pilots = n_pilots

    def decode_with_pilots(self, y, x_pilot):
        """Decode y given the first `n_pilots` transmitted BPSK bits x_pilot
        (known a priori to both sides, e.g. a shared preamble convention)."""
        s_pilot = diff_encode(x_pilot)
        h = ls_channel_estimate(y[:self.n_pilots], s_pilot)
        return self._decode(y, h)

    def _decode(self, y, h):
        n = y.size
        exp_y = np.array([[h[0] * u + h[1] * b + h[2] * a for u in (-1., 1.)]
                           for (a, b) in self.STATES])
        nxt = np.array([[self.STATES.index((b, u)) for u in (-1., 1.)]
                         for (a, b) in self.STATES])

        metric = np.zeros(4)
        bp_state = np.empty((n, 4), dtype=np.int8)
        bp_input = np.empty((n, 4), dtype=np.int8)

        for i in range(n):
            cand = metric[:, None] + np.abs(y[i] - exp_y) ** 2
            new_metric = np.full(4, np.inf)
            bs = np.zeros(4, dtype=np.int8)
            bi = np.zeros(4, dtype=np.int8)
            for st in range(4):
                for u in range(2):
                    ns = nxt[st, u]
                    if cand[st, u] < new_metric[ns]:
                        new_metric[ns] = cand[st, u]
                        bs[ns] = st
                        bi[ns] = u
            metric = new_metric
            bp_state[i] = bs
            bp_input[i] = bi

        st = int(np.argmin(metric))
        s_hat = np.empty(n)
        for i in range(n - 1, -1, -1):
            s_hat[i] = 2.0 * bp_input[i, st] - 1.0
            st = bp_state[i, st]

        # Differential decode of the recovered (diff-encoded) symbol sequence.
        return np.r_[1.0, s_hat[1:] * s_hat[:-1]]


def train_mlp(channel, hidden_size, seed=2, n_train=160_000, epochs=30, batch_size=256):
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []
    for snr_db in TRAIN_SNRS:
        for _ in range(4):
            bits = rng.integers(0, 2, per_snr // 4)
            x = 1.0 - 2.0 * bits
            s = diff_encode(x).astype(complex)
            y = channel(s, snr_db)
            X_list.append(cwin(y))
            T_list.append(x)
    X = np.vstack(X_list)
    T = np.concatenate(T_list)

    mlp = MLPRelay(input_size=2 * W, hidden_size=hidden_size, output_size=1,
                    window_size=None, seed=seed)
    mlp.train_on_data(X, T, epochs=epochs, batch_size=batch_size, lr=3e-3)
    n_params = sum(p.size for p in mlp.params)
    print(f"  MLP hid={hidden_size}: {n_params} params")
    return mlp


def run_ber_trial(relay_name, relay, channel_h1, channel_h2, num_bits, snr_db, seed):
    r = np.random.default_rng(seed)
    bits = r.integers(0, 2, num_bits)
    x = 1.0 - 2.0 * bits
    s = diff_encode(x).astype(complex)

    channel_h1.rng = r
    y = channel_h1(s, snr_db)

    if relay_name == 'AF':
        relay_out = relay.process(y)
        channel_h2.rng = r
        y_dest = channel_h2(relay_out, snr_db)
        d = np.real(y_dest[1:] * np.conj(y_dest[:-1]))
        bit_out = np.r_[0, (d < 0).astype(int)]
        return np.mean(bit_out[1:] != bits[1:])

    elif relay_name == 'DF-diff':
        d1 = np.real(y[1:] * np.conj(y[:-1]))
        x_hat = np.r_[1.0, np.sign(d1) + (d1 == 0)]
        channel_h2.rng = r
        y_dest = channel_h2(x_hat, snr_db)
        bit_out = (y_dest.real < 0).astype(int)
        return np.mean(bit_out[1:] != bits[1:])

    elif relay_name == 'Viterbi-diff':
        x_pilot = x[:relay.n_pilots]  # first n_pilots symbols of THIS trial's actual transmission
        x_hat = relay.decode_with_pilots(y, x_pilot)
        channel_h2.rng = r
        y_dest = channel_h2(x_hat, snr_db)
        bit_out = (y_dest.real < 0).astype(int)
        return np.mean(bit_out[1:] != bits[1:])

    else:  # MLP-169, MLP-large
        o = relay.fwd(cwin(y))
        x_hat = o / (np.sqrt(np.mean(o ** 2)) + 1e-12)
        channel_h2.rng = r
        y_dest = channel_h2(x_hat, snr_db)
        bit_out = (y_dest.real < 0).astype(int)
        return np.mean(bit_out != bits)


def main():
    print("=" * 80)
    print("E6_COMPOSITE: ISI x PA-nonlinearity x unknown-phase, DBPSK source")
    print("Ported to relaynet: CompositeChannel (hop1) + AdaptiveRayleighChannel (hop2)")
    print("=" * 80)

    channel_h1 = CompositeChannel(isi_taps=H_ISI, pa_sat=1.2, include_phase=True, seed=1)
    channel_h2 = AdaptiveRayleighChannel(seed=2)

    print("\nTraining MLPs...")
    mlp_small = train_mlp(CompositeChannel(isi_taps=H_ISI, pa_sat=1.2, include_phase=True, seed=3),
                           hidden_size=7, seed=2)   # ~169 params: 2*11*7+7 + 7+1
    mlp_large = train_mlp(CompositeChannel(isi_taps=H_ISI, pa_sat=1.2, include_phase=True, seed=4),
                           hidden_size=48, seed=3)  # larger

    af_relay = AmplifyAndForwardRelay(target_power=1.0)
    viterbi_relay = ViterbiDiffCompositeRelay(n_pilots=200)

    relays = {
        'AF': af_relay,
        'DF-diff': None,
        'Viterbi-diff': viterbi_relay,
        'MLP-169': mlp_small,
        'MLP-large': mlp_large,
    }

    results = {name: np.zeros((len(SNRS), N_TRIALS)) for name in relays}
    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 7000 * si + tr
            for name, relay in relays.items():
                ber = run_ber_trial(name, relay, channel_h1, channel_h2, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber

    summary = {}
    for name in relays:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>13}: " + " ".join(f"{m:7.4f}" for m in mu))

    output_path = '/tmp/e6_composite_ported_results.npy'
    np.save(output_path, {'snrs': SNRS, 'summary': summary}, allow_pickle=True)
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("E6_COMPOSITE: Complete")
    print("=" * 80)
    return summary


if __name__ == '__main__':
    main()
