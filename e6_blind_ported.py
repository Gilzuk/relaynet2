#!/usr/bin/env python3
"""E6_BLIND: the truly posterior-free (blind) regime -- ported to relaynet.

Composite channel (ISI x PA x unknown phase, DBPSK) as in E6_COMPOSITE, but
now with RANDOM ISI drawn fresh per block and NO pilots / no channel prior
available at all, per PORTING.md section 4.

Classical options WITHOUT a per-block channel estimate:
    AF          : forward as-is.
    DF-diff     : conventional differential detection (blind to ISI/PA, needs
                  no channel knowledge to begin with).
    CMA-blind   : blind adaptive linear equalizer (constant-modulus
                  algorithm), no pilots, self-adapting -- the canonical
                  classical blind equalizer.
    Viterbi-blind: MLSE that must ESTIMATE the channel from THIS block's data
                  with NO pilots, via decision-directed bootstrap LS.

Learned:
    MLP-169     : trained offline on the channel FAMILY (random ISI/PA/phase
                  draws), but at test time sees a NEW unseen realization with
                  no adaptation -- amortized inference, not per-block CSI.

Acceptance (PORTING.md): CMA ~0.0024 @20dB; MLP ties ~0.0026; Viterbi-blind
is UNSTABLE (mid-SNR CI ~0.164 vs MLP's ~0.014) -- the instability itself is
the finding, not a bug to fix.
"""

import numpy as np
from relaynet.channels import RandomISICompositeChannel, ComplexISIRayleighChannel
from relaynet.relays import MLPRelay

W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 40_000

rng = np.random.default_rng(41)


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


def cma_dfe(y, taps=7, mu=1e-3, iters=2):
    """Blind constant-modulus linear equalizer (no pilots).

    Ported verbatim from experiments-standalone/e6_blind.py's cma_dfe().
    Returns the equalized complex stream.
    """
    n = y.size
    w = np.zeros(taps, dtype=complex)
    w[taps // 2] = 1.0
    yp = np.r_[np.zeros(taps // 2, dtype=complex), y, np.zeros(taps // 2, dtype=complex)]
    out = np.zeros(n, dtype=complex)
    for _ in range(iters):
        for i in range(n):
            seg = yp[i:i + taps][::-1]
            o = np.vdot(w, seg)
            out[i] = o
            e = o * (np.abs(o) ** 2 - 1.0)  # CMA (R2=1 for unit-modulus)
            w -= mu * e * np.conj(seg)
    return out


_STATES = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]


def _mlse_pass(y, h):
    """One 4-state differential MLSE pass given a (possibly complex) channel h."""
    n = y.size
    exp_y = np.array([[h[0] * u + h[1] * b + h[2] * a for u in (-1., 1.)]
                       for (a, b) in _STATES])
    nxt = np.array([[_STATES.index((b, u)) for u in (-1., 1.)] for (a, b) in _STATES])

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
    return s_hat


def blind_viterbi(y, taps=3, rounds=3):
    """Decision-directed blind MLSE: bootstrap channel from hard differential
    decisions, no pilots. Ported verbatim from e6_blind.py's blind_viterbi().
    """
    n = y.size
    d = np.real(y[1:] * np.conj(y[:-1]))
    s_hat_bits = np.r_[1.0, np.sign(d) + (d == 0)]  # initial differential detection
    for _ in range(rounds):
        s_reencoded = diff_encode(s_hat_bits)
        X = np.stack([np.r_[np.zeros(k), s_reencoded[:n - k]] for k in range(taps)], axis=1)
        h, *_ = np.linalg.lstsq(X, y, rcond=None)
        s_rec = _mlse_pass(y, h)
        s_hat_bits = np.r_[1.0, s_rec[1:] * s_rec[:-1]]
    return s_hat_bits


def train_mlp(hidden_size=7, seed=2, n_train=160_000, epochs=30, batch_size=256):
    """Train on the channel FAMILY: a fresh random ISI + PA + phase draw per batch."""
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []
    channel = RandomISICompositeChannel(pa_sat=1.2, seed=99)
    for snr_db in TRAIN_SNRS:
        for _ in range(4):
            bits = rng.integers(0, 2, per_snr // 4)
            x = 1.0 - 2.0 * bits
            s = diff_encode(x).astype(complex)
            y = channel(s, snr_db)  # new random channel drawn each call
            X_list.append(cwin(y))
            T_list.append(x)
    X = np.vstack(X_list)
    T = np.concatenate(T_list)

    mlp = MLPRelay(input_size=2 * W, hidden_size=hidden_size, output_size=1,
                    window_size=None, seed=seed)
    mlp.train_on_data(X, T, epochs=epochs, batch_size=batch_size, lr=3e-3)
    n_params = sum(p.size for p in mlp.params)
    print(f"  MLP: {n_params} params (trained on the channel FAMILY, blind at test)")
    return mlp


def run_ber_trial(name, mlp, channel, hop2, num_bits, snr_db, seed):
    r = np.random.default_rng(seed)
    bits = r.integers(0, 2, num_bits)
    x = 1.0 - 2.0 * bits
    s = diff_encode(x).astype(complex)

    channel.rng = r
    y = channel(s, snr_db)  # NEW random channel realization, no pilots exposed

    hop2.rng = r
    if name == 'DF-diff':
        d = np.real(y[1:] * np.conj(y[:-1]))
        x_hat = np.r_[1.0, np.sign(d) + (d == 0)]
        y_dest = hop2(x_hat, snr_db)
        return np.mean((y_dest.real < 0).astype(int)[1:] != bits[1:])

    elif name == 'CMA-blind':
        eq = cma_dfe(y)
        d = np.real(eq[1:] * np.conj(eq[:-1]))
        x_hat = np.r_[1.0, np.sign(d) + (d == 0)]
        y_dest = hop2(x_hat, snr_db)
        return np.mean((y_dest.real < 0).astype(int)[1:] != bits[1:])

    elif name == 'Viterbi-blind':
        x_hat = blind_viterbi(y)
        y_dest = hop2(x_hat, snr_db)
        return np.mean((y_dest.real < 0).astype(int)[1:] != bits[1:])

    else:  # MLP-169
        o = mlp.fwd(cwin(y))
        x_hat = o / (np.sqrt(np.mean(o ** 2)) + 1e-12)
        y_dest = hop2(x_hat, snr_db)
        return np.mean((y_dest.real < 0).astype(int) != bits)


def main():
    print("=" * 80)
    print("E6_BLIND: random ISI x PA x unknown phase per block, NO pilots, NO channel prior")
    print("Ported to relaynet: RandomISICompositeChannel (hop1) + ComplexISIRayleighChannel (hop2)")
    print("=" * 80)

    channel = RandomISICompositeChannel(pa_sat=1.2, seed=1)
    hop2 = ComplexISIRayleighChannel(taps=np.array([1.0]), seed=2)  # trivial taps = no ISI, always-complex AWGN

    print("\nTraining MLP-169...")
    mlp = train_mlp(hidden_size=7, seed=2)

    keys = ('DF-diff', 'CMA-blind', 'Viterbi-blind', 'MLP-169')
    results = {k: np.zeros((len(SNRS), N_TRIALS)) for k in keys}

    print(f"\nSNR (dB): " + " ".join(f"{s:>7d}" for s in SNRS))
    for si, snr in enumerate(SNRS):
        for tr in range(N_TRIALS):
            seed_base = 8000 * si + tr
            for name in keys:
                ber = run_ber_trial(name, mlp, channel, hop2, N_BITS, snr, seed=seed_base)
                results[name][si, tr] = ber

    summary = {}
    for name in keys:
        mu = results[name].mean(axis=1)
        ci = 1.96 * results[name].std(axis=1) / np.sqrt(N_TRIALS)
        summary[name] = (mu, ci)
        print(f"  {name:>14}: " + " ".join(f"{m:7.4f}" for m in mu))

    print("\n  95% CI at mid-SNR (10dB), the reported instability check:")
    mid_idx = list(SNRS).index(10)
    for name in keys:
        mu, ci = summary[name]
        print(f"    {name:>14}: {mu[mid_idx]:.4f} +/- {ci[mid_idx]:.4f}")

    output_path = '/tmp/e6_blind_ported_results.npy'
    np.save(output_path, {'snrs': SNRS, 'summary': summary}, allow_pickle=True)
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("E6_BLIND: Complete")
    print("=" * 80)
    return summary


if __name__ == '__main__':
    main()
