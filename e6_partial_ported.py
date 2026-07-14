#!/usr/bin/env python3
"""E6_PARTIAL: partial-posterior pilot/block-length sweep -- ported to relaynet.

Composite channel (random ISI x PA x unknown phase per block, DBPSK), per
PORTING.md section 5. Classical pilot-aided Viterbi gets N_pilot symbols of
known preamble -> LS channel estimate -> MLSE. The family-trained MLP uses
ZERO pilots at test (amortized posterior) and CMA (blind) is the other
zero-pilot reference -- both are horizontal reference lines against the
pilot-count sweep.

Panel (a): pilot-count sweep {800,200,50,20,10,5} at a fixed operating SNR
  (10dB). Key target: Viterbi wins >=10 pilots, COLLAPSES at 5 pilots
  (identifiability floor -- only 2 degrees of freedom left for 3 unknowns'
  worth of noise averaging); MLP stays flat (no pilots needed at all).

Panel (b): block-length sweep {40,80,160,320,1000} with a FIXED 10-pilot
  preamble per block, same 10dB operating point. Overhead = 10/L varies
  from 25% (L=40) down to 1% (L=1000); MLP has zero overhead throughout.
  NOTE: panel (b)'s original standalone script was never present in this
  repo -- only its cached output (experiments-standalone/e6_blocklen.npy,
  Nmin=10, op=10.0dB) survived. Reconstructed here from that + PORTING.md's
  description, following the same channel/relay logic as panel (a).
"""

import numpy as np
from relaynet.channels import RandomISICompositeChannel, ComplexISIRayleighChannel
from relaynet.relays import MLPRelay

W = 11
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 6, 40_000
PILOTS = [800, 200, 50, 20, 10, 5]
BLOCK_LENGTHS = [40, 80, 160, 320, 1000]
N_MIN_PILOTS = 10
OP_SNR = 10.0

rng = np.random.default_rng(51)


def diff_encode(x):
    s = np.empty_like(x, dtype=float)
    s[0] = 1.0
    for i in range(1, x.size):
        s[i] = s[i - 1] * x[i]
    return s


def cwin(y, window=W):
    pad = window // 2
    yi = np.pad(y.real, (pad, pad))
    yq = np.pad(y.imag, (pad, pad))
    vi = np.lib.stride_tricks.sliding_window_view(yi, window)
    vq = np.lib.stride_tricks.sliding_window_view(yq, window)
    return np.concatenate([vi, vq], axis=1)


def ls_channel_estimate(y_pilot, s_pilot, taps=3):
    n = s_pilot.size
    X = np.stack([np.r_[np.zeros(k), s_pilot[:n - k]] for k in range(taps)], axis=1)
    h, *_ = np.linalg.lstsq(X, y_pilot[:n], rcond=None)
    return h


_STATES = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]


def viterbi_diff_decode(y, h):
    """4-state differential MLSE given a (possibly complex) 3-tap channel h."""
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
    s_rec = np.empty(n)
    for i in range(n - 1, -1, -1):
        s_rec[i] = 2.0 * bp_input[i, st] - 1.0
        st = bp_state[i, st]
    return np.r_[1.0, s_rec[1:] * s_rec[:-1]]


def cma_dfe(y, taps=7, mu=1e-3, iters=2):
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
            e = o * (np.abs(o) ** 2 - 1.0)
            w -= mu * e * np.conj(seg)
    return out


def train_mlp(hidden_size=7, seed=2, n_train=160_000, epochs=30, batch_size=256):
    per_snr = n_train // len(TRAIN_SNRS)
    X_list, T_list = [], []
    channel = RandomISICompositeChannel(pa_sat=1.2, seed=101)
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
    print(f"  MLP: {n_params} params")
    return mlp


def ber_viterbi_pilots(snr_db, n_pilots, n_trials, n_bits=N_BITS):
    """Panel (a): payload BER (pilot region excluded) at a fixed pilot count."""
    channel = RandomISICompositeChannel(pa_sat=1.2)
    hop2 = ComplexISIRayleighChannel(taps=np.array([1.0]))
    errs = np.zeros(n_trials)
    for tr in range(n_trials):
        r = np.random.default_rng(600 * n_pilots + tr)
        bits = r.integers(0, 2, n_bits)
        x = 1.0 - 2.0 * bits
        s = diff_encode(x).astype(complex)
        channel.rng = r
        y = channel(s, snr_db)

        s_pilot = diff_encode(x[:n_pilots])
        h_est = ls_channel_estimate(y[:n_pilots], s_pilot)
        x_hat = viterbi_diff_decode(y, h_est)

        hop2.rng = r
        y_dest = hop2(x_hat, snr_db)
        errs[tr] = np.mean((y_dest.real < 0).astype(int)[n_pilots:] != bits[n_pilots:])
    return errs.mean(), 1.96 * errs.std() / np.sqrt(n_trials)


def ber_viterbi_blocklen(snr_db, block_len, n_pilots, n_trials):
    """Panel (b): payload BER + overhead at a fixed pilot count, varying block length."""
    channel = RandomISICompositeChannel(pa_sat=1.2)
    hop2 = ComplexISIRayleighChannel(taps=np.array([1.0]))
    errs = np.zeros(n_trials)
    for tr in range(n_trials):
        r = np.random.default_rng(60_000 + 700 * block_len + tr)
        bits = r.integers(0, 2, block_len)
        x = 1.0 - 2.0 * bits
        s = diff_encode(x).astype(complex)
        channel.rng = r
        y = channel(s, snr_db)

        s_pilot = diff_encode(x[:n_pilots])
        h_est = ls_channel_estimate(y[:n_pilots], s_pilot)
        x_hat = viterbi_diff_decode(y, h_est)

        hop2.rng = r
        y_dest = hop2(x_hat, snr_db)
        errs[tr] = np.mean((y_dest.real < 0).astype(int)[n_pilots:] != bits[n_pilots:])
    overhead = n_pilots / block_len
    return errs.mean(), 1.96 * errs.std() / np.sqrt(n_trials), overhead


def ref_at(snr_db, mlp, n_trials, n_bits=N_BITS):
    """Zero-pilot flat references: MLP-169 and CMA-blind."""
    channel = RandomISICompositeChannel(pa_sat=1.2)
    hop2 = ComplexISIRayleighChannel(taps=np.array([1.0]))
    em = np.zeros(n_trials)
    ec = np.zeros(n_trials)
    for tr in range(n_trials):
        r = np.random.default_rng(999 + tr)
        bits = r.integers(0, 2, n_bits)
        x = 1.0 - 2.0 * bits
        s = diff_encode(x).astype(complex)
        channel.rng = r
        y = channel(s, snr_db)

        o = mlp.fwd(cwin(y))
        x_hat_mlp = o / (np.sqrt(np.mean(o ** 2)) + 1e-12)
        hop2.rng = r
        y_dest = hop2(x_hat_mlp, snr_db)
        em[tr] = np.mean((y_dest.real < 0).astype(int) != bits)

        eq = cma_dfe(y)
        d = np.real(eq[1:] * np.conj(eq[:-1]))
        x_hat_cma = np.r_[1.0, np.sign(d) + (d == 0)]
        hop2.rng = r
        y_dest = hop2(x_hat_cma, snr_db)
        ec[tr] = np.mean((y_dest.real < 0).astype(int)[1:] != bits[1:])

    mlp_ref = (em.mean(), 1.96 * em.std() / np.sqrt(n_trials))
    cma_ref = (ec.mean(), 1.96 * ec.std() / np.sqrt(n_trials))
    return mlp_ref, cma_ref


def main():
    print("=" * 80)
    print("E6_PARTIAL: pilot-count and block-length sweeps at fixed 10dB")
    print("=" * 80)

    print("\nTraining MLP-169...")
    mlp = train_mlp(hidden_size=7, seed=2)

    print(f"\n=== Panel (a): pilot-count sweep at {OP_SNR:.0f} dB (payload BER, pilots excluded) ===")
    mlp_ref, cma_ref = ref_at(OP_SNR, mlp, N_TRIALS)
    print(f"  MLP-169 (0 pilots): BER={mlp_ref[0]:.4f}  |  CMA blind (0 pilots): BER={cma_ref[0]:.4f}")

    panel_a = {}
    for n_pilots in PILOTS:
        mu, ci = ber_viterbi_pilots(OP_SNR, n_pilots, N_TRIALS)
        overhead = n_pilots / N_BITS
        panel_a[n_pilots] = (mu, ci)
        print(f"  Viterbi, {n_pilots:>4d} pilots: BER={mu:.4f} +/- {ci:.4f} "
              f"(overhead {100*overhead:.3f}% of {N_BITS}-bit block)")

    print(f"\n=== Panel (b): block-length sweep, fixed {N_MIN_PILOTS}-pilot preamble, {OP_SNR:.0f} dB ===")
    panel_b = {}
    for block_len in BLOCK_LENGTHS:
        mu, ci, overhead = ber_viterbi_blocklen(OP_SNR, block_len, N_MIN_PILOTS, N_TRIALS)
        panel_b[block_len] = (mu, ci, overhead)
        print(f"  L={block_len:>5d}: Viterbi payload BER={mu:.4f} +/- {ci:.4f}, "
              f"overhead={100*overhead:.2f}% (MLP overhead = 0% always)")

    output_path = '/tmp/e6_partial_ported_results.npy'
    np.save(output_path, {
        'op_snr': OP_SNR, 'pilots': PILOTS, 'panel_a': panel_a,
        'mlp_ref': mlp_ref, 'cma_ref': cma_ref,
        'block_lengths': BLOCK_LENGTHS, 'n_min_pilots': N_MIN_PILOTS, 'panel_b': panel_b,
        'n_bits': N_BITS,
    }, allow_pickle=True)
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("E6_PARTIAL: Complete")
    print("=" * 80)
    return panel_a, panel_b, mlp_ref, cma_ref


if __name__ == '__main__':
    main()
