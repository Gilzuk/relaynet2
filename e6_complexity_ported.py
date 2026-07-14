#!/usr/bin/env python3
"""E6_COMPLEXITY: Viterbi MLSE vs the 169-parameter MLP relay -- ported to relaynet.

Per PORTING.md section 6, this experiment is mostly analytical: Viterbi
per-symbol flops grow as M^L (states = M^(L-1), branches = M^L), while the
MLP relay's per-symbol cost is constant in M and L. No BER here -- this
quantifies the cost side of the accuracy/cost trade-off.

Panel (a): analytical operation-count formula, evaluated at the same
  (M, L) grid as the standalone script.
Panel (b): measured wall-clock inference time, using relaynet's ACTUAL
  Viterbi (ViterbiMLSERelay.process(), 4-state BPSK/L=3) and ACTUAL MLP
  (MLPRelay.fwd(), 169 params) implementations -- not hand-rolled
  reimplementations -- per PORTING.md's reconciliation note ("re-measure
  with relaynet's actual Viterbi implementation").

Honest caveat preserved from the standalone script: at BPSK/L=3, Viterbi
is CHEAPER per-flop than the MLP (64 vs ~330 flops/symbol); the MLP wins
on M^L scaling (constant vs exponential in M and L) and on wall-clock
(vectorized numpy matmul vs Python-loop sequential scan). The measured
wall-clock ratio is implementation-specific to this codebase, not a
universal claim -- reported honestly as numpy-MLP vs Python-Viterbi.
"""

import time
import numpy as np

from relaynet.relays import MLPRelay
from relaynet.relays.viterbi import ViterbiMLSERelay

W = 11
rng = np.random.default_rng(0)


def viterbi_ops(M, L):
    """MLSE per-symbol cost: states = M^(L-1), branches/state = M.

    Add-compare-select over M^L branch transitions per symbol; each
    branch metric costs ~8 flops (1 complex subtract + magnitude-square
    [~4 real mul/add] + 1 add + compare).
    """
    states = M ** (L - 1)
    branches = states * M  # = M^L
    flops_per_branch = 8
    return states, branches, branches * flops_per_branch


def mlp_ops(mlp):
    """MLP per-symbol cost read directly off a real MLPRelay instance."""
    din, hid = mlp.W1.shape
    macs = din * hid + hid * mlp.output_size
    flops = 2 * macs + hid + mlp.output_size  # mul+add per MAC, + activations
    params = sum(p.size for p in mlp.params)
    return params, macs, flops


def make_signal(n, h):
    x = rng.choice([-1., 1.], n).astype(complex)
    y = np.convolve(x, h)[:n] + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return y


def cwin(y, window=W):
    """I/Q sliding windows, matching MLPRelay's training-time input format."""
    pad = window // 2
    yi = np.pad(y.real, (pad, pad))
    yq = np.pad(y.imag, (pad, pad))
    vi = np.lib.stride_tricks.sliding_window_view(yi, window)
    vq = np.lib.stride_tricks.sliding_window_view(yq, window)
    return np.concatenate([vi, vq], axis=1)


def main():
    print("=" * 80)
    print("E6_COMPLEXITY: Viterbi MLSE vs MLP-169 relay -- analytical cost + wall-clock")
    print("=" * 80)

    mlp = MLPRelay(input_size=2 * W, hidden_size=7, output_size=1, window_size=None, seed=0)
    _, _, mlp_flops = mlp_ops(mlp)
    mlp_params = sum(p.size for p in mlp.params)

    print("\n=== Panel (a): analytical per-symbol cost ===")
    print(f'{"M":>3} {"L":>3} {"Viterbi states":>15} {"Vit branches/sym":>17} '
          f'{"Vit flops/sym":>14} {"MLP flops/sym":>14}')
    grid = [(2, 3), (2, 5), (2, 7), (4, 3), (4, 5), (16, 3), (16, 5)]
    panel_a = {}
    for M, L in grid:
        st, br, vf = viterbi_ops(M, L)
        panel_a[(M, L)] = (st, br, vf)
        print(f'{M:>3} {L:>3} {st:>15,} {br:>17,} {vf:>14,} {mlp_flops:>14,}')
    print(f"\nMLP-{mlp_params} is CONSTANT in M and L: {mlp_flops} flops/symbol, {mlp_params} params.")
    print("Viterbi grows as M^L per symbol (states M^(L-1) x M branches); intractable for large M or L.")
    print(f"Honest caveat: at BPSK/L=3, Viterbi ({viterbi_ops(2, 3)[2]} flops/sym) is CHEAPER "
          f"per-flop than the MLP ({mlp_flops} flops/sym); the MLP wins on scaling, not per-symbol cost here.")

    print("\n=== Panel (b): measured wall-clock (relaynet's actual Viterbi vs actual MLP) ===")
    h = np.array([1., 0.6, 0.4])
    h = h / np.linalg.norm(h)
    viterbi = ViterbiMLSERelay(channel_taps=h, channel_len=3)

    print(f'{"length":>8} {"Viterbi (ms)":>13} {"MLP (ms)":>10} {"speedup":>8}')
    panel_b = {}
    for n in [1000, 5000, 20000, 50000]:
        y = make_signal(n, h)

        t0 = time.perf_counter()
        for _ in range(3):
            viterbi.process(y.real)
        tv = (time.perf_counter() - t0) / 3 * 1000

        t0 = time.perf_counter()
        for _ in range(3):
            mlp.fwd(cwin(y))
        tm = (time.perf_counter() - t0) / 3 * 1000

        panel_b[n] = (tv, tm, tv / tm)
        print(f'{n:>8} {tv:>13.2f} {tm:>10.2f} {tv/tm:>7.1f}x')

    print("\nNote: MLP forward pass is a vectorized batched matmul; relaynet's Viterbi decoder is an")
    print("inherently sequential Python-loop scan (add-compare-select per symbol, per state).")
    print("The wall-clock ratio above is numpy-MLP vs Python-Viterbi in THIS codebase -- implementation")
    print("specific, not a universal claim. The M^L vs constant scaling in panel (a) is the robust claim.")

    output_path = '/tmp/e6_complexity_ported_results.npy'
    np.save(output_path, {
        'panel_a_grid': grid, 'panel_a': panel_a,
        'mlp_flops': mlp_flops, 'mlp_params': mlp_params,
        'panel_b': panel_b,
    }, allow_pickle=True)
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("E6_COMPLEXITY: Complete")
    print("=" * 80)
    return panel_a, panel_b


if __name__ == '__main__':
    main()
