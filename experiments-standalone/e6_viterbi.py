#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Add ViterbiMLSE(taps=3) detector for 4-state BPSK trellis.
# TWO VARIANTS: genie-CSI (true channel) and 200-pilot LS estimate.
# The viterbi() add-compare-select body is standard -> port verbatim into relaynet detector.
# ACCEPTANCE: genie Viterbi ~1-1.5 dB ahead of MLP-170 on unknown ISI. See PORTING.md sec 1.
# ===================================================================================
"""E6 addendum: Viterbi (MLSE) relay baselines on the unknown ISI channel.
- Viterbi-genie: MLSE with perfect channel knowledge (upper baseline / bound).
- Viterbi-est:   200 pilot symbols -> LS channel estimate -> MLSE (practical classical).
Same conventions as e6_sim.py. Compares against stored AF/DF/MLP results.
"""
import numpy as np

rng = np.random.default_rng(43)
H_ISI = np.array([1.0, 0.7, 0.5]); H_ISI = H_ISI / np.linalg.norm(H_ISI)
L = len(H_ISI)                       # channel memory+1 = 3 -> 4 states
SNRS = np.arange(0, 21, 2)
N_TRIALS, N_BITS, N_PILOT = 5, 50_000, 200

STATES = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]   # (x[i-2], x[i-1])
# transition: state s=(a,b), input u -> next state (b,u); expected y = h0*u + h1*b + h2*a

def viterbi(y, h):
    """MLSE over y with FIR channel h (len 3). Returns hard symbol decisions."""
    n = y.size
    h0, h1, h2 = h
    # precompute expected outputs for (state, input): 4x2
    exp_y = np.array([[h0 * u + h1 * b + h2 * a for u in (-1., 1.)] for (a, b) in STATES])
    nxt = np.array([[STATES.index((b, u)) for u in (-1., 1.)] for (a, b) in STATES])
    metric = np.zeros(4)
    bp_state = np.empty((n, 4), dtype=np.int8)   # best previous state
    bp_in    = np.empty((n, 4), dtype=np.int8)   # input on best branch
    for i in range(n):
        cand = metric[:, None] + (y[i] - exp_y) ** 2       # 4x2 candidate metrics
        new = np.full(4, np.inf); bs = np.zeros(4, np.int8); bi = np.zeros(4, np.int8)
        for s in range(4):
            for u in range(2):
                ns = nxt[s, u]
                if cand[s, u] < new[ns]:
                    new[ns] = cand[s, u]; bs[ns] = s; bi[ns] = u
        metric = new; bp_state[i] = bs; bp_in[i] = bi
    # traceback
    s = int(np.argmin(metric))
    xh = np.empty(n)
    for i in range(n - 1, -1, -1):
        xh[i] = 2.0 * bp_in[i, s] - 1.0
        s = bp_state[i, s]
    return xh

def ls_estimate(y_p, x_p):
    """LS estimate of 3-tap channel from pilots."""
    n = x_p.size
    X = np.stack([x_p, np.r_[0.0, x_p[:-1]], np.r_[0.0, 0.0, x_p[:-2]]], axis=1)
    h, *_ = np.linalg.lstsq(X, y_p[:n], rcond=None)
    return h

def hop1_isi(x, snr_db, r):
    sigma = 10 ** (-snr_db / 20.0)
    return np.convolve(x, H_ISI)[:x.size] + sigma * r.standard_normal(x.size)

def hop2(xr, snr_db, kind, r):
    sigma = 10 ** (-snr_db / 20.0)
    n = sigma * r.standard_normal(xr.size)
    if kind == 'awgn':
        return xr + n
    hh = np.abs((r.standard_normal(xr.size) + 1j * r.standard_normal(xr.size)) / np.sqrt(2))
    return hh * xr + n

def run(kind2):
    out = {r: np.zeros((len(SNRS), N_TRIALS)) for r in ('VIT-genie', 'VIT-est')}
    for si, s in enumerate(SNRS):
        for tr in range(N_TRIALS):
            r = np.random.default_rng(1000 * si + tr)
            b = r.integers(0, 2, N_BITS); x = 1.0 - 2.0 * b
            yr = hop1_isi(x, s, r)
            # pilot-based channel estimate from the first N_PILOT symbols
            h_est = ls_estimate(yr[:N_PILOT], x[:N_PILOT])
            for name, h in (('VIT-genie', H_ISI), ('VIT-est', h_est)):
                xr = viterbi(yr, h)
                yd = hop2(xr, s, kind2, r)
                bh = (yd < 0).astype(int)
                out[name][si, tr] = np.mean(bh != b)
    return {k: v.mean(1) for k, v in out.items()}

for kind2, tag in (('awgn', 'S1: unknown ISI -> AWGN'), ('rayleigh', 'S2: unknown ISI -> Rayleigh')):
    res = run(kind2)
    print(f'\n== {tag} ==   SNR: ' + ' '.join(f'{s:>7d}' for s in SNRS))
    for k, mu in res.items():
        print(f'{k:>9}: ' + ' '.join(f'{m:7.4f}' for m in mu))
    np.save(f'/home/claude/e6_viterbi_{kind2}.npy', res, allow_pickle=True)
print('\nsaved.')
