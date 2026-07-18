#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Add CompositeChannel(parts=['isi','pa','phase']):
#   DBPSK -> 3-tap ISI [1,0.6,0.4]/norm -> Rapp PA pa(z,sat=1.2) -> unknown phase -> noise.
# pa(z): g=a/(1+(a/sat)^2)^0.5, preserves angle -> port verbatim.
# BASELINES: AF, DF-diff, pilot-LS Viterbi+differential (matched to ISI+phase, blind to PA),
#            MLP-169 and MLP-1153 (H3 check).
# ACCEPTANCE: AF/DF floored 0.25; MLP-169 -> 5e-3 @20dB; Viterbi ~2 dB ahead; MLP-1153 ~= MLP-169.
#   See PORTING.md section 3.
# ===================================================================================
"""E6 addendum 5: COMPOSITE (mixed) unknown channel -- a cascade no single classical relay is matched to.
Hop-1 signal path (applied in order, all unknown to the relay):
   x --(diff-encode)--> s --(3-tap ISI)--> --(saturating PA nonlinearity)--> --(x block phase theta)--> + noise
This mixes: memory (ISI), amplitude nonlinearity (PA), and unknown phase (needs noncoherent).
Baselines, each the BEST classical option for ONE sub-impairment (so the comparison is honest):
   AF                : forward as-is
   DF-diff           : conventional differential detector (matched to phase, blind to ISI/PA)
   Viterbi-diff      : pilot-LS linear-channel MLSE then differential decode (matched to ISI+phase, blind to PA nonlinearity)
   MLP-169 / MLP-large: learned relay on I/Q window (raw), no impairment knowledge
   Genie             : perfect phase + channel, coherent MLSE-ish bound
Hop 2 = canonical Rayleigh, compensated. Conventions match prior E6 scripts.
"""
import numpy as np
rng = np.random.default_rng(31)
W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 40_000
H_ISI = np.array([1.0, 0.6, 0.4]); H_ISI /= np.linalg.norm(H_ISI)

def diff_encode(x):
    s = np.empty_like(x); s[0] = 1.0
    for i in range(1, x.size): s[i] = s[i-1]*x[i]
    return s

def pa(z, sat=1.2):
    """Soft-limiter PA (Rapp-like on the magnitude), memoryless nonlinearity."""
    a = np.abs(z); ang = np.angle(z)
    g = a / (1 + (a/sat)**2)**0.5
    return g*np.exp(1j*ang)

def composite(x, snr_db, r, parts=('isi','pa','phase')):
    s = diff_encode(x).astype(complex)
    if 'isi' in parts:
        s = np.convolve(s, H_ISI)[:x.size]
    if 'pa' in parts:
        s = pa(s)
    if 'phase' in parts:
        s = np.exp(1j*r.uniform(0, 2*np.pi))*s
    sigma = 10**(-snr_db/20.0)
    n = sigma*(r.standard_normal(x.size)+1j*r.standard_normal(x.size))/np.sqrt(2)
    return s + n

def hop2(xr, snr_db, r):
    h = np.abs((r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2))
    n = 10**(-snr_db/20.0)*(r.standard_normal(xr.size) + (1j*r.standard_normal(xr.size) if np.iscomplexobj(xr) else 0))/(np.sqrt(2) if np.iscomplexobj(xr) else 1)
    return h*xr + n

def cwin(y):
    yi = np.pad(y.real, (W//2, W//2)); yq = np.pad(y.imag, (W//2, W//2))
    vi = np.lib.stride_tricks.sliding_window_view(yi, W)
    vq = np.lib.stride_tricks.sliding_window_view(yq, W)
    return np.concatenate([vi, vq], 1)

class MLP:
    def __init__(self, din, hid, seed):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((din, hid))*np.sqrt(2/din); self.b1 = np.zeros(hid)
        self.W2 = r.standard_normal((hid, 1))*np.sqrt(2/hid);   self.b2 = np.zeros(1)
        self.P = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.P]; self.v = [np.zeros_like(p) for p in self.P]; self.t = 0
    def fwd(self, X):
        self.h = np.tanh(X@self.W1+self.b1); return np.tanh(self.h@self.W2+self.b2).ravel()
    def step(self, X, tgt, lr=3e-3):
        o = self.fwd(X); B = X.shape[0]; do = 2*(o-tgt)/B*(1-o**2)
        g = [X.T@(do[:, None]@self.W2.T*(1-self.h**2)), (do[:, None]@self.W2.T*(1-self.h**2)).sum(0),
             self.h.T@do[:, None], np.array([do.sum()])]
        self.t += 1
        for p, gi, m, v in zip(self.P, g, self.m, self.v):
            m[:] = .9*m+.1*gi; v[:] = .999*v+.001*gi**2
            p -= lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)

def train(hid, seed=2, n=160_000, epochs=30, bs=256):
    per = n//len(TRAIN_SNRS); Xs, Ts = [], []
    for s in TRAIN_SNRS:
        for _ in range(4):
            x = rng.choice([-1., 1.], per//4)
            Xs.append(cwin(composite(x, s, rng))); Ts.append(x)
    X = np.vstack(Xs); T = np.concatenate(Ts)
    net = MLP(2*W, hid, seed); idx = np.arange(X.shape[0])
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), bs): net.step(X[idx[i:i+bs]], T[idx[i:i+bs]])
    print(f'  MLP hid={hid}: {sum(p.size for p in net.P)} params')
    return net

def ls_channel(yp, sp, taps=3):
    n = sp.size
    X = np.stack([np.r_[np.zeros(k), sp[:n-k]] for k in range(taps)], 1)
    h, *_ = np.linalg.lstsq(X, yp, rcond=None)
    return h

def viterbi_diff(y, h):
    """MLSE on complex y with complex FIR h (len 3), states over last-2 differential symbols, then diff-decode."""
    n = y.size; L = len(h)
    STATES = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]
    # branch: state (a,b) input u -> expected diff-encoded sample uses running product; approximate by
    # treating transmitted s as +-1 stream (differential handled post-hoc). Expected y = h0*c + h1*b + h2*a
    exp = np.array([[h[0]*u + h[1]*b + h[2]*a for u in (-1., 1.)] for (a, b) in STATES])
    nxt = np.array([[STATES.index((b, u)) for u in (-1., 1.)] for (a, b) in STATES])
    met = np.zeros(4); bps = np.empty((n, 4), np.int8); bpi = np.empty((n, 4), np.int8)
    for i in range(n):
        cand = met[:, None] + np.abs(y[i] - exp)**2
        new = np.full(4, np.inf); bs = np.zeros(4, np.int8); bi = np.zeros(4, np.int8)
        for st in range(4):
            for u in range(2):
                ns = nxt[st, u]
                if cand[st, u] < new[ns]: new[ns] = cand[st, u]; bs[ns] = st; bi[ns] = u
        met = new; bps[i] = bs; bpi[i] = bi
    st = int(np.argmin(met)); sh = np.empty(n)
    for i in range(n-1, -1, -1):
        sh[i] = 2.*bpi[i, st]-1.; st = bps[i, st]
    xh = np.r_[1., sh[1:]*sh[:-1]]                 # differential decode of recovered s
    return xh

def run(net_s, net_l):
    keys = ('AF', 'DF-diff', 'Viterbi-diff', 'MLP-169', 'MLP-large')
    out = {k: np.zeros((len(SNRS), N_TRIALS)) for k in keys}
    for si, s in enumerate(SNRS):
        for tr in range(N_TRIALS):
            r = np.random.default_rng(7000*si+tr)
            b = r.integers(0, 2, N_BITS); x = 1.-2.*b
            y = composite(x, s, r)
            # AF
            xa = y/(np.sqrt(np.mean(np.abs(y)**2))+1e-12)
            yd = hop2(xa, s, r); d = np.real(yd[1:]*np.conj(yd[:-1]))
            out['AF'][si, tr] = np.mean(np.r_[0, (d < 0)][1:] != b[1:])
            # DF-diff
            d1 = np.real(y[1:]*np.conj(y[:-1])); xh = np.r_[1., np.sign(d1)+(d1 == 0)]
            yd = hop2(xh, s, r); out['DF-diff'][si, tr] = np.mean((yd.real < 0)[1:] != b[1:])
            # Viterbi-diff with pilot LS (200 pilots): estimate |channel| ignoring phase via differential-referenced
            NP = 200; xp = x[:NP]; sp = diff_encode(xp)
            h_est = ls_channel(y[:NP], sp)
            xh = viterbi_diff(y, h_est)
            yd = hop2(xh, s, r); out['Viterbi-diff'][si, tr] = np.mean((yd.real < 0)[1:] != b[1:])
            # MLPs
            for name, net in (('MLP-169', net_s), ('MLP-large', net_l)):
                o = net.fwd(cwin(y)); xh = o/(np.sqrt(np.mean(o**2))+1e-12)
                yd = hop2(xh, s, r); out[name][si, tr] = np.mean((yd.real < 0) != b)
    return {k: (v.mean(1), 1.96*v.std(1)/np.sqrt(N_TRIALS)) for k, v in out.items()}

print('Composite channel = ISI(3-tap) x PA-nonlinearity x unknown-phase, DBPSK source')
net_s = train(7)        # 2*11*7+7 + 7+1 = 169
net_l = train(48)       # larger
res = run(net_s, net_l)
print('SNR: ' + ' '.join(f'{s:>6d}' for s in SNRS))
for k in ('AF', 'DF-diff', 'Viterbi-diff', 'MLP-169', 'MLP-large'):
    mu, _ = res[k]; print(f'{k:>13}: ' + ' '.join(f'{m:6.4f}' for m in mu))
np.save('/home/claude/e6_composite.npy', {'snrs': SNRS, 'res': res}, allow_pickle=True)
print('saved.')
