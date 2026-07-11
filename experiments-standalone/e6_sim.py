#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet framework. Replace standalone infra with relaynet classes.
# CHANNELS TO ADD: UnknownISIChannel(taps=[1.0,0.7,0.5], normalize=True)  [hop 1]
#                  reuse relaynet AWGN + coherently-compensated Rayleigh for hop 2.
# RELAYS: AF and symbol-wise DF already exist in relaynet -> reuse.
#         MLP-170: relaynet MLP relay, real input, window=11 -> hidden=13 -> 1, tanh/tanh.
# SNR CONVENTION: gamma=1/sigma^2 (sigma=10^(-snr/20)); AWGN single-hop BER=Q(sqrt(gamma)).
#                 VERIFY relaynet matches before porting (else rescale SNR axis).
# TRAIN: SNRs [5,10,15] dB, MSE(relay_out, x). Final run: raise to 10x100k trials.
# ACCEPTANCE: DF/AF pinned at analytic 0.25 ISI floor (DF non-monotonic);
#             MLP-170 < 5e-5 @16dB (AWGN hop2). See PORTING.md section 1.
# ===================================================================================
"""E6: Unknown-channel two-hop relay experiment.
Hop 1 = unknown channel (ISI / nonlinear-bias) or control (AWGN / Rayleigh).
Relays: AF, DF (sign), MLP-170 params (window 11 -> 13 tanh -> 1 tanh).
Hop 2 = AWGN or coherently-compensated Rayleigh at the same SNR.
Noise convention matches thesis: gamma = 1/sigma^2, single-hop AWGN BER = Q(sqrt(gamma)).
"""
import numpy as np

rng = np.random.default_rng(42)
W = 11          # window length
HID = 13        # hidden units -> params = 11*13+13 + 13*1+1 = 170
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000

# ---------------- channels (hop 1) ----------------
H_ISI = np.array([1.0, 0.7, 0.5]); H_ISI = H_ISI / np.linalg.norm(H_ISI)

def hop1(x, snr_db, kind):
    sigma = 10 ** (-snr_db / 20.0)
    n = sigma * rng.standard_normal(x.size)
    if kind == 'isi':          # unknown 3-tap ISI, unit energy
        s = np.convolve(x, H_ISI)[:x.size]
        return s + n
    if kind == 'nlbias':       # unknown saturating amp with DC bias
        return np.tanh(1.5 * x) + 0.5 + n
    if kind == 'awgn':
        return x + n
    if kind == 'rayleigh':     # coherently compensated: y = |h| x + n
        h = np.abs((rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2))
        return h * x + n
    raise ValueError(kind)

def hop2(xr, snr_db, kind):
    sigma = 10 ** (-snr_db / 20.0)
    n = sigma * rng.standard_normal(xr.size)
    if kind == 'awgn':
        return xr + n
    if kind == 'rayleigh':
        h = np.abs((rng.standard_normal(xr.size) + 1j * rng.standard_normal(xr.size)) / np.sqrt(2))
        return h * xr + n
    raise ValueError(kind)

# ---------------- relays ----------------
def relay_af(y):
    p = np.sqrt(np.mean(y ** 2)) + 1e-12
    return y / p

def relay_df(y):
    return np.sign(y) + (y == 0)

def windows(y):
    yp = np.pad(y, (W // 2, W // 2))
    return np.lib.stride_tricks.sliding_window_view(yp, W)

class MLP:
    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((W, HID)) * np.sqrt(2.0 / W)
        self.b1 = np.zeros(HID)
        self.W2 = r.standard_normal((HID, 1)) * np.sqrt(2.0 / HID)
        self.b2 = np.zeros(1)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
    def fwd(self, X):
        self.h = np.tanh(X @ self.W1 + self.b1)
        self.o = np.tanh(self.h @ self.W2 + self.b2).ravel()
        return self.o
    def step(self, X, tgt, lr=3e-3):
        o = self.fwd(X); B = X.shape[0]
        do = 2 * (o - tgt) / B * (1 - o ** 2)          # MSE + tanh'
        gW2 = self.h.T @ do[:, None]; gb2 = do.sum(0, keepdims=True).ravel()
        dh = do[:, None] @ self.W2.T * (1 - self.h ** 2)
        gW1 = X.T @ dh; gb1 = dh.sum(0)
        self.t += 1
        for p, g, m, v in zip(self.params, [gW1, gb1, gW2, gb2], self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            mh = m / (1 - 0.9 ** self.t); vh = v / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return np.mean((o - tgt) ** 2)

def train_mlp(kind, seed=0, n_train=120_000, epochs=25, batch=256):
    per = n_train // len(TRAIN_SNRS)
    Xs, Ts = [], []
    for s in TRAIN_SNRS:
        x = rng.choice([-1.0, 1.0], per)
        Xs.append(windows(hop1(x, s, kind))); Ts.append(x)
    X = np.vstack(Xs); T = np.concatenate(Ts)
    net = MLP(seed)
    idx = np.arange(X.shape[0])
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), batch):
            net.step(X[idx[i:i + batch]], T[idx[i:i + batch]])
    n_params = sum(p.size for p in net.params)
    return net, n_params

def relay_mlp(net, y):
    out = net.fwd(windows(y))
    p = np.sqrt(np.mean(out ** 2)) + 1e-12
    return out / p                                     # unit tx power

# ---------------- BER runner ----------------
def run(kind1, kind2, net):
    res = {r: np.zeros((len(SNRS), N_TRIALS)) for r in ('AF', 'DF', 'MLP')}
    for si, s in enumerate(SNRS):
        for tr in range(N_TRIALS):
            b = rng.integers(0, 2, N_BITS); x = 1.0 - 2.0 * b
            yr = hop1(x, s, kind1)
            for name, xr in (('AF', relay_af(yr)), ('DF', relay_df(yr)), ('MLP', relay_mlp(net, yr))):
                yd = hop2(xr, s, kind2)
                bh = (yd < 0).astype(int)
                res[name][si, tr] = np.mean(bh != b)
    return {r: (v.mean(1), 1.96 * v.std(1) / np.sqrt(N_TRIALS)) for r, v in res.items()}

setups = [
    ('S1: unknown ISI -> AWGN',      'isi',      'awgn'),
    ('S2: unknown ISI -> Rayleigh',  'isi',      'rayleigh'),
    ('S3: nonlinear+bias -> AWGN',   'nlbias',   'awgn'),
    ('S4 control: Rayleigh -> Rayleigh (canonical)', 'rayleigh', 'rayleigh'),
]
nets = {}
for _, k1, _ in setups:
    if k1 not in nets:
        nets[k1], npar = train_mlp(k1, seed=1)
        print(f'trained MLP for {k1}: {npar} params')

results = {}
for name, k1, k2 in setups:
    results[name] = run(k1, k2, nets[k1])
    print(f'\n== {name} ==   SNR: ' + ' '.join(f'{s:>7d}' for s in SNRS))
    for r in ('AF', 'DF', 'MLP'):
        mu, ci = results[name][r]
        print(f'{r:>4}: ' + ' '.join(f'{m:7.4f}' for m in mu))

np.save('/home/claude/e6_results.npy', {'setups': setups, 'results': results, 'snrs': SNRS}, allow_pickle=True)
print('\nsaved.')
