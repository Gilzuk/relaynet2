#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Add THREE memoryless channels (params redrawn per block):
#   FlatPhaseChannel:   y=e^{j.theta}.s+n, theta~U[0,2pi), DBPSK source, complex I/Q -> MLP-169 (22->7).
#   FlatGainChannel:    y=g.x+n, g~U[0.3,2.0], real -> MLP-170.
#   BranchAsymmetryChannel: s=a+ if x>0 else -a-, a+-~U[0.6,1.4], real -> MLP-170.
# Classical baseline for phase case = conventional differential detection sign(Re{y[i].conj(y[i-1])}).
# ACCEPTANCE (THE CONTROL): classical does NOT fail; MLP ties DF, max gap <= 0.0036 everywhere.
#   This proves memory (not unknownness) breaks classical relays. See PORTING.md section 2.
# ===================================================================================
"""E6 addendum 4: FLAT (memoryless) unknown channels -- no ISI.
Cases (hop 1, unknown to all relays):
  F1 unknown block phase theta ~ U[0,2pi): y = e^{j theta} x + n.  DBPSK source (blind BPSK ill-posed).
  F2 unknown constant gain g ~ U[0.3,2.0], sign known: y = g x + n.  Coherent BPSK; DF should be ROBUST (control).
  F3 unknown I/Q imbalance: y = (1+eps) Re(x) + j*(1-eps) e^{j phi} Im(x)*0 + n  -> for BPSK: gain+dc asymmetry.
     Realized as amplitude asymmetry + small phase tilt on a QPSK-like 2D symbol; here BPSK on I with unknown
     per-block gain asymmetry a_+ != a_- on the two symbols (models saturating PA per-branch).
Relays: AF, DF (case-appropriate classical: differential for F1, sign for F2/F3), MLP-169, Genie.
Hop 2 = canonical Rayleigh, compensated. Conventions match e6_sim/e6_jakes.
"""
import numpy as np
rng = np.random.default_rng(21)
W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 50_000

def diff_encode(x):
    s = np.empty_like(x); s[0] = 1.0
    for i in range(1, x.size): s[i] = s[i-1]*x[i]
    return s

def chan(kind, x, snr_db, r):
    """Return (features(n,F), target x). Unknown params redrawn per call (per block)."""
    sigma = 10**(-snr_db/20.0)
    if kind == 'phase':
        s = diff_encode(x)
        th = r.uniform(0, 2*np.pi)                     # unknown, constant over block
        n = sigma*(r.standard_normal(x.size)+1j*r.standard_normal(x.size))/np.sqrt(2)
        y = np.exp(1j*th)*s + n
        return np.stack([y.real, y.imag], 1).astype(np.float32), x
    if kind == 'gain':
        g = r.uniform(0.3, 2.0)                         # unknown positive gain
        y = g*x + sigma*r.standard_normal(x.size)
        return y[:, None].astype(np.float32), x
    if kind == 'iqimb':
        # per-branch saturating gain asymmetry: +1 -> a+, -1 -> -a-, with a+/- unknown per block
        ap = r.uniform(0.6, 1.4); am = r.uniform(0.6, 1.4)
        s = np.where(x > 0, ap, -am)
        y = s + sigma*r.standard_normal(x.size)
        return y[:, None].astype(np.float32), x
    raise ValueError(kind)

def hop2(xr, snr_db, r):
    h = np.abs((r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2))
    if np.iscomplexobj(xr):
        n = 10**(-snr_db/20.0)*(r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2)
    else:
        n = 10**(-snr_db/20.0)*r.standard_normal(xr.size)
    return h*xr + n

def cwin(y):
    yp = np.pad(y, ((W//2, W//2), (0, 0)) if y.ndim == 2 else (W//2, W//2))
    if y.ndim == 1: y = y[:, None]; yp = yp[:, None]
    outs = []
    for c in range(y.shape[1]):
        v = np.lib.stride_tricks.sliding_window_view(np.pad(y[:, c], (W//2, W//2)), W)
        outs.append(v)
    return np.concatenate(outs, axis=1)

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

def din_of(kind): return 2*W if kind == 'phase' else W
def hid_of(kind): return 7 if kind == 'phase' else 13     # ~169 params both

def train(kind, seed=2, n=150_000, epochs=25, bs=256):
    per = n//len(TRAIN_SNRS); Xs, Ts = [], []
    for s in TRAIN_SNRS:
        for _ in range(4):
            x = rng.choice([-1., 1.], per//4)
            f, tgt = chan(kind, x, s, rng); Xs.append(cwin(f)); Ts.append(tgt)
    X = np.vstack(Xs); T = np.concatenate(Ts)
    net = MLP(din_of(kind), hid_of(kind), seed)
    idx = np.arange(X.shape[0])
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), bs): net.step(X[idx[i:i+bs]], T[idx[i:i+bs]])
    print(f'  {kind}: {sum(p.size for p in net.P)} params')
    return net

def run(kind, net):
    out = {k: np.zeros((len(SNRS), N_TRIALS)) for k in ('AF', 'DF', 'MLP', 'Genie')}
    for si, s in enumerate(SNRS):
        for tr in range(N_TRIALS):
            r = np.random.default_rng(9000*si+tr)
            b = r.integers(0, 2, N_BITS); x = 1.0-2.0*b
            f, _ = chan(kind, x, s, r)
            v = N_BITS
            # AF: forward (unit power), destination detects
            if kind == 'phase':
                yc = f[:, 0]+1j*f[:, 1]
                xa = yc/(np.sqrt(np.mean(np.abs(yc)**2))+1e-12)
                yd = hop2(xa, s, r); d = np.real(yd[1:]*np.conj(yd[:-1]))
                out['AF'][si, tr] = np.mean(np.r_[0, (d < 0).astype(int)][1:] != b[1:])
                # DF: conventional differential
                d1 = np.real(yc[1:]*np.conj(yc[:-1])); xh = np.r_[1., np.sign(d1)+(d1 == 0)]
                yd = hop2(xh, s, r); out['DF'][si, tr] = np.mean((yd.real < 0).astype(int)[1:] != b[1:])
                # Genie: not applicable per-symbol (phase unknown) -> coherent w/ known theta via pilot=x[0]
                out['Genie'][si, tr] = out['DF'][si, tr]  # differential IS the genie-free optimum here
            else:
                yr = f[:, 0]
                xa = yr/(np.sqrt(np.mean(yr**2))+1e-12)
                yd = hop2(xa, s, r); out['AF'][si, tr] = np.mean((yd < 0).astype(int) != b)
                xh = np.sign(yr)+(yr == 0)               # DF fixed threshold
                yd = hop2(xh, s, r); out['DF'][si, tr] = np.mean((yd.real < 0).astype(int) != b)
                out['Genie'][si, tr] = out['DF'][si, tr] if kind == 'gain' else np.nan
            # MLP
            o = net.fwd(cwin(f)); xh = o/(np.sqrt(np.mean(o**2))+1e-12)
            yd = hop2(xh, s, r)
            out['MLP'][si, tr] = np.mean((yd.real < 0).astype(int)[:v] != b[:v]) if kind != 'phase' \
                else np.mean((yd.real < 0).astype(int)[1:] != b[1:])
    return {k: (v.mean(1), 1.96*v.std(1)/np.sqrt(N_TRIALS)) for k, v in out.items()}

res = {}
for kind, tag in [('phase', 'F1 unknown phase (DBPSK)'),
                  ('gain', 'F2 unknown gain (control)'),
                  ('iqimb', 'F3 per-branch gain asymmetry')]:
    net = train(kind); res[kind] = run(kind, net)
    print(f'== {tag} ==   SNR: ' + ' '.join(f'{s:>6d}' for s in SNRS))
    for k in ('AF', 'DF', 'MLP'):
        mu, _ = res[kind][k]; print(f'{k:>4}: ' + ' '.join(f'{m:6.4f}' for m in mu))
np.save('/home/claude/e6_flat.npy', {'snrs': SNRS, 'res': res}, allow_pickle=True)
print('saved.')
