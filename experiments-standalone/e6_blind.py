#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Composite channel with RANDOM ISI per block, NO pilots exposed.
# ADD BASELINES: CMAEqualizer (constant-modulus, 7 taps, mu=1e-3, 2 passes) -> port cma_dfe verbatim.
#                decision-directed blind MLSE -> port blind_viterbi verbatim.
# ACCEPTANCE: CMA ~0.0024 @20dB; MLP ties ~0.0026; decision-directed blind MLSE UNSTABLE
#   (mid-SNR CI 0.164 vs MLP 0.014). The instability IS the finding -- if your port stabilizes
#   it, that changes the claim, so flag it. See PORTING.md section 4.
# ===================================================================================
"""E6 addendum 6: the TRULY posterior-free (blind) regime.
Composite channel (ISI x PA x unknown phase, DBPSK), but now NO pilots and NO channel prior available.
Question: when the receiver has no posterior/estimate of the channel, can any *classical* method work,
and how does the (offline-trained, but channel-blind at test) MLP compare?

Classical options WITHOUT a per-block channel estimate:
  AF                 : forward as-is.
  DF-diff            : conventional differential detection (needs no channel, but blind to ISI/PA).
  CMA-DFE            : blind adaptive linear equalizer (constant-modulus algorithm) + differential decode.
                       This is the canonical *blind* classical equalizer -- no pilots, self-adapting.
  Viterbi-perblock   : MLSE that must ESTIMATE the channel from THIS block's data with NO pilots,
                       via blind LS on hard differential decisions (decision-directed, bootstrapped).

Learned:
  MLP-169            : offline-trained on the channel FAMILY (random ISI+PA+phase draws), but at test sees
                       a NEW unseen channel realization with no adaptation -- amortized inference, not per-block CSI.

The distinction the experiment makes precise:
  - Viterbi-with-pilots (previous study) has a per-block posterior -> ~2 dB better than MLP.
  - Blind classical (CMA, decision-directed) has NO posterior and must bootstrap online per block.
  - MLP has no per-block posterior either; it amortizes over the family at training time.
Hop 2 = canonical Rayleigh. Conventions match prior E6 scripts.
"""
import numpy as np
rng = np.random.default_rng(41)
W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 5, 40_000

def rand_isi(r):
    h = np.array([1.0, r.uniform(0.3, 0.7), r.uniform(0.2, 0.5)])
    return h/np.linalg.norm(h)

def diff_encode(x):
    s = np.empty_like(x); s[0] = 1.0
    for i in range(1, x.size): s[i] = s[i-1]*x[i]
    return s

def pa(z, sat=1.2):
    a = np.abs(z); return (a/(1+(a/sat)**2)**0.5)*np.exp(1j*np.angle(z))

def composite(x, snr_db, r, fixed_h=None):
    s = diff_encode(x).astype(complex)
    h = fixed_h if fixed_h is not None else rand_isi(r)
    s = np.convolve(s, h)[:x.size]
    s = pa(s)
    s = np.exp(1j*r.uniform(0, 2*np.pi))*s
    n = 10**(-snr_db/20.0)*(r.standard_normal(x.size)+1j*r.standard_normal(x.size))/np.sqrt(2)
    return s + n

def hop2(xr, snr_db, r):
    h = np.abs((r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2))
    n = 10**(-snr_db/20.0)*(r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2)
    return h*xr + n

def cwin(y):
    yi = np.pad(y.real,(W//2,W//2)); yq = np.pad(y.imag,(W//2,W//2))
    return np.concatenate([np.lib.stride_tricks.sliding_window_view(yi,W),
                           np.lib.stride_tricks.sliding_window_view(yq,W)],1)

class MLP:
    def __init__(s,din,hid,seed):
        r=np.random.default_rng(seed)
        s.W1=r.standard_normal((din,hid))*np.sqrt(2/din);s.b1=np.zeros(hid)
        s.W2=r.standard_normal((hid,1))*np.sqrt(2/hid);s.b2=np.zeros(1)
        s.P=[s.W1,s.b1,s.W2,s.b2];s.m=[np.zeros_like(p) for p in s.P];s.v=[np.zeros_like(p) for p in s.P];s.t=0
    def fwd(s,X): s.h=np.tanh(X@s.W1+s.b1);return np.tanh(s.h@s.W2+s.b2).ravel()
    def step(s,X,t,lr=3e-3):
        o=s.fwd(X);B=X.shape[0];do=2*(o-t)/B*(1-o**2)
        g=[X.T@(do[:,None]@s.W2.T*(1-s.h**2)),(do[:,None]@s.W2.T*(1-s.h**2)).sum(0),s.h.T@do[:,None],np.array([do.sum()])]
        s.t+=1
        for p,gi,m,v in zip(s.P,g,s.m,s.v):
            m[:]=.9*m+.1*gi;v[:]=.999*v+.001*gi**2
            p-=lr*(m/(1-.9**s.t))/(np.sqrt(v/(1-.999**s.t))+1e-8)

def train(hid=7,seed=2,n=160_000,epochs=30,bs=256):
    per=n//len(TRAIN_SNRS);Xs,Ts=[],[]
    for s in TRAIN_SNRS:
        for _ in range(4):
            x=rng.choice([-1.,1.],per//4)
            Xs.append(cwin(composite(x,s,rng)));Ts.append(x)   # random channel each draw
    X=np.vstack(Xs);T=np.concatenate(Ts);net=MLP(2*W,hid,seed);idx=np.arange(X.shape[0])
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0,len(idx),bs): net.step(X[idx[i:i+bs]],T[idx[i:i+bs]])
    print(f'  MLP: {sum(p.size for p in net.P)} params (trained on the channel FAMILY, blind at test)')
    return net

def cma_dfe(y, taps=7, mu=1e-3, iters=2):
    """Blind constant-modulus linear equalizer (no pilots). Returns equalized complex stream."""
    n=y.size; w=np.zeros(taps,complex); w[taps//2]=1.0
    yp=np.r_[np.zeros(taps//2,complex),y,np.zeros(taps//2,complex)]
    out=np.zeros(n,complex)
    for _ in range(iters):
        for i in range(n):
            seg=yp[i:i+taps][::-1]
            o=np.vdot(w,seg); out[i]=o
            e=o*(np.abs(o)**2-1.0)                 # CMA (R2=1 for unit-modulus)
            w-=mu*e*np.conj(seg)
    return out

def blind_viterbi(y, taps=3, rounds=3):
    """Decision-directed blind MLSE: bootstrap channel from hard differential decisions, no pilots."""
    n=y.size
    # init decisions from plain differential detection
    d=np.real(y[1:]*np.conj(y[:-1])); sh=np.r_[1.,np.sign(d)+(d==0)]
    for _ in range(rounds):
        s_hat=diff_encode(sh)                       # re-encode current estimate
        # LS channel estimate from (s_hat -> y)
        Xmat=np.stack([np.r_[np.zeros(k),s_hat[:n-k]] for k in range(taps)],1)
        h,*_=np.linalg.lstsq(Xmat,y,rcond=None)
        # one MLSE pass (reuse trellis)
        STATES=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]
        exp=np.array([[h[0]*u+h[1]*b+h[2]*a for u in(-1.,1.)] for(a,b)in STATES])
        nxt=np.array([[STATES.index((b,u)) for u in(-1.,1.)] for(a,b)in STATES])
        met=np.zeros(4);bps=np.empty((n,4),np.int8);bpi=np.empty((n,4),np.int8)
        for i in range(n):
            cand=met[:,None]+np.abs(y[i]-exp)**2
            new=np.full(4,np.inf);bs=np.zeros(4,np.int8);bi=np.zeros(4,np.int8)
            for st in range(4):
                for u in range(2):
                    ns=nxt[st,u]
                    if cand[st,u]<new[ns]:new[ns]=cand[st,u];bs[ns]=st;bi[ns]=u
            met=new;bps[i]=bs;bpi[i]=bi
        st=int(np.argmin(met));srec=np.empty(n)
        for i in range(n-1,-1,-1): srec[i]=2.*bpi[i,st]-1.;st=bps[i,st]
        sh=np.r_[1.,srec[1:]*srec[:-1]]
    return sh

def run(net):
    keys=('DF-diff','CMA-blind','Viterbi-blind','MLP-169')
    out={k:np.zeros((len(SNRS),N_TRIALS)) for k in keys}
    for si,s in enumerate(SNRS):
        for tr in range(N_TRIALS):
            r=np.random.default_rng(8000*si+tr)
            b=r.integers(0,2,N_BITS);x=1.-2.*b
            y=composite(x,s,r)                       # NEW random channel, no pilots exposed
            # DF-diff
            d=np.real(y[1:]*np.conj(y[:-1]));xh=np.r_[1.,np.sign(d)+(d==0)]
            yd=hop2(xh,s,r);out['DF-diff'][si,tr]=np.mean((yd.real<0)[1:]!=b[1:])
            # CMA blind linear eq + differential
            eq=cma_dfe(y);d=np.real(eq[1:]*np.conj(eq[:-1]));xh=np.r_[1.,np.sign(d)+(d==0)]
            yd=hop2(xh,s,r);out['CMA-blind'][si,tr]=np.mean((yd.real<0)[1:]!=b[1:])
            # decision-directed blind Viterbi
            xh=blind_viterbi(y)
            yd=hop2(xh,s,r);out['Viterbi-blind'][si,tr]=np.mean((yd.real<0)[1:]!=b[1:])
            # MLP (blind at test)
            o=net.fwd(cwin(y));xh=o/(np.sqrt(np.mean(o**2))+1e-12)
            yd=hop2(xh,s,r);out['MLP-169'][si,tr]=np.mean((yd.real<0)!=b)
    return {k:(v.mean(1),1.96*v.std(1)/np.sqrt(N_TRIALS)) for k,v in out.items()}

print('BLIND composite: random ISI x PA x unknown phase per block, NO pilots, NO channel prior')
net=train()
res=run(net)
print('SNR: '+' '.join(f'{s:>6d}' for s in SNRS))
for k in ('DF-diff','CMA-blind','Viterbi-blind','MLP-169'):
    mu,_=res[k];print(f'{k:>14}: '+' '.join(f'{m:6.4f}' for m in mu))
np.save('/home/claude/e6_blind.npy',{'snrs':SNRS,'res':res},allow_pickle=True)
print('saved.')
