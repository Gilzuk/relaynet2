#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Reuses composite channel + Viterbi + CMA + MLP.
# PANEL (a): pilot count sweep {800,200,50,20,10,5} @10dB. Viterbi vs pilots; MLP/CMA flat refs.
#   KEY: Viterbi wins >=10 pilots, COLLAPSES at 5 (0.119) = identifiability; MLP flat 0.045.
# PANEL (b): block-length sweep {40..1000}, 10 pilots/block -> overhead 25% @L=40 vs 1% @L=1000.
# RECONCILIATION RISK: 5-pilot collapse depends on LS rcond regularization; reproduce the cliff,
#   report your pipeline's exact crossover. See PORTING.md section 5.
# ===================================================================================
"""E6 addendum 7: PARTIAL posterior -- sweep the pilot budget.
Composite channel (random ISI x PA x unknown phase per block, DBPSK).
Classical pilot-aided Viterbi gets N_pilot in {800,200,50,20,10,5,0} symbols -> LS channel estimate -> MLSE.
  N_pilot large  -> near-full posterior (matched receiver, ~2 dB ahead of MLP).
  N_pilot small  -> estimate is noisy -> MLSE degrades.
  N_pilot = 0    -> falls back to decision-directed blind (unstable).
The family-trained MLP uses ZERO pilots at test (amortized posterior), so it is a horizontal reference line.
CMA (blind, no pilots) is the other horizontal reference.
Question: at what pilot budget does the pilot-aided classical receiver stop beating the pilot-free MLP?
Measured at a fixed operating SNR (10 dB) and also full curves at N_pilot in {200,20,5}.
Also reports the *pilot overhead*: pilots are not payload, so effective rate = (L-N_pilot)/L.
"""
import numpy as np
rng = np.random.default_rng(51)
W = 11
SNRS = np.arange(0, 21, 2)
TRAIN_SNRS = [5, 10, 15]
N_TRIALS, N_BITS = 6, 40_000
PILOTS = [800, 200, 50, 20, 10, 5]
OP_SNR = 10.0

def rand_isi(r):
    h = np.array([1.0, r.uniform(0.3, 0.7), r.uniform(0.2, 0.5)]); return h/np.linalg.norm(h)
def diff_encode(x):
    s = np.empty_like(x); s[0]=1.
    for i in range(1,x.size): s[i]=s[i-1]*x[i]
    return s
def pa(z,sat=1.2):
    a=np.abs(z); return (a/(1+(a/sat)**2)**0.5)*np.exp(1j*np.angle(z))
def composite(x,snr,r):
    s=diff_encode(x).astype(complex); h=rand_isi(r)
    s=np.convolve(s,h)[:x.size]; s=pa(s); s=np.exp(1j*r.uniform(0,2*np.pi))*s
    n=10**(-snr/20.)*(r.standard_normal(x.size)+1j*r.standard_normal(x.size))/np.sqrt(2)
    return s+n, h
def hop2(xr,snr,r):
    h=np.abs((r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2))
    return h*xr+10**(-snr/20.)*(r.standard_normal(xr.size)+1j*r.standard_normal(xr.size))/np.sqrt(2)
def cwin(y):
    yi=np.pad(y.real,(W//2,W//2));yq=np.pad(y.imag,(W//2,W//2))
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
    per=n//3;Xs,Ts=[],[]
    for sn in TRAIN_SNRS:
        for _ in range(4):
            x=rng.choice([-1.,1.],per//4);y,_=composite(x,sn,rng);Xs.append(cwin(y));Ts.append(x)
    X=np.vstack(Xs);T=np.concatenate(Ts);net=MLP(2*W,hid,seed);idx=np.arange(X.shape[0])
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0,len(idx),bs): net.step(X[idx[i:i+bs]],T[idx[i:i+bs]])
    print(f'  MLP: {sum(p.size for p in net.P)} params');return net

def ls_ch(y,s,taps=3):
    n=s.size;X=np.stack([np.r_[np.zeros(k),s[:n-k]] for k in range(taps)],1)
    h,*_=np.linalg.lstsq(X,y[:n],rcond=None);return h
def viterbi(y,h):
    n=y.size;ST=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]
    exp=np.array([[h[0]*u+h[1]*b+h[2]*a for u in(-1.,1.)] for(a,b)in ST])
    nxt=np.array([[ST.index((b,u)) for u in(-1.,1.)] for(a,b)in ST])
    met=np.zeros(4);bps=np.empty((n,4),np.int8);bpi=np.empty((n,4),np.int8)
    for i in range(n):
        cand=met[:,None]+np.abs(y[i]-exp)**2;new=np.full(4,np.inf);bs=np.zeros(4,np.int8);bi=np.zeros(4,np.int8)
        for st in range(4):
            for u in range(2):
                ns=nxt[st,u]
                if cand[st,u]<new[ns]:new[ns]=cand[st,u];bs[ns]=st;bi[ns]=u
        met=new;bps[i]=bs;bpi[i]=bi
    st=int(np.argmin(met));sr=np.empty(n)
    for i in range(n-1,-1,-1):sr[i]=2.*bpi[i,st]-1.;st=bps[i,st]
    return np.r_[1.,sr[1:]*sr[:-1]]

def cma(y,taps=7,mu=1e-3,it=2):
    n=y.size;w=np.zeros(taps,complex);w[taps//2]=1.;yp=np.r_[np.zeros(taps//2,complex),y,np.zeros(taps//2,complex)]
    out=np.zeros(n,complex)
    for _ in range(it):
        for i in range(n):
            seg=yp[i:i+taps][::-1];o=np.vdot(w,seg);out[i]=o;w-=mu*(o*(np.abs(o)**2-1.))*np.conj(seg)
    return out

def ber_viterbi_at(snr, npil, ntr):
    errs=np.zeros(ntr)
    for tr in range(ntr):
        r=np.random.default_rng(600*npil+tr)
        b=r.integers(0,2,N_BITS);x=1.-2.*b;y,_=composite(x,snr,r)
        sp=diff_encode(x[:npil]);h=ls_ch(y[:npil],sp)
        xh=viterbi(y,h);yd=hop2(xh,snr,r)
        # exclude pilot region from payload BER, and count overhead
        errs[tr]=np.mean((yd.real<0)[npil:]!=b[npil:])
    return errs.mean(), 1.96*errs.std()/np.sqrt(ntr)

def ref_at(snr, net, ntr):
    em=np.zeros(ntr);ec=np.zeros(ntr)
    for tr in range(ntr):
        r=np.random.default_rng(999+tr)
        b=r.integers(0,2,N_BITS);x=1.-2.*b;y,_=composite(x,snr,r)
        o=net.fwd(cwin(y));xh=o/(np.sqrt(np.mean(o**2))+1e-12);yd=hop2(xh,snr,r)
        em[tr]=np.mean((yd.real<0)!=b)
        eq=cma(y);d=np.real(eq[1:]*np.conj(eq[:-1]));xc=np.r_[1.,np.sign(d)+(d==0)];yd=hop2(xc,snr,r)
        ec[tr]=np.mean((yd.real<0)[1:]!=b[1:])
    return (em.mean(),1.96*em.std()/np.sqrt(ntr)),(ec.mean(),1.96*ec.std()/np.sqrt(ntr))

net=train()
print(f'\n=== Pilot-budget sweep at {OP_SNR:.0f} dB (payload BER; pilots excluded from payload) ===')
mlp_ref, cma_ref = ref_at(OP_SNR, net, N_TRIALS)
print(f'  MLP-169 (0 pilots): BER={mlp_ref[0]:.4f}  |  CMA blind (0 pilots): BER={cma_ref[0]:.4f}')
sweep={}
for npil in PILOTS:
    mu,ci=ber_viterbi_at(OP_SNR, npil, N_TRIALS)
    overhead=npil/N_BITS
    sweep[npil]=(mu,ci)
    print(f'  Viterbi, {npil:>4d} pilots: BER={mu:.4f} (overhead {100*overhead:.2f}% of block)')
np.save('/home/claude/e6_partial.npy',
        {'op_snr':OP_SNR,'pilots':PILOTS,'sweep':sweep,'mlp':mlp_ref,'cma':cma_ref,'nbits':N_BITS},
        allow_pickle=True)
print('saved.')
