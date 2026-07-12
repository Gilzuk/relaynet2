#!/usr/bin/env python3
# ============================ PORT SPEC (for Claude Code) ============================
# TARGET: relaynet. Mostly ANALYTICAL: Viterbi flops = M^(L-1) states x M branches x ~8;
#   MLP flops = const (~330). Panel (a) formula plot; panel (b) measured wall-clock.
# HONEST CAVEAT TO PRESERVE: on BPSK/L=3 Viterbi is CHEAPER per-flop (64 vs 330); MLP wins on
#   M^L scaling + 30-90x wall-clock (vectorized vs sequential).
# RECONCILIATION: wall-clock ratio is implementation-specific. Re-measure with relaynet's actual
#   Viterbi, OR keep only the M^L scaling as the robust claim. See PORTING.md section 6.
# ===================================================================================
"""E6 addendum 9: COMPLEXITY -- Viterbi MLSE vs the 170-parameter MLP relay.
Two axes:
  (1) analytical per-symbol operation counts, and scaling in modulation order M and channel memory L;
  (2) measured wall-clock inference time on identical signal lengths.
No BER here -- this quantifies the cost side of the accuracy/cost trade-off the thesis rests on.
"""
import numpy as np, time
rng = np.random.default_rng(0)
W = 11

# ---------------- analytical per-symbol cost ----------------
def viterbi_ops(M, L):
    """MLSE per-symbol: states = M^(L-1); branches per state = M; each branch metric ~ a few flops.
    Add-compare-select over M^L branch transitions per symbol."""
    states = M ** (L - 1)
    branches = states * M                 # = M^L
    # per branch: 1 complex subtract + magnitude-square (~4 real mul+add) + 1 add + compare
    flops_per_branch = 8
    return states, branches, branches * flops_per_branch

def mlp_ops(win=W, hid=7, cin=2):
    """MLP-169 per-symbol: one forward pass. din = cin*win."""
    din = cin * win                        # 22
    # layer1: din*hid mul-add + hid tanh; layer2: hid*1 mul-add + 1 tanh
    macs = din * hid + hid * 1             # 22*7 + 7 = 161
    flops = 2 * macs + hid + 1             # mul+add each MAC, + activations
    params = din * hid + hid + hid + 1     # 169
    return params, macs, flops

print('=== Analytical per-symbol cost ===')
print(f'{"M":>3} {"L":>3} {"Viterbi states":>15} {"Vit branches/sym":>17} {"Vit flops/sym":>14} {"MLP flops/sym":>14}')
_, _, mlp_f = mlp_ops()
for M, L in [(2,3),(2,5),(2,7),(4,3),(4,5),(16,3),(16,5)]:
    st, br, vf = viterbi_ops(M, L)
    print(f'{M:>3} {L:>3} {st:>15,} {br:>17,} {vf:>14,} {mlp_f:>14,}')
print(f'\nMLP-169 is CONSTANT in M and L: {mlp_ops()[2]} flops/symbol, {mlp_ops()[0]} params.')
print('Viterbi grows as M^L per symbol (states M^(L-1) x M branches); intractable for large M or L.')

# ---------------- measured wall-clock ----------------
def make_signal(n):
    x = rng.choice([-1.,1.], n).astype(complex)
    h = np.array([1.,0.6,0.4]); h/=np.linalg.norm(h)
    y = np.convolve(x, h)[:n] + 0.1*(rng.standard_normal(n)+1j*rng.standard_normal(n))
    return y

def viterbi_run(y, h=np.array([1.,0.6,0.4])/np.linalg.norm([1.,0.6,0.4])):
    n=y.size; ST=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]
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
    return sr

W1=rng.standard_normal((22,7));b1=np.zeros(7);W2=rng.standard_normal((7,1));b2=np.zeros(1)
def mlp_run(y):
    yi=np.pad(y.real,(W//2,W//2));yq=np.pad(y.imag,(W//2,W//2))
    X=np.concatenate([np.lib.stride_tricks.sliding_window_view(yi,W),
                      np.lib.stride_tricks.sliding_window_view(yq,W)],1)
    return np.tanh(np.tanh(X@W1+b1)@W2+b2).ravel()

print('\n=== Measured inference time (BPSK, L=3, 4-state Viterbi vs MLP-169) ===')
print(f'{"length":>8} {"Viterbi (ms)":>13} {"MLP (ms)":>10} {"speedup":>8}')
for n in [1000, 5000, 20000, 50000]:
    y=make_signal(n)
    t0=time.perf_counter(); [viterbi_run(y) for _ in range(3)]; tv=(time.perf_counter()-t0)/3*1000
    t0=time.perf_counter(); [mlp_run(y) for _ in range(3)]; tm=(time.perf_counter()-t0)/3*1000
    print(f'{n:>8} {tv:>13.2f} {tm:>10.2f} {tv/tm:>7.1f}x')

print('\nNote: MLP is vectorized (single batched matmul); Viterbi is an inherently sequential scan.')
print('The MLP gap widens further for larger M/L, where Viterbi states grow but MLP cost is unchanged.')
