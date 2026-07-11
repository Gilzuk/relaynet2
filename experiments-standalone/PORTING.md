# PORTING.md — Integrating the Chapter 7 (E6) experiments into `relaynet`

## Purpose of this document

The seven `e6_*.py` scripts in `experiments-standalone/` are **self-contained reference
implementations** written to validate the Chapter 7 (unknown-channel) results. They use
their own numpy/PyTorch channel, relay, and simulation code — NOT the `relaynet`
framework. This document is the spec for porting them onto `relaynet`'s real classes so
that every Chapter 7 number is regenerated through the same pipeline as Chapters 5–6.

**Why this matters:** the thesis (Appendix C) states all results come from the `relaynet`
framework. Until these are ported, Chapter 7's numbers come from a second implementation.
The results are genuine, but reconciling them through `relaynet` is required for
methodological consistency and to catch any small divergences (channel normalization, SNR
convention, estimator regularization) between the two implementations.

## How to run this port (Claude Code)

Open Claude Code in the `relaynet2` working tree with these scripts available, then:

> "Read PORTING.md and the scripts in experiments-standalone/. For each experiment, port
> it to use relaynet's existing Channel, Relay, and simulation/runner classes instead of
> its standalone implementations, following the interface mapping in PORTING.md. Preserve
> the exact experimental logic and conventions; only swap the infrastructure. Regenerate
> each figure through our runner at the project's standard trial budget. Flag any place
> where relaynet's behavior differs from the standalone assumption so we can reconcile it."

Do them **one script at a time**, verify the ported BER matches the standalone within
Monte-Carlo noise before moving on. Start with `e6_sim.py` (simplest) to establish the
channel/relay mapping, then reuse that mapping for the rest.

---

## Shared conventions (must match `relaynet` exactly — verify first)

| Quantity | Standalone value | Action in port |
|---|---|---|
| SNR → noise | `sigma = 10**(-snr_db/20)`, so **γ = 1/σ²**, single-hop AWGN BER = Q(√γ) | Confirm relaynet uses the same γ definition. If relaynet defines SNR as Es/N0 or Eb/N0 with a different factor, **all SNR axes must be rescaled**. This is the #1 reconciliation risk. |
| Modulation | BPSK: `x ∈ {+1,−1}`, bit = `(x<0)` | Use relaynet's BPSK mapper; check bit-to-symbol polarity matches. |
| Window relay | length `W=11` sliding window, symmetric, zero-padded | Map to relaynet's windowed-relay input. Confirm padding convention. |
| MLP-170 | real input: `11 → 13 → 1`, tanh, tanh-output | Instantiate relaynet's MLP relay with `in=11, hidden=13, out=1`, tanh activations. |
| MLP-169 | complex I/Q input: `22 → 7 → 1` | Only used where noted (phase cases). `in=22, hidden=7`. |
| Training SNRs | `[5, 10, 15] dB`, MSE to clean symbol | Match relaynet's training loop; loss = MSE(relay_out, x). |
| Trials | standalone uses 5–6 × 30k–50k | **Raise to the project standard (10 × 100k)** when regenerating for the thesis. |
| Second hop | Rayleigh: `y = |h|·x_R + n`, coherently compensated (real) | Use relaynet's Rayleigh channel with magnitude compensation. |
| Hard decision | `sign(y)` (real part after hop 2) | Use relaynet's detector. |

**Reconciliation checkpoint:** before porting any experiment, run relaynet's existing
canonical BPSK/Rayleigh case and confirm it reproduces the closed-form Q(√γ) AWGN and the
Rayleigh BER curve. If it does, the SNR/noise conventions align and the ports will match.
If not, resolve the convention mismatch *first* — otherwise every E6 curve will be shifted.

---

## Per-experiment mapping

### 1. `e6_sim.py` + `e6_viterbi.py` → Unknown ISI (the core positive result)
**Produces:** `results/e6_unknown_channel.png`, tables tbl:tableE6.
- **Channel to add:** 3-tap ISI `H_ISI = [1.0, 0.7, 0.5]/‖·‖`, applied as `np.convolve(x, H)[:n]`.
  → Implement as a `Channel` subclass `UnknownISIChannel(taps=[1,0.7,0.5])`. Hop 1 only.
- **Relays:** AF (`y/√E[|y|²]`), symbol-wise DF (`sign(y)`), MLP-170. All already exist in
  relaynet except the ISI channel — reuse existing relays.
- **Viterbi (`e6_viterbi.py`):** 4-state BPSK MLSE over the trellis, two variants — genie CSI
  and 200-pilot LS estimate. If relaynet has no MLSE detector, add `ViterbiMLSE(taps=3)` as a
  detector/relay; port the `viterbi()` function body verbatim (it's standard add-compare-select).
- **Key numbers to reproduce:** DF/AF pinned at analytic 0.25 floor (DF non-monotonic in SNR);
  MLP-170 < 5×10⁻⁵ at 16 dB (AWGN hop 2); genie Viterbi ~1–1.5 dB ahead of MLP.
- **Analytic check:** the 0.25 floor is exact (effective amplitude h0±h1±h2 has one sign-flip in 4).
  Verify the ported DF hits it.

### 2. `e6_flat.py` → Flat (memoryless) unknown channels — THE CONTROL
**Produces:** table tbl:tableE6flat (three rows, MLP-170).
- **Three channels to add (all memoryless, params redrawn per block):**
  - `phase`: `y = e^{jθ}·s + n`, θ~U[0,2π), DBPSK source. Uses complex I/Q → MLP-169 (`22→7`).
  - `gain`: `y = g·x + n`, g~U[0.3,2.0]. Real → MLP-170.
  - `iqimb`: per-branch amplitude asymmetry `s = a+ if x>0 else −a−`, a±~U[0.6,1.4]. Real.
  → Add as `FlatPhaseChannel`, `FlatGainChannel`, `BranchAsymmetryChannel`.
- **Expected result (the point of the control):** classical does NOT fail; MLP only ties DF,
  max gap ≤ 0.0036. This must hold — it's the falsification test proving memory (not
  unknownness) is what breaks classical relays.
- **DBPSK helper:** `diff_encode(x)` — running product; port verbatim. Classical baseline for
  the phase case is conventional differential detection `sign(Re{y[i]·conj(y[i−1])})`.

### 3. `e6_composite.py` → Composite cascade
**Produces:** `results/e6_composite.png`.
- **Channel:** cascade DBPSK → 3-tap ISI (`[1,0.6,0.4]/‖·‖`) → Rapp-like PA `pa(z,sat=1.2)`
  → unknown phase → noise. Random ISI per block in the blind/partial variants; fixed here.
  → `CompositeChannel(parts=['isi','pa','phase'])`.
- **`pa(z)`:** soft-limiter on magnitude, `g = a/(1+(a/sat)²)^0.5`, preserves phase. Port verbatim.
- **Baselines:** AF, DF-diff, pilot-LS Viterbi+differential (matched to ISI+phase, blind to PA),
  MLP-169 and MLP-1153.
- **Key numbers:** AF/DF floored at 0.25; MLP-169 → 5×10⁻³ @ 20 dB; Viterbi ~2 dB ahead of MLP
  at low-mid SNR; MLP-1153 ≈ MLP-169 (H3).

### 4. `e6_blind.py` → Posterior-free (blind) regime
**Produces:** `results/e6_blind.png`.
- **Same composite channel but random ISI per block, no pilots exposed.**
- **New baseline to add:** `cma_dfe(y)` — constant-modulus blind equalizer (7 taps, μ=1e-3,
  2 passes). Port verbatim as a `CMAEqualizer` relay. Also `blind_viterbi()` — decision-directed
  bootstrap MLSE (port verbatim).
- **Key result:** CMA works (0.0024 @ 20 dB), MLP ties (0.0026); **decision-directed blind
  MLSE is unstable** — mid-SNR CI 0.164 vs MLP 0.014. The instability is the finding; verify it
  reproduces (it may be sensitive to the bootstrap — flag if your port stabilizes it, that
  changes the claim).

### 5. `e6_partial.py` (+ `e6_blocklen` logic inside) → Partial-posterior sweep
**Produces:** `results/e6_partial.png` (2 panels).
- Reuses the composite channel + Viterbi + CMA + MLP from above.
- **Panel (a):** sweep pilot count ∈ {800,200,50,20,10,5} at 10 dB; Viterbi BER vs pilots,
  MLP/CMA as flat references. **Key:** Viterbi wins ≥10 pilots, COLLAPSES at 5 (0.119) —
  identifiability. MLP flat at 0.045.
- **Panel (b):** block-length sweep {40,80,160,320,1000}, classical spends 10 pilots/block →
  overhead 25% at L=40 vs 1% at L=1000; MLP zero overhead.
- **Reconciliation risk:** the 5-pilot collapse may depend on LS regularization (`rcond` in
  `lstsq`). If relaynet's estimator regularizes differently, the collapse point may shift —
  reproduce the qualitative cliff, report the exact crossover your pipeline gives.

### 6. `e6_complexity.py` → Complexity analysis
**Produces:** `results/e6_complexity.png` (2 panels).
- **Mostly analytical** — Viterbi flops = M^L · const, MLP flops = const (~330). Panel (a) is
  a formula plot; panel (b) is measured wall-clock (numpy MLP vs Python-loop Viterbi).
- **Honest caveat to preserve:** on BPSK/L=3 Viterbi is *cheaper* per-flop (64 vs 330); the MLP
  wins on M^L scaling and 30–90× wall-clock (vectorized vs sequential).
- **Reconciliation note:** the wall-clock ratio is implementation-specific. A C-optimized
  Viterbi in relaynet would narrow the 30–90×. Either re-measure honestly with relaynet's
  actual Viterbi implementation, or state the comparison is numpy-MLP vs Python-Viterbi and
  keep only the M^L scaling as the robust claim.

---

## Acceptance criteria (per experiment)

A port is done when:
1. It runs entirely through `relaynet`'s Channel/Relay/runner classes — no standalone numpy
   channel or hand-rolled MLP.
2. Its BER curve matches the standalone within Monte-Carlo CI at the standalone trial budget.
3. Re-run at the project standard (10×100k) for the final thesis figure.
4. The figure regenerates via relaynet's plotting, styled consistently with Ch5–6 figures.
5. Any convention divergence (SNR, normalization, estimator) is documented and reconciled.

## After porting — update the thesis
- Replace the six `results/e6_*.png` with the relaynet-generated versions.
- Update the Chapter 7 tables with the regenerated numbers (expect small shifts; the
  qualitative claims — 0.25 floor, ~1–1.5 dB Viterbi gap, 5-pilot collapse, M^L scaling —
  should be stable).
- Update Appendix C so the reproducibility statement is now literally true for Chapter 7.
- Remove the "clean-room / standalone" caveat once the numbers come from relaynet.
