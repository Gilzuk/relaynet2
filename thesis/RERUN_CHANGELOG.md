# Re-measurement record — August 2026

Every simulation result in the thesis was re-measured. This file records what
was re-run, what the numbers were before and after, and which written
conclusions changed as a result. It is separate from `CHANGELOG.md`, which
tracks editorial changes to the document; this one covers only the data.

Baseline for every comparison below is commit `c05611f`, the state of the
thesis before any re-run output landed.

---

## Why the results were re-measured

Three independent reasons, discovered in this order.

**1. A 3 dB error in the noise convention.** `relaynet/channels/awgn.py` set
`sigma = sqrt(noise_power)` on its real-valued branch instead of
`sqrt(noise_power/2)`, so the channel delivered 3 dB more noise than its label
claimed. Fixed in `97ca397`.

The fix reaches further than it appears. Every learned relay generates its
*training* data through `awgn_channel`
(`relaynet/utils/activations.py:214`), so correcting the channel moved the
trained weights — and therefore moved results on Rayleigh and on QPSK/16-QAM,
whose own evaluation paths the fix never touched. "The channel was not
modified, so the result stands" is false in this codebase.

**2. Tables resting on configurations the thesis no longer evaluates.** The
pairing rule — a real constellation on a real channel, a complex constellation
on a complex one — left three tables measured on pairings that had been
withdrawn. Each was found the same way: a default argument nobody had looked
at.

| Table | Was measured on | Cause |
|---|---|---|
| `tbl:table2` | BPSK / Rayleigh | real constellation on a complex channel |
| `tbl:table8` | BPSK / Rayleigh | `evaluate_relays` defaults to `modulation="bpsk"` |
| `tbl:table24` | 16-QAM / AWGN | `run_monte_carlo` defaults to `awgn_channel` when `channel_fn` is omitted |

**3. Result files older than the code that produced them.** `relaynet/relays/vae.py`
and `relaynet/relays/cgan.py` were both created on 2026-04-01, while the
results carrying those relays were dated 2026-03-23. Those numbers came from a
superseded implementation.

---

## What was re-run

Budget throughout: 10 trials × 10,000 bits per SNR point, SNR 0:2:20, matching
`ch04_methods.tex`. Runtimes are wall-clock on 4 CPU cores, no CUDA.

| Stage | Produces | Runtime |
|---|---|---|
| 7.1 channel validation | calibration figures | 5 s |
| 7.10 modulation | `tbl:table2`, `tbl:table14` | 2,892 s |
| 7.8 normalized-3K | `tbl:table8` | 747 s |
| 7.17 16-class | `tbl:table24` | 4,548 s |
| 7.11 activation | `tbl:table15` | 6,608 s |

`7.17` was run twice. The first pass put its twelve learned variants on
Rayleigh but left the AF and DF baselines on AWGN, because a *second*
`run_monte_carlo` call for the baselines also omitted `channel_fn`. It was
caught before transcription: the run reported AF at 0.00063 and DF at exactly
0.00000, against 0.0450 and 0.0375 for the same relays on the same channel in
`tbl:table14` — impossible, and the tell that the two halves of the table were
measured on different channels.

**Not re-run, deliberately:**

- **The cGAN**, dropped on instruction. Its column is absent from
  `tbl:table2`, `tbl:table14`, `tbl:table15` and `tbl:table24`, and the
  captions say so. It cannot be reinstated from the old numbers (superseded
  implementation, above), nor run alone and pasted in: training it consumes
  the shared RNG stream, so its presence shifts every other relay's numbers
  and the table would carry rows from two different draws.
- **`tbl:table13` and `tbl:table25`**, which report wall-clock timings
  measured on different hardware. The verifier treats them as informational.

---

## Numbers, before and after

### `tbl:table2` — canonical relay comparison

Reconfigured as well as re-measured: BPSK/Rayleigh (9 relays) → **QPSK/Rayleigh
(8 relays)**, QPSK having become the canonical constellation. Rows are not
comparable one-to-one; the whole table was replaced.

| Relay | 0 dB | 8 dB | 20 dB |
|---|---|---|---|
| AF | 0.4076 | 0.1852 | 0.0110 |
| DF | 0.3337 | 0.1218 | 0.0097 |
| MLP (169p) | 0.3349 | 0.1229 | 0.0099 |
| Hybrid | 0.3329 | 0.1220 | 0.0097 |
| VAE | 0.3359 | 0.1235 | 0.0097 |
| Transformer | 0.4014 | 0.1424 | 0.0101 |
| Mamba-S6 | 0.4071 | 0.1448 | 0.0100 |
| Mamba-2 SSD | 0.4032 | 0.1416 | 0.0098 |

The DF closed form was re-derived for the new constellation. Converting the
abscissa from `Es/N0` to `Eb/N0` (3.01 dB lower at k=2), `2P(1−P)` predicts
0.0098 at 20 dB against a measured 0.00972, and 0.0239 at 16 dB against
0.02451 — within 1% at both.

### `tbl:table8` — normalized 3K

BPSK/Rayleigh → **QPSK/Rayleigh**, and the study now takes its constellation
from `CANONICAL_PAIRS` rather than the `evaluate_relays` default.

### `tbl:table14` — 16-QAM

Was BPSK, QPSK *and* 16-QAM all on AWGN — two of those pairings withdrawn, the
third duplicating Chapter 5. Now **16-QAM on Rayleigh alone**, which states the
chapter's finding directly: AF and DF fall with SNR (DF 0.4197 → 0.0375) while
every learned relay flattens at 0.25–0.26.

### `tbl:table15` — output activation

AWGN + Rayleigh → **Rayleigh only**. The claimed benefit shrank: bounding the
output improves the sequence models by a factor of **2.4**, not the 5× reported
from the AWGN run (Transformer 0.2595 → 0.1073), narrowing their gap to DF from
3.1× to 1.3×. The feedforward relays are unmoved.

### `tbl:table24` — 4-class versus 16-class

16-QAM/AWGN → **16-QAM/Rayleigh**.

| Architecture | 4-class | 16-class | Ratio |
|---|---|---|---|
| MLP | 0.05491 | 0.03840 | 1.43× |
| VAE | 0.05382 | 0.03762 | 1.43× |
| Hybrid | 0.05362 | 0.03748 | 1.43× |
| Transformer | 0.05360 | 0.03814 | 1.41× |
| Mamba-S6 | 0.05421 | 0.03934 | 1.38× |
| Mamba-2 SSD | 0.05381 | 0.04953 | 1.09× |

On AWGN the 16-class heads reached ~0 and the improvement read as 51× for the
MLP. On Rayleigh the channel itself limits DF to 0.03748, so raw BER cannot
approach zero and the ratio is meaningless in isolation. The floor is properly
measured as **excess over DF**: 0.0161–0.0174 for the per-axis relays,
0.0001–0.0019 for the joint ones — a reduction of 95% or more for five of six.
The finding survives; its magnitude and the way it must be stated do not.

---

## Written conclusions that changed

**H4 — architecture convergence at equal scale.** Read "confirmed: at 3K
params, all methods within ~1% BER (except VAE)". Neither half survived. The
spread is SNR-dependent, falling monotonically from 17.4% at 0 dB to 4.1% at
20 dB; and at low SNR it is not a spread but a clean split by family, the
feedforward models at 0.336–0.342 near DF and the sequence models at
0.400–0.407 near AF. Since all six carry the same parameter budget this is
architecture mattering, not capacity. Now recorded as **confirmed at high SNR,
rejected at low**.

**The VAE failure.** The thesis reported the VAE "pinned at 0.25–0.40 at every
SNR, meaning it never learns the task". It does not reproduce: the VAE now
tracks DF, reaching 0.00972 at 20 dB against DF's 0.00972.

The obvious explanation was the 3 dB bug, since learned relays train through
the corrected channel. **A controlled A/B refutes it.** Evaluation runs on
Rayleigh, untouched by the fix, so the only difference between arms is the
convention used to synthesise training data; the patch was probe-verified to
bite (a nominal 10 dB arrives as 7.00 dB in the old arm, 10.01 dB in the new).
Both arms behave — 0.3349 and 0.3368 at 0 dB — so training on 3 dB-hot data
does not break the VAE. The cause is that `relaynet/relays/vae.py` postdates
the published numbers. Recorded in `results/vae_convention_ab.json`.

Chapter 2's conclusion, that the generative paradigm confers no advantage
here, survives and is better supported: the VAE arrives where a 169-parameter
regressor already is, at greater cost, rather than failing.

**16-QAM joint classification.** "Removes the structural floor, achieving
near-zero BER" becomes parity with DF — the floor is removed, but the endpoint
is the classical baseline rather than past it. This is H2 reappearing one
constellation higher.

---

## Claim audit — statements checked against measurement

A separate pass over the prose, looking for three faults the table verifier
cannot catch: claims that contradict the measurements, claims stated more
confidently than the evidence supports, and claims that conflict with each
other. Seven were found and fixed. Numbers below are from the files named in
the sections above.

### Contradicted by measurement

**1. The 3K architecture gap (§5.4).** Read "the performance gap between
feedforward and sequence architectures narrows to ~1% BER". Measured spread is
**17.4% at 0 dB and 4.1% at 20 dB** — the claim is wrong at every SNR, and
wrong by a factor of seventeen at the low end. The same claim had already been
corrected in Chapter 8 when H4 was revised; this instance was missed. Now
states the SNR dependence and the feedforward/sequence split.

**2. "Removes the floor entirely" (§8, §9).** Both chapters said joint
16-class classification eliminates the per-axis floor and that the top variants
match DF. Measured as excess over DF, which is the meaningful quantity because
DF is itself channel-limited at 0.03748: per-axis relays carry 0.0161–0.0174,
joint relays 0.0001–0.0019 — but **Mamba-2 SSD retains 0.0121** and does not
reach parity. "Entirely" overstated it and the exception was unmentioned.

**3. cGAN priority claim (§2.4).** Claimed "the first systematic comparison of
VAE and cGAN-based relay processing". The cGAN is excluded from every table in
the current revision, so the thesis reports no cGAN comparison to be first at.
Rewritten to describe what is actually reported, with the exclusion stated.

**4. QPSK framing (§9).** "The canonical findings generalized fully to QPSK"
predates QPSK becoming the canonical constellation; there is no longer a
separate thing for findings to generalise to.

### Overconfident

**5. "Proved correct" (§4.2).** AWGN calibration was said to be where the
simulator "can be proved correct". Agreement within 0.6% against a closed form
validates; it does not prove. Now "checked against an exact result rather than
only against another simulation".

**6. Timing speedup (§8.3).** The 10.7× Mamba-2 figure was stated as a property
of the architecture. It is a wall-clock measurement of one reference
implementation on one machine, and the timing tables were **not** re-measured
in this revision. Now stated as such.

### Conflicting between sections

**7. "The only pilot-free option that remains reliable" (§7.5).** True at the
block lengths of that study — CMA sits at 0.1723 on 40-symbol blocks and 0.1653
at 1,000 — but it contradicted §7.6, where blocks are long enough for CMA to
converge and it comes within a factor of 1.3 of the learned relay. The claim is
now scoped to short blocks and the difference is stated.

### Checked and left alone

Uses of "optimal" for MAP and MLSE detection are technically correct and stay.
"Fixed" is retained where literally true, as for the three-tap ISI filter whose
taps genuinely do not change. The universal-approximation citation is a named
theorem, not a claim about this work. The "169 parameters matching models 140×
larger" ratio checks out against the recorded parameter counts.

---

## New experiment: coded block-DF (not a re-measurement — added this pass)

Everything above re-measures results the thesis already claimed. This is
different: a genuinely new experiment, added to close a caveat the thesis had
only ever asserted, not measured. Remark `rem:df-terminology` (Ch1) states
that the DF baseline used everywhere in this thesis is *symbol-wise* (uncoded,
per-symbol hard slicing), not the *block* DF of the information-theoretic
relay-channel literature, and cautions that "the reported DF results should
not be read as bounds on coded block-DF performance." That sentence had no
number behind it until now.

**What was built** (`relaynet/coding/convolutional.py`,
`relaynet/coding/convolutional_qam16.py`,
`relaynet/relays/coded_df.py` / `coded_df_qam16.py`): a rate-1/2 convolutional
code (constraint length `K ∈ {3, 5, 7}`, standard maximal-free-distance
generators, zero-tail terminated per frame) with a soft-decision Viterbi
decoder, and `CodedDecodeAndForwardRelay` — genuine block DF: decode the full
frame, re-encode, re-modulate, forward, as opposed to the per-symbol slicing
used everywhere else in this thesis. 16-QAM needed a separate decoder
(`QAM16CodeDecoder`) rather than a parameter change, since its 2-bit Gray
mapping onto one PAM-4 level is not decomposable into independent per-bit soft
observations the way QPSK's is. Two coded-aware learned relays were trained on
the same task for comparison: a windowed 756-parameter MLP and the Mamba-S6
architecture already used in `tbl:table2` (24,084 parameters at this
configuration) — reusing the existing architectures, not building new ones.

**Where it's written up:** `thesis/chapters/ch05_experiments.tex`,
§Coded Block-DF (new `tbl:table34`–`tbl:table36`); pointers added from the
Ch1 remark, the Ch8 limitations/future-work items that used to name this as
open, and both abstracts (English and Hebrew — the Hebrew abstract was found
stale relative to the English one during this pass, missing the four-layer
ladder framing added earlier, and was brought up to parity as part of this
change, not only extended).

**Two findings, reported as measured:**

1. **The caveat holds only above a threshold, not everywhere.** Coded block-DF
   beats uncoded symbol-wise DF decisively from ~8 dB up (5.6× lower BER at
   16 dB, 7.6× at 20 dB) but is *worse* than uncoded DF below ~4–6 dB — the
   well-known convolutional-code error-propagation threshold, sharper for
   stronger codes. A constraint-length sweep (K=3,5,7, both QPSK and 16-QAM)
   found larger K does **not** monotonically help within the measured frame
   length (200 information bits) and trial budget (10×100,000 bits/point): K=3
   remained competitive with or better than K=5/K=7 at nearly every SNR point
   tested, the stronger codes only costing more at low-mid SNR.
2. **Neither coded-aware learned relay beats the classical decoder, even with
   real temporal structure to exploit** — unlike the canonical memoryless
   channel, where §5's parameter-normalization study found "the memoryless
   channel simply offers no temporal structure for [the sequence models']
   inductive bias to exploit." Both the MLP-coded and Mamba-coded relays close
   to within trial noise of coded-DF at 16–20 dB but are measurably worse at
   4–8 dB, and the 32×-larger Mamba-S6 relay shows no clear advantage over the
   756-parameter MLP anywhere in the sweep — the same H2/H3 pattern
   ("matches, doesn't beat"; "less is more") that recurs throughout this
   thesis, unchanged by giving the sequence model a task with genuine memory.

Data: `results/coded_df_experiment.json`. Scripts:
`coded_df_experiment.py`, `coded_learned_relay.py`, `coded_mamba_relay.py`,
`coded_k_sweep_qpsk.py`, `coded_k_sweep_qam16.py`. Tests:
`tests/test_coding.py` (19 tests: round trips for K∈{3,5,7} on both
decoders, exact bit-packing cross-validation against `qam16_modulate`,
relay-level shape/fidelity checks).

Deliberately not attempted in this pass: retraining either learned relay per
(K, modulation) combination (each training run too expensive to repeat
six-fold — the classical Viterbi decoder was swept instead, and it is the one
the learned relays are compared against); an adaptive rate/K selection scheme
per SNR point; soft-information (LLR) learned relays, as opposed to the
hard-decision ones built here. Recorded as open items in Ch8.

---

## Verification

`verify_thesis_tables.py` checks every published cell against the file that
produced it: **421 cells, 0 inconsistencies** (349 before the coded-DF work;
`check_table34`/`35`/`36` added, sourced from `results/coded_df_experiment.json`).
One real transcription error was caught in the process — 0.157745 had been
written as 0.1578, which rounds to 0.1577 — and fixed before it could ship.
Coverage was extended earlier during this work too, and three checks were
found to be silently disabled:

- `check_table8` aborted on a `KeyError` for `GenAI-3K` after the relay was
  renamed `MLP-3K`, losing 40 cells while reporting "skipped".
- `check_table14ray` failed the same way and lost all 68 of its cells.
- `check_table24` matched row labels literally, so `cGAN`, `Mamba-S6`,
  `Mamba-2 SSD`, `AF` and `DF` were never compared — five of nine rows
  unchecked, while the table reported OK on the four that were.

Name resolution is now central (`RELAY_ALIASES`, `resolve()`), row labels are
normalised, and a table that cannot cover a row reports partial coverage
instead of dropping it from the count. `tbl:table15` had never had any coverage
at all, which is why it sat on a pre-correction AWGN run unnoticed; it now has
24 cells.

## Reproducing

```
python3 run_experiments.py --exp 7.10 7.8 7.17 7.11 \
    --num-trials 10 --bits-per-trial 10000 --skip-relays cgan --retrain
python3 verify_thesis_tables.py
```

`CANONICAL_PAIRS` in `run_experiments.py` is the single definition of which
(constellation, channel) configurations are evaluated.

## Follow-up: the high-SNR claim was wrong, and the mechanism behind it

The section above reported the coded-aware learned relays as "within trial
noise" of coded block-DF at 16-20 dB. That was a judgement on 10 unpaired
trials, and it does not survive testing. Re-measured with **100 paired
trials** (identical bits and identical channel realizations per trial across
relays, so channel-draw variance cancels in the per-trial difference),
Wilcoxon signed-rank:

| SNR | relay | BER | diff vs coded-DF | W/L | p |
|---|---|---|---|---|---|
| 16 dB | coded-DF | 0.004245 | --- | --- | --- |
| | MLP-coded | 0.004437 | +0.000192 | 23/77 | <1e-4 |
| | Mamba-coded | 0.004124 | -0.000121 | 68/32 | <1e-4 |
| 20 dB | coded-DF | 0.001362 | --- | --- | --- |
| | MLP-coded | 0.001066 | -0.000295 | 99/1 | <1e-4 |
| | Mamba-coded | 0.000967 | -0.000395 | 100/0 | <1e-4 |

Both directions are real. At 20 dB both learned relays beat the classical
decoder in essentially every trial; at 16 dB the result splits by
architecture. The original call was wrong to dismiss the gaps, and it would
have been equally wrong to upgrade it to "learned beats Viterbi" -- neither
sentence describes the data.

**Why it happens, and why it is not a decoding result.** Instrumenting the
relay output (`coded_error_mechanism.py`, 20 trials x 500 frames):

| SNR | relay | relay symbol ER | repaired downstream | final BER |
|---|---|---|---|---|
| 16 dB | coded-DF | 0.00507 | 15.5% | 0.004282 |
| | MLP-coded | 0.01075 | 57.2% | 0.004603 |
| | oracle | --- | --- | 0.002192 |
| 20 dB | coded-DF | 0.00183 | 25.8% | 0.001355 |
| | MLP-coded | 0.00390 | 74.4% | 0.001001 |
| | oracle | --- | --- | 0.000655 |

Viterbi is the better decoder, exactly as the optimality result says: it puts
~2.1x *fewer* symbol errors on the air. It loses end-to-end because block-DF
re-encodes, so a wrong decode leaves the relay as a valid-but-wrong codeword
the destination cannot detect. This also explains the 16-vs-20 dB split: at
16 dB the 2.1x raw-error penalty still outweighs the repair advantage, at
20 dB it does not.

**Soft decision** (`coded_soft_decision.py`, paired, 20 trials x 500 frames).
Soft read-out helps the learned relay at every SNR on *identical weights*
(argmax -> softmax posterior mean, no retraining): 0.4327/0.4424 at 0 dB,
0.0849/0.0970 at 8 dB, 0.000885/0.001069 at 20 dB. At 16 dB it flips the MLP
from significantly losing to block-DF (2/20) to significantly beating it
(20/20).

BCJR soft block-DF, by contrast, does **not** rescue block-DF -- contradicting
the hypothesis that motivated building it. It ties hard block-DF at 8-16 dB
and is worse at 20 dB (0.002379 vs 0.001376, 0/20). A variance sweep
(`coded_soft_df_calibration.py`) shows the 20 dB degradation is
mis-calibration: the relay assumes the nominal noise variance while
post-equalization Rayleigh noise is 1/|h|^2 larger, and inflating the assumed
variance 2x removes the entire gap (0.002394 -> 0.001367), every factor up to
100x holding there. Viterbi is structurally immune, its squared-distance
metric being invariant to uniform variance scaling.

But even calibrated, soft block-DF only *ties* hard block-DF; it never beats
it, because at high SNR the BCJR posteriors saturate and the posterior mean
degenerates to the hard constellation point. **So the liability is not hard
quantization at the relay -- it is consuming the code's redundancy at the
relay at all.** Any relay that decodes the code, Viterbi or BCJR, hard or
soft, has spent that redundancy before transmitting. The learned relay wins at
high SNR precisely because it never decodes.

Written up in Chapter 5 (`tbl:table37`-`tbl:table39`, `fig:fig57`), with the
Ch1/Ch8 pointers and both abstracts updated. Verifier: **467 cells, 0
inconsistencies**. Tests: 146 passing.

## Correction: the coding gain was quoted on the wrong axis

The coded block-DF section reported coding as beating uncoded DF by 5.6x
at 16 dB and 7.6x at 20 dB. Both figures are measured at equal Es/N0, and
at equal Es/N0 the rate-1/2 coded QPSK link carries **half the
information per channel use** (0.99 info bits/symbol against uncoded
QPSK's 2.00). The comparison was therefore not like-for-like, and the
quoted factors overstate what a link designer would get.

The equal-spectral-efficiency comparison needs no new simulation: rate-1/2
16-QAM delivers 1.98 info bits/symbol, matching uncoded QPSK's 2.00, and
both were already measured.

| SNR | uncoded QPSK | rate-1/2 16-QAM | coding gain |
|---|---|---|---|
| 0 dB | 0.3336 | 0.4687 | 0.71x (worse) |
| 4 dB | 0.2213 | 0.4204 | 0.53x (worse) |
| 8 dB | 0.1204 | 0.2722 | 0.44x (worse) |
| 12 dB | 0.0558 | 0.0989 | 0.56x (worse) |
| 16 dB | 0.0242 | 0.0269 | 0.90x (worse) |
| 20 dB | 0.0099 | 0.0074 | **1.35x (better)** |

Held to equal throughput the code does not pay for itself anywhere below
20 dB, and where it finally does the gain is 1.35x rather than 5.6-7.6x.
Coded block-DF remains genuinely the stronger *detector* and the
Remark 1.2 caveat remains real -- but this bounds how much it matters.

## Latency and compute, which the BER tables also omitted

| relay | buffer before first output | us/symbol | vs Viterbi |
|---|---|---|---|
| AF / symbol-wise DF | 0 symbols | --- | --- |
| MLP hard (756p) | 10 symbols | 0.30 | 50x faster |
| MLP soft (756p) | 10 symbols | 0.22 | 68x faster |
| hard block-DF (Viterbi) | 202 symbols | 15.07 | 1x |
| soft block-DF (BCJR) | 202 symbols | 29.30 | 1.9x slower |

Buffering is the structural cost: a block relay cannot emit until it has
decoded a whole frame, and BCJR cannot even in principle, its backward
recursion starting from the frame's end. That cost scales with frame
length; the learned relay's w-symbol look-ahead does not. So where the
code does pay for itself (20 dB, 1.35x), it asks 20x the latency and 50x
the arithmetic of a learned relay that comes within a few ten-thousandths
of it there.

A measurement note: the first pass of this benchmark reported MLP-hard at
0.88 us/symbol, making the soft variant look 4x cheaper than an
essentially identical forward pass. That was cold BLAS/allocator state on
the first relay benchmarked. Re-run with a discarded warm-up and 7
repeats it is 0.30 us, and the two variants are comparable as they should
be. The warm-up is now part of the harness.

Written up as Section 5.5.3 (`tbl:table40`, `tbl:table41`), with the
overclaim in Section 5.5 corrected in place and pointers added from Ch8's
latency discussion and both abstracts. Verifier: **479 cells, 0
inconsistencies**.

## Adaptive modulation and coding: bounding the mechanism finding

"Let's adjust the rate per SNR point to achieve maximal capacity" -- built
punctured rates (2/3, 3/4 from the same K=3 mother code, standard
max-free-distance patterns, decoder re-inserts deleted bits as soft zeros)
and a BICM pipeline (relaynet/coding/puncturing.py, relaynet/coding/bicm.py)
so puncturing, modulation and decoding are separable, then ran
coded_rate_adaptation.py: goodput G = R*(1-FER) maximized over
{QPSK,16-QAM} x {uncoded,1/2,2/3,3/4}, block-DF vs denoise-only relay, 10
trials x 200 frames x 200 info bits per point.

All 126 measured cells independently re-checked: goodput formula, envelope
argmax selection, and range/monotonicity sanity all recomputed from raw FER
and matched the stored values exactly (0 mismatches).

The envelope is a real staircase:

| SNR | best MCS (both strategies) | goodput |
|---|---|---|
| 8-16 dB | QPSK 1/2 | 0.00-0.60 |
| 20 dB | QPSK 3/4 or 2/3 | ~0.93 |
| 24-28 dB | 16-QAM 2/3 -> 3/4 | 1.5-2.3 |
| 32+ dB | **16-QAM uncoded** | 2.68-3.70 |

The top step is "turn the code off": once the channel is good enough,
coding redundancy is pure overhead and the highest-goodput choice is raw
16-QAM at zero coding loss -- the same conclusion the equal-throughput
correction reached by hand, now reached by direct optimization.

**This bounds the mechanism finding of the soft-decision section rather
than reversing it.** The two relay strategies' envelopes are close at
every SNR and identical for SNR >= 32 dB (both select uncoded, no code
to differ over). Where they differ, block-DF leads at 8/12/24/28 dB,
denoise-only leads at 16/20 dB, and the largest gap over the whole sweep
is 0.095 info bits/symbol at 24 dB -- under 6% of the envelope value
there. Compare to the MCS-choice spread at fixed relay strategy: 1.31 at
24 dB (14x the relay-strategy gap), 1.71+ at 32 dB and above (relay gap
exactly zero there). Rate adaptation dominates the relay-decoding
decision by more than an order of magnitude across most of the sweep --
the destination-repairability mechanism is real, but an adaptive link
mostly avoids the regime where it produces a large gap in the first
place.

Written up as Section 5.5.4 (tbl:table42, fig:fig58), with pointers from
Ch1's system-model paragraph and Ch8's future-work item (which
previously and now-incorrectly listed "adaptive code-rate-per-SNR scheme
... not attempted" -- corrected). Both abstracts updated. 13 new tests
(tests/test_coding.py, punctured round trips, BICM demapping, 40 total
in that file). Verifier: **497 cells, 0 inconsistencies** (was 479).

## Capacity under a latency constraint (bounding the AMC bound)

The AMC/goodput study (previous entry) optimizes rate with no bound on
how long the relay may take to decide. A link with a deadline can't make
that assumption, and "achievable rate under a fixed blocklength" is a
standard question in the literature: finite-blocklength capacity
(Polyanskiy, Poor & Verdu 2010) and delay-limited capacity (Hanly & Tse
1998) for fading channels specifically. Re-derived the AMC envelope under
a round-trip latency budget, using the same already-measured FER data
(coded_rate_adaptation.json) -- no new simulation.

Key accounting fact, not previously made explicit: the destination
*always* buffers a full coded frame to decode, for either relay strategy.
Block-DF adds a second full-frame buffer at the relay (decode, re-encode,
forward), so its round-trip latency is the frame length twice. The
denoise-only relay adds only its window (10 symbols, constant), so its
round-trip latency is the frame length once.

At a 150-symbol round-trip budget: denoise-only leads block-DF by 10.0x
at 16 dB (0.372 vs 0.037) and 2.0x at 20 dB (0.916 vs 0.467) -- the
unconstrained comparison showed these two within 6% of each other. At a
tighter 100-symbol budget, block-DF has *no* coded option left at 16-28
dB at all and is pinned to uncoded transmission throughout, while
denoise-only still reaches 16-QAM 2/3 or 3/4; this reverses the ordering
even at 24 and 28 dB, where the unconstrained envelope had favored
block-DF -- denoise-only leads by 1.46x and 1.54x there under the
100-symbol budget.

All 4 illustrative ratios (10.0x, 2.0x, 1.46x, 1.54x) independently
recomputed from the curve data before writing them into the thesis: 0
mismatches.

Written up as Section 5.5.5 (tbl:table43, fig:fig59), citing
PolyanskiyPoorVerdu2010FiniteBlocklength and HanlyTse1998DelayLimited
(both added to references.bib). Pointers extended in Ch1's system-model
paragraph and Ch8 (both coding-caveat items, plus an explicit "not a
URLLC reliability claim" note added to Ch8's open-items list -- the
latency accounting here says nothing about the 1e-5-range block-error
targets URLLC also requires, and that gap should stay visible rather
than be implied away). Both abstracts updated. Verifier: **515 cells, 0
inconsistencies** (was 497).

## Dropped the activation-function sweep (Ch.6, kept the 2D classifier)

Removed §"Higher-Order Modulation Scalability (Constellation-Aware
Training)" from Chapter 6 (16-QAM extension) at the user's request, to
tighten the thesis to its coherent through-line: the section (Table 14,
Table 15, Figures 25/28/36) swept relay output activations (tanh, linear,
hardtanh, ...) as a partial fix for the 16-QAM per-axis BER floor. It's a
real, previously-reviewed result, but the thesis's own abstract never
cited it -- the joint 2D classifier in the section right after it (kept,
untouched) removes the same floor outright and is what's actually load-
bearing in the narrative. The 2D classifier section is self-contained: its
own Table 24 already shows the per-axis ("4-cls") baseline alongside the
16-class fix, so nothing further needed rescuing from the deleted section.

Also dropped the now-stale "constellation-aware activations reduce but
cannot eliminate" clause from Ch.8's discussion (the floor-generalization
bullet), replaced with a plain statement that the floor exists under
per-axis processing, and rewrote Ch.6's own intro paragraph from "two
studies" to "one study."

Noted for the record, not acted on: while reading Table 14 before
deleting it, its MLP-169p value at 20 dB (0.2590, tanh, per-axis) does not
match Table 24's "4-cls BER @ 20 dB" column for the same relay (0.05491)
-- nominally the same measurement (4-class per-axis MLP, 16-QAM, Rayleigh,
20 dB) under two different table sources. Table 24 was already corrected
once for data-provenance issues (see its REV note); Table 14 was not.
Since Table 14 is now removed, this discrepancy no longer appears in the
thesis, but it's recorded here in case Table 24's own numbers are ever
re-audited.

Verifier: check_table14/check_table15 removed (both tables gone).
**443 cells, 0 inconsistencies** (was 515; -72 cells from the two removed
tables). Cold rebuild: **164 pages** (was 168), 0 undefined refs. Bundles
rebuilt: 61 entries / 27 figures (was 64/30).

## Theoretical-bounds check, written into the charts and tables it applies to

Compared measured results against theoretical bounds and added the
comparison directly into the affected chart/table rather than leaving it
as a one-off analysis, plus fixed the two loose ends the check surfaced.

**Checked (all already-published prose claims re-derived independently,
zero discrepancies found against the thesis's own numbers):**
- Canonical DF (Table 2) vs. the closed-form two-hop composition
  2P(1-P): matches to <1% at every SNR once the QPSK Es/N0->Eb/N0
  (3.01 dB) conversion is applied -- this derivation was already in prose
  (Section 2.5.2.2) but not visible in the table or figure.
- Single-hop "oracle" floor (no relay may beat a genie that recovers hop
  1 perfectly): zero violations across all 8 relays x 11 SNR points.
- Ergodic Rayleigh (Shannon) capacity vs. the AMC goodput envelope
  (Table 42): zero violations, large margin (e.g. 3.70 achieved vs.
  12.46-bit ceiling at 40 dB -- the gap is the constellation/coding
  constraint, not a bound violation). This check did not exist anywhere
  in the thesis before.
- The analytic 0.25 memoryless-relay error floor (unknown-ISI, BPSK):
  independently re-derived from the stated taps h=[1.0,0.7,0.5] and
  confirmed exact (already drawn as a reference line in fig:figE6).
- Genie-CSI Viterbi MLSE vs. the MLP (Ch.7, unknown ISI): mostly holds
  (1-1.5 dB Viterbi lead, 2-10 dB); raw npy data shows near-parity at the
  0 dB floor and again from 12-18 dB (Rayleigh 2nd hop) that the prose
  previously only called out at the single 20 dB endpoint -- differences
  there are all smaller than the reported standard errors, not a real
  reversal. Prose extended to state this precisely (not previously
  false, just narrower than the underlying data supports).

**Added to the thesis:**
- Table 2: new "DF th." column (closed-form two-hop composition per SNR),
  verified cell-by-cell against the same formula. Table wrapped in
  \footnotesize with abbreviated headers after the extra column pushed it
  past the page margin.
- Figure 10 (fig:fig10) regenerated with two reference curves: "DF
  (theory)" (dashed, tracks the measured DF curve almost exactly) and
  "single-hop floor" (dotted, the physical minimum). New script:
  scripts/plot_canonical_theory_overlay.py, reusing run_experiments.py's
  own plot_ber_chart for pixel-identical styling on the 8 measured curves.
- Table 42: new "C (Shannon)" column (ergodic Rayleigh capacity per SNR),
  with a sentence noting the envelope sits well inside it throughout.
- relaynet/modulation/qpsk.py docstring corrected: the "identical to BPSK
  per bit" claim is only exact once Es/N0 (what `snr_db` actually is) is
  converted to Eb/N0 -- this never produced a wrong published number
  (QPSK and BPSK are never plotted on the same axis), but the docstring
  was imprecise about which SNR convention it meant.
- ch07_unknown_and_mismatch_channels.tex: the "genie-CSI MLSE beats the
  MLP by 1-1.5 dB" claim now states precisely where that margin holds
  (2-10 dB) and where it closes to noise level (0 dB, 12-18 dB Rayleigh),
  instead of only flagging the single 20 dB indistinguishable point.

check_table2 and check_table42 extended to verify the new closed-form
columns cell-by-cell (58 + 27 cells respectively). Verifier: **458 cells,
0 inconsistencies**. Cold rebuild: 158 pages (unchanged), 0 undefined
refs. Bundles rebuilt.

## Abstract and introduction: fold in the theoretical-bounds additions

The DF closed-form validation and the Shannon-capacity check on the AMC
envelope (previous entry) were added to the relevant chapter table/figure
but not yet to the top-level narrative documents. Added two sentences to
each of frontmatter.tex, ch01_introduction.tex, and hebrew_abstract.tex:

1. A second, independent simulator calibration point alongside the
   existing BPSK/AWGN one: symbol-wise DF on the canonical Rayleigh
   channel matches the closed-form two-hop composition to within 1% at
   every measured SNR.
2. The AMC goodput envelope's headroom to the channel's own ergodic
   Shannon capacity: under a third of it even at the top of the swept
   range, confirming the gap is set by the finite constellation and
   coding grid, not by anything a different relay strategy could claim.

Both are confirmatory/contextualizing facts, not new conclusions -- they
strengthen claims already in the abstract rather than adding new ones.

Verified: 458 cells / 0 inconsistencies (unchanged), 159 tests, cold
rebuild 158 pages (unchanged) / 0 undefined refs, Hebrew renders cleanly
(checked via direct page extraction). Bundles rebuilt.

## Resolve the outstanding system-model duplication comment

AK's comment on the system model being stated in both Ch.1 and Ch.4 --
"either give it its own chapter, or state it once in Ch.4" -- was still
open (not yet in Appendix E's log; this was a later comment). Resolved by
the second option, consolidating into Chapter 4:

- Ch.1 Section 1.1 (System Model) is now a short pointer paragraph: keeps
  the motivating "canonical setup" narrative (why this configuration,
  preview of extensions) but no equations. The "Scope of the canonical
  model" paragraph and the entire "Two-Hop Relay Model" subsection --
  both hop equations, the coherent-compensation reduction, the noise
  variance, and both Remarks (window-causality, DF-terminology) -- moved
  to Ch.4.
- Ch.4 Section 4.1 (System Model) now states the complete model once:
  its existing "Hop Model" subsection gained the concrete per-symbol
  Rayleigh equations (previously only in Ch.1) ahead of its existing
  abstract H(.) operator notation, plus the coherent-compensation
  reduction and noise-variance equation; its "Relay Processing"
  subsection gained Remark (window-causality) right where the window w
  is defined; its "Relay Strategies" section gained Remark
  (DF-terminology) right after DF is introduced. The existing
  Eq. eq:relay-received-siso (which Appendix F's finding #9 specifically
  discusses) was left untouched to avoid disturbing that already-resolved
  finding.
- All labels carried over unchanged (labels are document-global), so no
  cross-reference elsewhere in the thesis needed updating -- including
  Ch.8's existing reference to the "Two-Hop Relay Model" section, which
  now resolves to its new home in Ch.4 via a second \label on the same
  subsection.
- Added as comment #33 in Appendix E (a later comment, not part of the
  original 32).

Verified: 0 undefined refs, 0 duplicate-label warnings, 159 tests,
verify_thesis_tables clean (no numeric tables touched), cold rebuild 158
pages (unchanged -- content moved, not added). Bundles rebuilt.

## Address the independent "not ready for submission" review

An external review report (not the Appendix F independent review already
in this thesis -- a separate, later document) judged the thesis "NOT
READY FOR SUBMISSION" across ~20 findings. Before acting on it, verified
several of its concrete claims against the current text; two were
factually wrong about the current document (the normalized-3K-parameter
study already holds window size constant at 11 across all seven
architectures, contradicting the review's claim that context window
confounds the model-size comparison; BPSK is stated as AWGN-calibration-
only consistently everywhere checked, contradicting the claimed
contradiction). Given the user's explicit instruction to adhere to the
feedback regardless, worked through every actionable item:

**Done (documentation/structural, no fabrication involved):**
- Revision-history material (REV/AK/GZ annotations) suppressed from the
  compiled output: main.tex's \AK/\GZ/\REV macros switched to no-ops
  (previously scaffolded as a commented-out "final mode" -- just needed
  enabling). Text remains in the .tex source for audit purposes but no
  longer renders in main.pdf or either Overleaf bundle.
- Appendix E (supervisor comments) and Appendix F (independent review
  findings) excluded from the compiled document via commented-out
  \include lines; files retained in the repository. One dangling
  cross-reference into the removed Appendix E (ch04, AWGN scoping
  sentence) rewritten to stand alone.
- New Appendix E: Master Experiment Ledger -- every major study's
  modulation, channel, table/figure, and exact result-file path in one
  table, addressing the "contradictory experimental descriptions /
  create a master ledger" finding directly.
- Single-seed training and unequal training budgets across architectures
  (VAE/cGAN epoch counts differ) added as an explicit Limitations item;
  the U-shaped complexity and architecture-vs-capacity claims are now
  scoped as "observations from the single trained instance," not claims
  shown robust to seed variance.
- Multiple-hypothesis-testing caveat added to the statistical-testing
  section: alpha=0.05 is stated as per-comparison, no Holm/Bonferroni/FDR
  correction applied across the many pairwise tests in this thesis.
- BER-optimal (BCJR/APP) benchmark for the unknown-ISI channel named as
  an explicit, concrete future-work item, sharpening what was previously
  a related but less specific point about sequence-ML vs.\ bit-BER
  optimality (already correctly caveated in Ch.7's own prose).
- Half-duplex capacity/goodput accounting clarified: goodput and the new
  Shannon-capacity reference (Table 42) are both per active-hop channel
  symbol; stated explicitly that expressing them over the full two-slot
  half-duplex cycle would halve every figure without changing which
  MCS/relay strategy wins.
- "Family-agnostic" / "without channel knowledge" generalization language
  tightened in Ch.7 and both abstracts: the relay generalizes across
  per-block realizations of the channel family it was trained on, not to
  channel families never presented in training -- this was already
  implicit in the body text but not stated at the abstract level.
- Reproducibility: stale "108 tests" corrected to 159; added a paragraph
  naming the repository and verify_thesis_tables.py as the source-of-
  truth mechanism, without hardcoding a commit hash into the text it
  would be typeset from.
- Minor terminology slip fixed: "re-modulates clean BPSK symbols" in the
  generic DF definition (applies to QPSK on the canonical channel).

**Explicitly not done, and not faked:**
- Multi-seed retraining with mean/std/CI across seeds -- real new
  experiments (retraining 9+ architectures repeatedly), not a documentation
  change; flagged as a Limitation instead of silently claimed.
- An actual BCJR/APP run as a new classical benchmark -- named as future
  work, not fabricated.
- Multiple-hypothesis correction actually applied to the existing test
  results -- would change reported p-values/significance calls without
  new data to justify it; documented as a caveat instead.

Verified: 0 undefined refs, 458 cells / 0 inconsistencies (verifier
unaffected -- no numeric tables changed), 159 tests. Cold rebuild:
**150 pages** (was 158 before this entry; 164 at the start of this
session), 0 undefined refs. Bundles rebuilt.

## Act on the automated PR review (Copilot) on PR #15

Three findings, all real, all fixed.

**1. Tail double-counted in the latency-budget frame length (genuine bug).**
`coded_latency_capacity.py:frame_symbols()` called
`PuncturedCode.n_coded_bits(n_steps)` after already computing
`n_steps = pc.n_steps(FRAME_INFO_BITS)`. But `n_coded_bits()` takes an
*information-bit* count and applies `n_steps()` itself, so the 2-bit tail
was added twice and every frame length came out 1-2 symbols too long.
Fixed to `pc.n_coded_bits(FRAME_INFO_BITS)`; `results/coded_latency_capacity.json`
and `results/coded_latency_capacity.png` regenerated (the script is a
re-derivation over already-measured FER data -- no new simulation, and
regenerating it before the fix reproduced the committed JSON exactly,
confirming determinism).

Effect on reported results: round-trip latencies drop by 1-2 symbols
(e.g. 16-QAM 2/3 block-DF 154 -> 152, QPSK 3/4 denoise 146 -> 145).
**No MCS selection flips and no goodput value changes** -- the 150-symbol
budget snapshot (Table 43) is numerically identical, and every claim in
Section~sec:coded-latency-capacity still holds, including "16-QAM 3/4
(136 symbols) barely clears the budget while 16-QAM 2/3 does not" (152 is
still over 150) and the 10.0x/2.0x gaps at 16 and 20 dB. Two quoted
latency figures in Ch.5 updated to match.

**2. Sentence fragment in Ch.2.** "...so no claim is made about it here.
and the results show that..." -- an inserted cGAN-scope caveat had been
dropped into the middle of an existing sentence. Reordered so the
benchmark sentence completes and the caveat follows it.

**3. `NaN` in `results/coded_error_mechanism.json`.** Not valid JSON per
RFC 8259 (Python's `json` reads it, strict parsers do not).
`coded_error_mechanism.py` now emits `None` -> `null` for `repaired_frac`
when the relay makes zero symbol errors and the ratio is genuinely
undefined; the committed JSON was patched in place rather than
regenerated, because regenerating it would re-run a Monte Carlo and churn
numbers the thesis already reports. Verified every other value is
byte-for-byte identical and the file now parses under a strict
(`parse_constant`-rejecting) reader. The verifier never read this field.

Verified: 458 cells / 0 inconsistencies (Table 43 re-checked against the
regenerated JSON), 159 tests, cold rebuild 0 errors / 0 undefined refs,
150 pages. Bundles rebuilt.

## Second independent review round ("still not ready for submission")

Five blocking items, four "major" items, and a set of presentation items.
Verified each against the source before acting.

**Blocking items -- all confirmed real, all fixed.**

1. *Obsolete canonical BER numbers in Ch.8.* Section 8.1.1 quoted
   MLP/cGAN/Hybrid $\approx$0.247 against DF 0.245, and Transformer/Mamba
   0.317--0.325. None of these appear in the canonical dataset, which reads
   (0 dB) AF 0.4076, DF 0.3337, MLP 0.3349, Hybrid 0.3329, VAE 0.3359,
   Transformer 0.4014, Mamba-S6 0.4071, Mamba-2 0.4032. The stale figures
   predate the SNR-convention correction; the paragraph also named the cGAN,
   which is excluded, and called the VAE "worse still" when it is in fact in
   the leading group. Rewritten from Table 2's 0 dB row. The qualitative
   claim (minimal feedforward relays capture the AF advantage, sequence
   models do not) survives and is in fact strengthened: the sequence models
   sit essentially at AF's level.

2. *Figure 50 captioned "on AWGN" in a Rayleigh chapter.* The figure is
   Rayleigh, not AWGN: its DF value at 20 dB is 0.03748, identical to
   Table 24's canonical-Rayleigh DF, and the decay is far too slow for AWGN.
   So neither of the reviewer's proposed remedies (regenerate, or delete)
   was right -- the data is correct and only the caption was stale. Caption
   rewritten. It also claimed the 4-class variants "plateau at BER 0.008";
   they reach 0.054 and are still descending, so that was corrected too, as
   was a truncated short caption ("16-QAM BER vs").

3. *Viterbi/BPSK claim technically wrong.* "For BPSK, each trellis branch
   carries exactly one bit, so sequence-optimal and bit-optimal detection
   coincide" is false: MLSE and bit-MAP are distinct criteria on any channel
   with memory, independent of bits per branch. Corrected in Ch.7 and in the
   Ch.8 future-work item that repeated it; the QPSK reversal is now reported
   as a measurement with the criterion mismatch named as an unevaluated
   candidate explanation pending the BCJR/APP run.

4. *H3 overstated.* Downgraded from "Confirmed" to "Partially supported"
   in the outcome table, the section heading, Ch.3, Ch.5 and Ch.9. The
   claim carried forward is the measured one -- larger evaluated models did
   not improve BER under the training protocol used. Stronger finding than
   the reviewer reported: the 11,201-parameter "Maximum MLP" said to show
   "clear overfitting" does not exist in any results file or in the
   experiments chapter, so that claim was withdrawn rather than softened.
   The bias-variance subsection is now explicitly framed as an
   interpretation, not a measurement.

5. *H4 overstated.* Renamed to an equal-parameter-budget comparison in
   Ch.3, Ch.4, Table 8's caption and the Ch.8 outcome table. Table 28 shows
   the budget was met by changing width, depth, state dimension and window
   size together -- MLP-3K sees an 11-sample window against its canonical 5,
   so the models are not even given the same input context. The comparison
   controls capacity, not architecture.

**Major items -- all confirmed, all fixed.**

6. Causal-window contradiction reconciled: Remark 4.1 asserted a $w>0$
   window "is not part of this system model" while the canonical MLP uses
   $w=2$. Rewritten to state that learned relays use a redundant, non-causal
   window everywhere as a fixed architectural input convention, that it is
   redundant under the canonical assumptions, and that the look-ahead cost is
   therefore only real in Ch.7.
7. Complexity claim corrected: per-symbol cost is fixed for a *deployed*
   architecture but the window grows with $L$ and the output head with $M$,
   so the contrast with MLSE is linear-vs-exponential growth, not constant-
   vs-growing. Fixed at all four sites.
8. Hybrid recommendation conditioned on a usable operating-SNR estimate,
   noting the threshold is fixed at the measured crossover and its
   robustness to estimation error was never evaluated.
9. Added the requested single-link caveat to Table 42's capacity column.

**Presentation items -- one real, the rest not reproducible.**

"high SNemerged" (real, fixed) and "(H2).." (real, fixed). But
"AComparative Study", "SystemModel", "AConstraint-Length" and "AComposite"
are **not** defects: the source reads "A Comparative Study", "System Model"
and "A Constraint-Length Sweep", and pdftotext extracts all of them with
correct spacing. These are artifacts of the reviewer's PDF extractor
dropping spaces at kerning boundaries. Adding spaces would have introduced
real errors, so no change was made.

Verified: 458 cells / 0 inconsistencies, 159 tests, cold rebuild 0 errors /
0 undefined refs, 151 pages. Bundles rebuilt.

## Scope reduction toward a 120-page budget: 16-QAM, Mamba-S6 study, AWGN study

Removed, per instruction, the three named non-supporting trails.

**16-QAM.** Chapter 6 (the 16-QAM extension) removed from the build; its
`.tex` is retained in the repo. Section 4.5 reduced to QPSK only: the
16-QAM constellation definition and the joint 2D $N$-class formulation are
gone, and the I/Q-splitting limitation is now stated as a general
"more than two levels per axis" caveat pointing at future work. The 16-QAM
constraint-length sweep (Table 36 and its figure) removed from Chapter 5.
Twenty-five narrative references to the extension chapter were rewritten
across Ch.1-5, 7, 8 and 9 rather than merely relinked.

*One deliberate exception, stated explicitly in the thesis:* 16-QAM
survives as a rung of the modulation-and-coding ladder in the
link-adaptation and latency studies (Tables 42, 43). Removing it there
would not be a scope cut but a different experiment -- the AMC study
selects 16-QAM at nearly every SNR at or above 16 dB, so a QPSK-only grid
would change every reported number and reduce "adaptive modulation and
coding" to rate adaptation alone. A scope note at the head of Section 5.4
draws the line: 16-QAM is not studied as a relay-architecture question,
only as a transmission mode the adaptation picks from.

**Mamba-S6.** The Mamba-S6-specific study is gone: H5 (the S6-vs-SSD
training-time crossover) removed from the hypothesis set, which now runs
H1-H4 plus H6; Section 8.3 (State Space vs.\ Attention) removed; the H5 row
removed from the outcome table; the two future-work and limitation items
that referenced the context-length benchmark rewritten. Mamba-S6 is
*retained* as one architecture among several in the canonical comparison
(Tables 2 and 8). Deleting one architecture's measured data point while
keeping its peers would be selective reporting, not a scope reduction;
removing its dedicated study is the scope reduction.

**AWGN.** Section 5.2 (the AWGN calibration baseline, E1) removed. The
simulator is still calibrated, on the canonical channel itself: symbol-wise
DF's measured BER against the closed-form two-hop composition (Table 2's
DF-theory column). AWGN remains as the noise term of the Rayleigh model and
in the closed-form background of Chapter 2, which is unavoidable and not a
study. The configurations table now lists QPSK/Rayleigh (canonical) and
BPSK/unknown-ISI (the extension's actual use of BPSK).

Verifier: `check_ber_validation`, `check_table26`, `check_table24` and
`check_table36` retired from the checks list (functions kept for
restoration); 407 cells / 0 inconsistencies.

**Page count: 151 -> 134.** Short of the 120 target by 14 pages. The three
named trails are now fully removed; the remaining gap cannot be closed from
them without breaking the AMC study as described above. Options for the
balance are prose condensation in Ch.2 (19 pp of background) and Ch.4
(19 pp of methods), or removing a further study.

## Further reduction: material not supporting the narrative (134 -> 128 pp)

Continuing the same criterion, with model-runtime comparison as the named
example.

- **Table 13 (model complexity and timing) removed.** Machine-dependent
  training/eval wall-clock per architecture, with the Mamba-S6 vs Mamba-2
  training times bolded -- the runtime comparison that H5 rested on. The
  verifier already treated it as informational rather than pass/fail. Three
  dependent references reworded (the cGAN exclusion note now gives the cost
  reason directly rather than citing a GPU-second figure).
- **Two orphaned conclusions removed** that still asserted removed studies:
  "Mamba-2 SSD trained 10.7x faster than S6 at longer contexts" and the
  16-QAM joint-2D finding, plus the 16-QAM row of the deployment
  recommendations table.
- **Sequence-model background condensed** (Section 2.4): the S4 recurrence
  derivation, the Mamba selective-scan internals and the SSD matrix algebra
  existed largely to motivate the chunk-parallel runtime claim. Replaced by a
  compact treatment that keeps the concepts, all citations, and one figure.
- **cGAN background condensed** (Section 2.3.2): six equations of WGAN-GP
  theory for a model excluded from every reported comparison, reduced to a
  paragraph that states the method, the instability it addresses, and the
  cost reason for its exclusion. The VAE subsection was left intact, since
  the VAE is evaluated.
- **Appendix C reduced** from "Software Architecture" to "Reproducibility":
  the UML component diagram and framework description removed, the testing,
  seeding and re-measurement material kept. Stale "108 automated tests"
  corrected to 159.
- **Two long discussion items condensed** where they restated their source
  chapters nearly in full: the H6 conclusion (3,984 -> 1,650 characters) and
  the coded block-DF future-work item (2,713 -> 1,365).

Verified: 407 cells / 0 inconsistencies, 159 tests, cold rebuild 0 errors /
0 undefined refs / 0 bidi errors, 128 pages.

## Reaching the 120-page target (128 -> 120)

Same criterion, continued.

- **Two supplementary complexity scatter figures removed** (complexity vs.
  low-SNR BER, low-SNR vs. high-SNR trade-off): both re-plotted data already
  tabulated in Tables 2 and 8, and both illustrated the U-shaped-complexity
  reading that H3's downgrade no longer supports. The now-empty "Results
  Figures" subsection was removed with them.
- **Three figures that only re-plotted the adjacent table removed** (their
  captions each began "The data of Table N plotted"): the coded-DF
  comparison, the QPSK constraint-length sweep and the soft-decision plot.
  The envelope and latency-budget figures were kept, since a step function
  is not readable off a table.
- **The constraint-length sweep folded into a paragraph.** Table 35 and its
  discussion answered "does a stronger code help?" with "no, not
  monotonically." The finding is retained in one paragraph with the closing
  figures and a pointer to the result file; the table and its expansion are
  gone. `check_table35` retired from the verifier.
- **Two sequence-model design rationales condensed**; both justified
  hyperparameters chosen for the removed runtime study.
- **Consistency fix found while cutting:** Section 2.2 still explained the
  U-shaped curve by overfitting, contradicting H3's downgrade. Rewritten to
  present it as the standard argument and defer to Section 7.2's weaker,
  measured claim. Similarly, the appendix's re-measurement note still listed
  the 16-class conclusion as one of the three that changed; now two.
- **A wasted page recovered in the Hebrew back matter.** The Hebrew abstract
  opened its language environments before `\chapter*`, so the environments
  started one page and the chapter heading immediately started another,
  leaving a page containing only the running header. Reordered so the
  heading comes first. Rendering verified visually: RTL layout, heading and
  justification unchanged.
- Small structural tidying to absorb two single-line spillover pages: the
  two duplicate "topology" scope items in Chapter 3 merged, the "training
  regime" bullet dropped (it is covered in the Limitations section), and the
  appendix's closing sentence about the verifier removed as a duplicate of
  the reproducibility paragraph in Chapter 4, which still names it.

**Page count: 120.** Target met. Verified: 389 cells / 0 inconsistencies,
159 tests, cold rebuild 0 errors / 0 undefined refs / 0 bidi errors.
