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
