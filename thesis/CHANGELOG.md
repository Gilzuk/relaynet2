# Changelog

Notable changes to the thesis, newest first. The response to the October 2026
review is `submission/reply_to_khina_2026-10.md` (one paragraph per comment, with
the thesis location of each fix) and the task chain behind it is `HANDOVER.md` at
the repository root. Earlier rounds were recorded in review-response appendices
that the current build no longer includes.

Format follows [Keep a Changelog](https://keepachangelog.com/). Verification
claims refer to `verify_thesis_tables.py` at the repository root, which checks a
declared subset of table cells and prose numbers against the data files that
produced them; it does not cover every number or figure.

---

## [Independent recheck, combined with the author corrections] — 2026-10-12

From draft #171 (another session's independent recheck), reviewed and combined
with #172:

### Changed
- **Figure 5.3 caption.** It could be read as giving the matched comparison 100
  trials; it now says the figure has 100 paired trials and Table 5.9 is the
  separate 30-trial matched comparison.
- **Sec. 5.5.2.** Discloses that the 30-trial budget was retained after the
  running means were inspected (as `coded_matched_metrics.py` records), so the
  paired tests are exploratory.
- **Reply, comment 4.** States the 30-of-100 trial subset and the disclosure;
  "every SNR from 0 to 20 dB" becomes the six tested points; "longer constraint
  lengths help" becomes the measured K = 3/5/7 ordering from 4 dB up with the
  0 dB exception (K = 5 and 7 tie at zero errors at 16 and 20 dB); "block DF
  reaches the oracle" becomes equal zero-error counts in 6,000,000 bits at 24,
  28 and 32 dB, which do not establish equal BER.
- **Repository.** `docs/independent_khina_audit_2026-10-12.md` maps the nine
  comments to evidence; `docs/thesis_feedback.md`, `instructor_response_plan.md`
  and `professor_feedback.md` are marked historical (the last ends with a
  "[Professor's Name]" placeholder and is no endorsement).

The body still ends at 119 countable pages.

---

## [Author corrections: bit-wise MAP, DF baseline, Transformer count] — 2026-10-12

### Changed
- **Sec. 4.1.5 (hard-relay objective).** The BER-optimal relay target is bit-wise
  MAP. It coincides with symbol-wise MAP for BPSK, or for QPSK with a posterior
  that factorizes across its bit axes; a joint four-class QPSK arg-max minimizes
  symbol error, not generally BER. Cross-entropy learns the class probabilities
  but does not make the joint arg-max BER-optimal.
- **Sec. 6.4.1 (MMSE baseline).** Outperforming the zero-threshold DF does not by
  itself show a benefit from temporal context: the scalar MAP control removes its
  high-SNR floor without a window, while the windowed MLP reaches far lower BER
  (Sec. 6.2).
- **Reply, comments 1 and 8.** Same bit-wise MAP statement. A feedforward width of
  128 gives 26,017 parameters for the full model; the 25,408 in the review covers
  the two encoder blocks only, excluding the 609 input and output projection
  parameters (both checked on the instantiated model).
- **README.** Page count stated as 119 numbered pages (135 PDF pages).

The body still ends at 119 countable pages.

---

## [Figures checked against their data] — 2026-10-12

Every figure was checked against the data it plots by regenerating the 16 that
have a generating script from the committed files and comparing them with the
included images. Twelve matched. Fixed:

### Fixed
- **Figures 5.3–5.5 (coded link)** had been drawn from earlier runs (Figures
  5.3 and 5.4 printed 20 and 10 trials; Figure 5.5 printed no count) while Tables 5.8, 5.12 and 5.13
  report the committed 100-trial runs. Redrawn from those runs with
  `scripts/plot_coded_df.py`; figures and tables now show the same values.
  Figure 5.5's title and axis now name the idealized buffer-readiness budget,
  as its caption does, instead of a latency budget.
- **Table 5.8, 20 dB soft block-DF.** The cell showed the doubled-variance
  sensitivity run (0.001376) and its footnote called it the run Figure 5.3
  plots; the figure plots the nominal run. The cell is the nominal run again
  (0.002348), the doubled-variance value is a footnoted sensitivity check, and
  the Figure 5.3 caption and the following paragraph say so. This reverses the
  part of the C5 fix (#151) that changed the cell.
- **Figure 6.3 (QPSK unknown channel)** still carried the ISI → Rayleigh hop-2
  panel removed with its table row; it now shows the AWGN case only
  (`scripts/plot_e6_qpsk_unknown_channel_awgn.py`), at full width in the style
  of Figure 6.2. Its title had called the first hop "unknown ISI"; it names
  the per-symbol fading of Eq. (6.2), and the axis is per-hop Es/N0. The
  caption no longer says the MLP "separates" from the taps-only trellis
  from 2 dB: it lies just below it there (by at most 0.0046 up to 12 dB) and
  the gap opens above 12 dB.
- **Figure 6.7(a)** had no generating script; `scripts/plot_e6_complexity_cost.py`
  now draws it from the counting rule of Section 6.5. The network is labelled as
  the deployed 170-parameter relay (it read "MLP-169") and drawn as a reference
  line rather than an identical bar at every (M, L), which Section 6.5 says it is
  not; the y-axis label is no longer clipped. Panel (b) is unchanged, since its
  timings were never committed; its caption and Section 6.5 now say that it
  timed the 169-parameter I/Q relay, and give the unit.
- **Figure 5.1's title** no longer carries "(§7.10)", a section number internal
  to `run_experiments.py`.

The body still ends at 119 countable pages.

---

## [Khina revision round] — 2026-10-12

Response to Dr. Anatoly Khina's nine comments and presentation notes on the
120-page final-revision PDF: PRs #131–#164. The body (Chapter 1 through the
appendices) ends at 119 pages, within the faculty's 120-page limit (135 PDF pages
with the front and Hebrew back matter). Section and table numbers below are
those of that build; the appendices are numbered 9.1–9.6 (their letter names,
which began at B, were dropped).

### Changed — conclusions
- **Coded relaying (comment 4).** With reliability-weighted (matched) decoder
  metrics, block DF has lower BER than both learned relays at every SNR from 0 to
  20 dB, in 30 of 30 paired trials, under either destination metric. The
  published high-SNR advantage of the learned relays came from the unweighted
  metric. New Sec. 5.5.2 and Table 5.9; matched values beside the published ones
  in Tables 5.6, 5.10, 5.12 and 5.13; the reversal is stated in Ch. 1, Ch. 7,
  Ch. 8 and both abstracts.
- **AF (comment 3).** The learned relays' low-SNR advantage is over the
  implemented block-normalized AF. Against a conventional variable-gain AF the
  0 dB margin is 0.007, not 0.073 (Secs. 4.1.3, 5.2, 7.1.1, Ch. 8, abstracts).
- **Flat controls.** Hop-2 noise corrected and rerun; below 8 dB the
  unknown-phase MLP trails differential detection by up to 0.0106 at 0 dB
  (see `RERUN_CHANGELOG.md`).

### Added
- Sec. 4.1.5, the relay-design problem and its training surrogates (comment 1).
- Sec. 3.1.1 hierarchy paragraph and Table 3.1 mapping each research question
  and hypothesis to the experiments that answer it; O1 and O2 answered by name in
  Ch. 8 (comment 2).
- Exact I/Q factorization of QPSK MLSE, checked on 264,000 decisions. The cost
  comparison in Sec. 6.5 gives the factorized count beside the joint one, the
  Fig. 6.7 caption notes it, and Sec. 2.6 states the square-QAM case (comment 5).
- Appendix G (Sec. 9.6, Table 9.7): channel models, amplifier, normalization,
  pilots, windows, DBPSK and the single SNR-penalty definition; Appendix E
  size-sweep configuration and Table 9.3 (comment 7).
- Sec. 2.2.4 positioning against estimate-and-forward, Farsad–Goldsmith and
  ViterbiNet; Table 3.2 compares named studies; a non-monotonic scalar MAP relay
  is measured on the ISI channel (Sec. 6.2) (comment 9).
- AF block-length sensitivity and a conventional finite-power AF (comment 3).

### Fixed
- **Delay accounting (comment 6):** detector look-ahead, normalization
  buffering, protocol scheduling and end-to-end delay are distinguished
  (Tables 5.11 and 6.8, Sec. 7.3.1); operational-readiness claims removed.
- **Architecture comparisons (comment 8):** Transformer count 17,697; tanh/ReLU
  protocol qualified; initialization spread 0.39 / 0.07 / 0.03 dB; equal-budget
  conclusions tied to the training protocols, with no claim to isolate
  architecture.
- **Methods described as the code runs:** canonical relays trained once on
  BPSK over AWGN and applied per axis; Adam for every relay; sample counts
  (Sec. 4.3.3, Eq. 4.16). The Hybrid compares a fixed 5 dB threshold with a
  per-axis received-power estimate, so its switch falls between 6 and 8 dB
  (Sec. 4.2). The sequence models hold out 15% and keep the lowest-training-loss
  epoch. Appendix B (Table 9.1) and the test counts corrected.
- **Remaining-alternatives statements:** Ch. 7 (comparator scope) and the Ch. 8
  closing paragraph no longer list reliability-weighted decoding as unmeasured;
  what remains unmeasured is a destination metric that also models relay
  decision errors (#162).
- **Presentation:** Chapter 5 running headers; ledger cites tables and figures,
  not labels; verification-coverage statements agree (Sec. 4.3.4, Appendices C
  and D); Table 5.8 shows the corrected value its figure plots; coded captions
  name the unweighted metric.

### Fixed — final page-by-page review (#164)
- The body had grown to 122 pages against the 120 limit; redundant passages were
  cut where Chapters 2, 7 and 8 ended with two or three lines on their last page,
  and the duplicate commentary on Table 5.2 was removed (119 pages).
- Table 5.13 ran off the page; Tables 5.3, 5.9, 5.10, 5.11 and 9.3 overflowed.
- Sec. 4.2 still described a sqrt(2) rescaling of each axis before the MLP,
  contradicting Eq. 4.16 and the code; the Transformer rationale gave a
  feedforward width of 128 against 64; the cGAN's lambda_L1 is 20, not 100;
  Eq. 2.9 used the window half-width w for the input vector.
- DF closed-form agreement: 1.0% / 2.5% / 3.6% from the printed values; Table
  5.12's caption sentence order; boundary wording that contradicted Sec. 3.1.1;
  "earlier drafts" and similar revision language; preprint identifiers in the
  bibliography.

### Verification
`latexmk -xelatex`: 0 errors, 0 undefined references, 0 overfull boxes, 119
countable pages.
`verify_thesis_tables.py`: 678 cells, 0 inconsistencies. `provenance_audit.py`:
clean. `pytest`: 347 passed. Mirror `relaynet2-thesis` at 162fb85 (from main
1eaeb08), `thesis.pdf` byte-identical to `thesis/main.pdf`.

## [AWGN companion removed from Chapter 5] — 2026-08-17

### Removed — the AWGN relay comparison in the canonical section

Chapter 5's canonical section carried three tables: the canonical Rayleigh
BPSK comparison, an AWGN companion comparison on a lean relay set, and
QPSK-versus-BPSK on Rayleigh. The AWGN companion is now gone, and
QPSK/Rayleigh takes its place as the canonical section's second table.

Removed: `tbl:table2awgn` and its two paragraphs, and `fig:fig10awgn`
(`results/awgn_comparison_ci.png`, which stays in the repo but is no longer
referenced). The §5.3 configuration line no longer advertises an AWGN
companion and now records the modulation as BPSK and QPSK.

**AWGN's remaining role is calibration only.** Section 5.2 is unchanged: the
theory-versus-simulation validation still runs on AWGN, because the closed
forms the simulator is checked against are AWGN expressions. What is gone is
any *relay* measured on AWGN in the canonical chapter. Appendix E's response
to comment 4 is updated to say exactly that, since it previously cited the
now-removed companion table as part of AWGN's bounded role.

### Changed — verifier

`check_table2awgn` removed from `verify_thesis_tables.py` along with its
entry in the source map and the check registry. Suite goes from 352 to
**327 cells, 0 inconsistencies** — the 25 lost cells are the companion's.

### Verification

Cold `latexmk -xelatex`: exit 0, **146 pages** (was 149), **0 undefined
references**. `verify_thesis_tables.py`: 327 cells, 0 inconsistencies.

---

## [Canonical restructure: Rayleigh with BPSK and QPSK] — 2026-08-17

### Changed — QPSK on Rayleigh moved into Chapter 5 as canonical; AWGN demoted to baseline

The previous arrangement paired BPSK with AWGN in Chapter 5 and put QPSK on
Rayleigh in Chapter 6. That made the canonical chapter's headline comparison
run on the one channel the thesis explicitly does *not* draw conclusions from
(Appendix E, comment 4), while the canonical channel's second constellation
sat in an extension chapter. The two are now swapped:

- **Section 5.2** is retitled *Simulation Baseline: AWGN Calibration*. Its
  role is stated up front: AWGN is where closed forms exist, so it is what
  the simulator is validated against, and the relay comparison reported there
  (Table 5.5, lean set: AF / DF / MLP / Transformer / Mamba-2 at 0–8 dB) is a
  baseline, not an operating point. No relay is ranked on it.
- **Section 5.3** is retitled *Canonical Relay Comparison (SISO, Rayleigh,
  BPSK and QPSK)* and now carries both constellations on the canonical
  channel. The QPSK-vs-BPSK comparison moved here from Chapter 6 and is
  Table 5.6.
- **Chapter 6** is retitled *Extension: Multi-Level Modulation (16-QAM)* and
  keeps only the constellation that genuinely requires a different relay
  formulation. Its former QPSK block is replaced by a pointer to Section 5.3.

### Changed — the canonical-setup statement, in all seven places that assert it

`ch01_introduction.tex` (scope paragraph, scope-table caption, Channel row,
Modulation row, and the E1/E2/E4 rows of the experiment summary),
`ch03_objectives.tex`, `ch04_methods.tex`, `ch08_discussion.tex` and
`ch09_summary.tex` all defined the canonical setup as BPSK-only; each now
reads "complex baseband, BPSK and QPSK". Appendix E's response to comment 4
states why this is still *one* setting rather than two: topology, channel
model, and metric each remain single-valued, and QPSK is the two-axis use of
the same complex baseband processed by the identical relays.

### Verification

Cold `latexmk -xelatex`: exit 0, 149 pages, **0 undefined references**.
`verify_thesis_tables.py`: **352 cells, 0 inconsistencies**.
`pytest tests/`: **119 passed**.

### Known follow-ups (not done here)

- Chapter 6's AWGN tables (`tbl:table14`, `tbl:table15`, `tbl:table24`) are
  still on the pre-correction AWGN convention and need a re-run.
- The Chapter 7 flat-channel control passes by 0.0097 against a 0.010
  tolerance — too thin a margin; it needs a larger trial budget.

---

## [SNR convention correction] — 2026-08-15

### Fixed — a 3 dB channel convention error, found by checking against literature

The Rayleigh channel used by Chapter 7 put the whole of N0 into its single
real noise dimension instead of N0/2. The channel was therefore 3 dB
pessimistic: it produced `0.5*(1 - sqrt(g/(2+g)))` where the textbook
expression (Proakis, with g = Eb/N0) is `0.5*(1 - sqrt(g/(1+g)))`.

The visible symptom was that Chapter 7's row labelled "control: canonical
Rayleigh" did **not** reproduce Chapter 5's canonical numbers, despite the
thesis presenting them as the same channel: DF read 0.1208 there against
0.068 in Table 5.4 at the same 8 dB. Both simulators were internally
consistent, which is why nothing flagged it; they simply implemented
different models. Chapter 5's (`relaynet/channels/fading.py`, zero-forcing
equalization with complex noise) was the one matching literature.

Eight channels in `relaynet/channels/e6_channels.py` carrying real-valued
noise are corrected to `sigma^2 = N0/2`, putting the whole thesis on the
Eb/N0 axis. After the fix and a re-run, Chapter 7's control DF agrees with
Chapter 5 and with theory at every point (0.0687 / 0.0683 / 0.0684 at 8 dB).
AF still differs between the two chapters, which is expected rather than a
defect: DF's `sign()` is scale invariant and so depends only on effective
SNR, whereas AF forwards the analog waveform and is sensitive to whether the
noise arrives through ZF amplification or magnitude compensation.

`tests/test_snr_convention.py` pins every BPSK channel to its closed form so
this cannot recur. The previous test asserted only that
`sigma = 10**(-snr/20)` and `noise_power = P/10**(snr/10)` are the same
expression, which they are; it never checked how many dimensions that power
was spread over, which is where the 3 dB went. The new tests fail by 0.0288
against a 0.004 tolerance under the old convention.

### Chapter 7 re-run on the corrected channels

Every Chapter 7 study has been re-run and its numbers re-transcribed: the main
unknown-ISI table, the flat control, composite, blind, partial-posterior and
the QPSK generalization. Table 7.1 and the composite prose moved substantially
(the verifier flagged 36 stale values, now zero). The QPSK study is unchanged,
as expected: its channel uses complex noise and was already Es/N0-consistent,
so it was never on the wrong axis.

The chapter's conclusions are unaffected, which is the point of the exercise.
The 0.25 memoryless-relay floor is a property of the ISI tap structure rather
than the noise scaling, and it is now approached more closely than before
(DF reaches 0.246 at 20 dB against 0.234 previously). H6 stands, the MLP still
restores reliable relaying where AF and DF fail, and genie-CSI Viterbi remains
the bound.

Most importantly, Chapter 7's "control: canonical Rayleigh" row now reproduces
Chapter 5: DF reads 0.0687 against Table 5.4's 0.0683 at 8 dB, where before it
read 0.1208.

### Known issues — follow-up required

### Chapter 5's AWGN companion regenerated on a lean relay set

`results/bpsk_comparison/awgn.json` is re-run on the corrected Eb/N0 axis with
a lean relay set: AF and DF, plus one representative of each learned family
(the minimal MLP, the Transformer, Mamba-2 SSD). Hybrid, VAE, cGAN and
Mamba-S6 are omitted via the new `--skip-relays`, because the cGAN alone is
7,293 s on CUDA and this machine has four CPU cores. The run took 56 minutes.

Omitting a relay drops its column rather than carrying forward a value from
the earlier 3 dB-offset axis, so the table is narrower but internally
consistent. Table 5.4, the canonical Rayleigh comparison, keeps all nine
relays and is untouched: the Rayleigh channel was already on the Eb/N0 axis.

The regenerated table is a direct check on the calibration. Symbol-wise DF
now reproduces the closed form 2P(1-P), P = Q(sqrt(2Eb/N0)), to within Monte
Carlo noise at every point, agreeing to 0.00038 against 0.00038 at 8 dB.

The table is cut at 8 dB rather than 12. On the corrected axis the AWGN BER
falls faster, so the resolution floor bites sooner: at 10 dB the estimates
rest on about two errors in 100,000 bits and by 12 dB every relay is at the
floor. 8 dB is the last point carrying a meaningful number of errors (38).

A caution worth recording: experiment 7.2 writes both `awgn.json` **and**
`rayleigh.json`, so the lean re-run silently reduced the canonical Rayleigh
file from nine relays to five. It was restored from a backup taken
beforehand. Anyone re-running a subset must check which files an experiment
touches, not only the one they mean to refresh.

### Known issues — follow-up required

- **Chapter 6's AWGN tables have not been regenerated** (`tbl:table14`,
  `tbl:table15`, `tbl:table24`). They still carry data on the earlier AWGN
  convention. Chapter 6's Rayleigh table (6.2) is unaffected.
- **The flat-channel control passed only narrowly.** Its largest shift after
  the correction was 0.0097 against a 0.010 Monte Carlo tolerance, so it
  verifies clean by a margin thinner than is comfortable. Worth re-running at
  a larger budget, or tightening that tolerance, before submission.

---

## [Overleaf import + fixes] — 2026-08-15

The Overleaf project, carrying the accumulated supervisor iterations, was
imported as the authoritative version, and the outstanding fixes were applied on
top of it. Imported first as a clean restore point (`de1ba61`), then fixed, so
the two steps can be separated.

### Added

- **Appendix F**, absent from the Overleaf version, restored as
  `chapters/appendix_f_review.tex` and included after Appendix E. All 23 of its
  cross-references resolve against this version.
- **A QPSK unknown-channel subsection** (`sec:qpsk-unknown-channel`, Table 7.3,
  Figure 7.3), reporting the results of `e6_qpsk_unknown_channel.py` (10 x
  100k, 2026-08-15). Repeats the BPSK unknown-ISI study of Section 7.1 with
  Gray-coded QPSK: the memoryless-relay failure mode generalizes (AF/DF
  plateau at an elevated, non-monotonic BER), but the BPSK ordering between
  the learned relay and genie-CSI Viterbi MLSE does not — MLP-QPSK ends up
  *below* Viterbi from about 2 dB upward, a reversal attributed to the
  sequence-vs-bit-error-rate distinction in MLSE and flagged explicitly as an
  unverified hypothesis, not a proven claim. Verified against the data source
  (`verify_thesis_tables.py`, `tbl:tableE6qpsk`, 32 cells, 0 mismatches).

### Changed — from the Overleaf version

- Chapter files are split and renamed to match their numbers:
  `ch05_experiments`, `ch06_..._Higher_Order_Modulation` and
  `ch07_unknown_and_mismatch_channels` are separate files, and the discussion
  and summary are `ch08`/`ch09`.
- Substantially expanded text in every chapter, and twelve figures the
  repository did not have, including split-out panels for the partial-posterior
  and complexity studies.
- The six hardcoded cross-references ("Table 15", "Table 24", "Table 13",
  "Section 3.7.6") are resolved to `\ref`.

### Changed — Chapter 7 sub-studies rerun at full scale; a real CMA bug fixed

The composite, blind, and partial-posterior studies, previously left at their
original 5-6 trial development budget and merely disclosed as such, were rerun
properly. Two genuine defects surfaced and were fixed rather than reported as
findings.

- **Blind equalizer (CMA) divergence — a real bug.** The CMA used a fixed,
  unnormalized step size. Its error term is cubic in the equalizer output, so a
  fixed step is a positive-feedback loop: verified directly, the weights stay
  bounded over 40,000 samples (max |w| = 1.50) but overflow to infinity over
  100,000. A first rerun at 10 x 100,000 duly produced a CMA BER of 0.128 at
  20 dB against the previously reported 0.0024 — a number that would have been
  written up as "CMA degrades" when it was in fact an implementation fault. The
  step is now NLMS-normalized by the input-segment energy, which is the standard
  formulation and a strictly stronger baseline; it is stable at every block
  length tested. Disclosed in the text.
- **Trials are channel draws, not just samples.** The blind and
  partial-posterior studies redraw the ISI/amplifier/phase realization for every
  block, so a trial is one draw from the impairment family. Raising bits per
  trial sharpens the estimate for one particular channel while leaving the
  ensemble average as noisy as before; only more trials help. These two studies
  therefore run at **50 x 20,000** (1,000,000 bits over 50 channel draws, ten
  times the previous ensemble) rather than the nominal 10 x 100,000. The
  composite study, whose channel is fixed, uses the standard **10 x 100,000**.
  Every experiment now has M >= 10 trials, so the Wilcoxon caveat that
  previously excluded these studies no longer applies (Section 4.6.1).

Results that changed, all updated in the text and re-verified:

- The pilot-budget crossover moved. Pilot-aided Viterbi was reported as beating
  the pilot-free MLP "down to approximately 10 pilots"; it now loses its edge at
  10 pilots (0.0545 against the MLP's 0.0487) and the crossover sits between 20
  and 10 pilots. The 5-pilot collapse persists (0.1235 +/- 0.0304).
- **An open point is now closed with data.** Whether blind CMA fails to converge
  at short block lengths was previously "left as an open point rather than
  asserted". Measured per block, CMA gives 0.1723 at L=40 and only 0.1653 at
  L=1000, against 0.0645 with a 20,000-symbol block: it does not converge within
  blocks of a thousand symbols or fewer.
- Viterbi and the MLP are now statistically indistinguishable in payload BER at
  every block length (both ~0.049), sharpening the panel-(b) argument from
  "similar BER" to "identical BER at a quarter less overhead".
- The composite MLP-169 figure at 8 dB moved 0.130 -> 0.126, and the claim that
  the 1,153-parameter network was "slightly worse at high SNR, consistent with
  mild overfitting" is withdrawn: the two now end identical (0.0050) with
  differences in both directions inside the confidence intervals. H3 is still
  supported, by a cleaner argument.
- The blind-regime CI comparison was requoted at 8 dB from the new data
  (+/-0.0348 blind MLSE, +/-0.0046 MLP, +/-0.0135 CMA); blind MLSE remains the
  clear instability, but the earlier +/-0.164 reflected a 5-draw ensemble.

- **`scripts/plot_e6_studies.py` added**, regenerating all four Chapter 7 study
  figures from the committed `.npy` files. These figures previously had no
  regeneration path in the repository, so a rerun could not be reflected in the
  document; that gap is now closed.
- Verification extended to the rewritten prose, with the three prose checks
  nearly tripling their coverage (blind 5->9, partial 4->13, composite 3->10)
  at a 5x tighter tolerance.

### Fixed — closed 3 outstanding supervisor (AK) follow-up comments

Three second-round AK pushbacks, nested inside earlier `\REV` resolutions,
had never received a reply and were still open in the compiled document:

- **"AF outperforms DF at low SNR"** (Abstract): checked against this
  thesis's own data (Table 5.4, canonical Rayleigh and its AWGN
  counterpart) — DF's per-symbol slicing has a strictly lower BER than AF
  at every SNR point evaluated (0–20 dB, both channels); no crossover in
  the range studied. Resolved with the data, while acknowledging the
  general (naive-DF-vs-selective-DF) point holds in the wider literature.
- **"You cannot refer to a model that you've never mentioned"** (Abstract):
  the sentence introducing Viterbi MLSE named the impairment inline (an
  unknown intersymbol-interference filter) instead of relying on a forward
  reference to Chapter 7, so the abstract is now self-contained.
- **"Either drop this or define the competing model"** (Chapter 1, window
  realizability remark): kept, scoped explicitly as an architectural
  convention only, with a forward pointer to Chapter 7's now-complete
  competing-model definition (including the new QPSK generalization,
  Section 7.1.2) rather than duplicating it in Chapter 1.

### Fixed — Monte Carlo scale audit

- **Stale trial-count text**: the main unknown-ISI study's "Trials" bullet and
  two table captions (Table 7.1 `tbl:tableE6`, Table 7.2 `tbl:tableE6flat`)
  said "5 trials $\times$ 50,000 bits", left over from before a 2026-07-14
  rescale (`4928e65`) that the Overleaf import never picked up. The
  underlying data was already at the current 10 $\times$ 100,000 scale (the
  numeric cells were correct and verified throughout); only the stated
  methodology was wrong. Corrected in three places.
- **Disclosed a genuine reduced-scale gap**: the composite (Section 7.1.3),
  blind (Section 7.1.4), and partial-posterior (Section 7.1.5) sub-studies
  really do run at 5, 5, and 6 trials $\times$ 40,000 bits respectively, not
  the chapter's 10 $\times$ 100,000 standard. This is real, not a text bug:
  `e6_composite_ported.py`, `e6_blind_ported.py`, and `e6_partial_ported.py`
  all carry a `# standalone's own dev budget` scale. Rather than rerun (which
  would shift every quoted number in the affected prose and requires
  reconstructing plotting code no longer in the repo for their 4 figures),
  each subsection and figure caption now explicitly states its actual trial
  count and bit budget. All three already used the correct 95% CI formula
  ($1.96\,\sigma/\sqrt{n}$); only the trial count $n$ was undisclosed.

### Fixed — applied on top of the import

- **`tbl:table8` had lost its label**, so the normalized-3K table was silently
  unverifiable: coverage had dropped from 200 cells to 158. Label restored;
  the table verifies again.
- **Figure 6.5's caption described a visual key the figure does not use** —
  "dashed lines = tanh/BPSK baseline, solid = linear/QAM16, dotted =
  hardtanh/QAM16", where the plot distinguishes activations by colour and
  marker in the legend, not by line style. Rewritten to describe what is
  actually plotted; the quantitative claims it makes were checked against
  Table 6.2 and are correct.
- **Package ordering**: `ragged2e` and a second `lineno` had been inserted after
  `polyglossia`, contrary to the note in the preamble. `bidi` (loaded by
  polyglossia) errors on packages loaded after it, and `latexmk` — which
  Overleaf uses — treats those errors as fatal and stops rerunning, leaving
  every cross-reference as `??`. `ragged2e` moved before the language block, the
  duplicate `lineno` removed (it was already loaded earlier), and zero packages
  now follow `polyglossia`.
- **Test count**: both Appendix C and Chapter 4 claimed 126 automated tests; the
  suite collects 108. Corrected, and Appendix C now records that the suite needs
  only NumPy and SciPy.
- Remaining em dashes in the compiled chapters converted to conventional
  punctuation (8 to 0).

### Known issues

- The VAE relay decodes stochastically at inference and the deterministic
  variant was not evaluated, so its weak result may be an inference-time
  artifact. Stated as open in the text.
- Blind CMA was not tested at the shortest block length; the corresponding claim
  is narrowed accordingly.
- Two references (Bergel 2023 / ICASSP 2024, Akdemir et al. 2024) are unverified
  against primary sources.
- `results_old/` from the Overleaf archive was not imported: 14 MB of superseded
  outputs already recoverable from git history.
- `thesis/main.pdf` is not rebuilt in this environment, which has no LaTeX
  installed. Regenerate with `make thesis`.

---

## [Major revision] — 2026-07-31

Restructured around a single canonical setup, all supervisor comments addressed,
and all findings from an independent technical review resolved.

### Changed — scope and structure

- Fixed **one canonical setup** and varied only the relay function: two-hop
  SISO, i.i.d. Rayleigh fast fading, complex baseband, BPSK, uncoded BER.
- Reframed the central claim from "learned relays replace classical processing"
  to "characterise when learning helps". On a matched, known channel,
  symbol-wise DF is at least as good at zero parameter cost.
- Chapter 2 became a literature review only, with no experimental results in it.

### Removed

- AWGN and Rician channel-robustness studies; the 2x2 MIMO second hop and its
  ZF/LMMSE/SIC equalization; 16-PSK; the CSI-injection ablation; the end-to-end
  autoencoder; the diversity–multiplexing tradeoff discussion. All redesignated
  as future work.
- A channel-validation subsection duplicating the Chapter 5 experiment, a
  Rician fading-distribution subsection, two duplicated figures, and a
  positioning table duplicating Chapter 3.

### Added

- **Appendix E** (supervisor comments and resolutions) and **Appendix F**
  (24 independent review findings).
- Remarks formalising symbol-wise versus block DF, and the realizability of the
  non-causal relay window.
- Automated verification of every table and quantitative claim against its data
  source.

### Fixed

- AF end-to-end SNR attribution and its variable-gain assumption; an overclaim
  about exact representation of the posterior; the Hurwitz parameterization
  underpinning the Mamba derivation; a relay observation equation that showed
  the AWGN case rather than the Rayleigh channel; the DF error-composition
  formula; and a pilot-budget effect mischaracterised as identifiability rather
  than estimation variance.
- One reference had the wrong author, title and pages (corrected to Samuel,
  Diskin & Wiesel, *Learning to Detect*, IEEE TSP 2019); two duplicate entries
  merged; the Mamba architecture repointed from the S4 paper.
- Six table cells corrected against their data sources, and one figure
  regenerated after it was found to derive from a superseded run.
- A package-ordering fault that broke compilation on Overleaf, where every
  cross-reference rendered as `??`.
- The English abstract, silently absent because `main.tex` included a
  nonexistent file.
- 142 em dashes replaced with conventional punctuation, and seven asides
  delimited by paired colons.

## [Supervisor-comment consistency pass] — 2026-08-15

Audited all 17 supervisor (AK) comments against the current document rather
than against the appendix's own account of them. Sixteen held; the exceptions
were bookkeeping, fixed here.

### Changed — channel pairing per author direction

At the author's direction the modulation/channel pairing is now explicit in
both experiment chapters:

- **Chapter 5 keeps the AWGN companion** (Table 5.5, Figure 5.5) alongside the
  canonical BPSK/Rayleigh comparison. Because Appendix E had told the
  supervisor that "the AWGN and Rician robustness studies ... were all removed"
  in answer to comment 4, AWGN is given an explicitly bounded role instead of
  being reinstated silently: it is the analytical calibration limit plus a
  BPSK-only companion comparison, no hypothesis is tested on it and no relay is
  ranked by it, and the canonical setup remains Rayleigh. Appendix E, the
  Section 5.3 discussion and the Chapter 8 future-work item all state that role
  identically. Rician remains removed outright.
- **Chapter 6 gains QPSK on the canonical Rayleigh channel** (Table 6.2). Its
  modulation comparison had been AWGN-only while every other conclusion is
  drawn on Rayleigh.

### Fixed

- **Appendix E cited the wrong abstract include paths** (`ch00_frontmatter.tex`,
  `ch09_hebrew_abstract.tex`); `main.tex` includes `chapters/frontmatter.tex`
  and `chapters/hebrew_abstract.tex`. Corrected.
- **The Overleaf merge re-added removed-study material.** Figures for the CSI
  injection study, end-to-end autoencoder, 2x2 MIMO and Rician study returned,
  contradicting Appendix E's "Summary of removed material". None was referenced
  by a compiled chapter, so the document was never wrong, but the sources were
  misleading. Removed; recoverable from git history.
- **Five superseded chapter drafts** returned alongside their replacements
  (`ch06_discussion`, `ch07_summary`, `ch09_hebrew_abstract`, `experiments`,
  `research_objectives`), none included by `main.tex`. Renamed with the leading
  underscore this repository uses for drafting history.
