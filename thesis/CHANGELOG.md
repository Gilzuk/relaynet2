# Changelog

Notable changes to the thesis. The in-document record is Appendix E (supervisor
comments and their resolutions) and Appendix F (independent review findings and
their resolutions).

Format follows [Keep a Changelog](https://keepachangelog.com/). Verification
claims refer to `verify_thesis_tables.py` at the repository root, which checks
every table cell and quantitative prose claim against the data files that
produced it.

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

### Reverted

- **The AWGN nine-relay companion comparison was withdrawn.** It had been added
  to Section 5.3 alongside the canonical Rayleigh comparison, but it
  contradicted Appendix E's statement that "the AWGN and Rician robustness
  studies ... were all removed" in answer to comment 4 — the comment the
  appendix itself calls the decisive one behind the restructure. Table 5.5,
  Figure 5.5, the surrounding discussion, the Chapter 1 wording changes, the
  `check_table2awgn` verifier check and the companion figure are all removed;
  AWGN returns to its calibration-only role (Section 5.2) and the canonical
  BPSK/Rayleigh comparison (Table 5.4) is untouched. Verification returns to
  **250 cells, 0 inconsistencies**.

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
