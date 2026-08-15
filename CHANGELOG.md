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

### Added — AWGN companion comparison (Section 5.3)

Table 5.5 (`tbl:table2awgn`) and Figure 5.5 report the same nine-relay
comparison as the canonical Rayleigh table/figure, but on AWGN, at the
identical $10\times10{,}000$ budget. The data (`results/bpsk_comparison/awgn.json`)
already existed and was previously used only for theory-vs-simulation
calibration (Section 5.2); Chapter 1's "AWGN retained as analytical
calibration" claims were updated to reflect the new companion use. Verified
(`tbl:table2awgn`, 54 cells, 0 mismatches).

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
