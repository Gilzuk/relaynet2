# Changelog

All notable changes to the thesis are recorded here. The in-document record is
Appendix E (supervisor comments and their resolutions) and Appendix F (independent
review findings and their resolutions).

Format follows [Keep a Changelog](https://keepachangelog.com/). Verification claims
refer to `verify_thesis_tables.py` at the repository root, which checks every table
cell and quantitative prose claim against the data files that produced it.

---

## [Major revision] — 2026-07-31

Restructured around a single canonical setup, all supervisor comments addressed, and
all findings from an independent technical review resolved. Clean copy: 123 pages.

### Changed — scope and structure

- Fixed **one canonical setup** and varied only the relay function: two-hop SISO,
  i.i.d. Rayleigh fast fading, complex baseband, BPSK, uncoded BER. Everything
  outside it was removed rather than retained as a side study.
- Split the two surviving out-of-scope threads into their own chapters, each with a
  stated configuration: QPSK/16-QAM extension (Ch. 6) and the unknown-channel study
  (Ch. 7, the principal contribution).
- Reframed the central claim from "learned relays replace classical processing" to
  "characterise when learning helps." On a matched, known channel, symbol-wise DF is
  at least as good at zero parameter cost; the learned relay's value is
  family-agnostic mitigation under unknown or mismatched channels.
- Chapter 2 is now a literature review only, with no experimental results in it.

### Removed

- AWGN and Rician channel-robustness studies; 2x2 MIMO second hop and its
  ZF/LMMSE/SIC equalization; 16-PSK; CSI-injection ablation; end-to-end autoencoder;
  diversity–multiplexing tradeoff discussion. All redesignated as future work.
- The "Theoretical Foundations: MIMO Equalization" section, which had survived the
  deletion above and described equalizers the thesis never uses.
- A channel-validation subsection in Ch. 2 duplicating the Ch. 5 experiment; its
  theoretical-SNR table moved to Ch. 5.
- A fading-coefficient-distribution subsection analysing Rician K-factors, although
  Rician fading is future work and no Rician result exists.
- Two channel-validation figures printed identically in Ch. 2 and Ch. 5; each
  validation figure now appears exactly once.
- A relay-comparison positioning table duplicating the research-positioning table in
  Ch. 3.
- A CSI/LayerNorm results table and three conclusions citing experiments cut in the
  restructure (a 48-variant search, 16-PSK rankings, and a nonexistent section).

### Added

- **Appendix E** — supervisor comments with their resolutions; deletions flagged
  explicitly rather than made silently.
- **Appendix F** — 24 independent review findings, grouped as substantive,
  consistency and editorial, each with the change made.
- Remarks formalising two scoping distinctions: symbol-wise versus block DF, and the
  realizability of the non-causal relay window under half-duplex store-and-forward.
- A per-experiment Monte Carlo budget table, replacing a blanket claim that one
  budget applied to all results, together with the caveat that a 5-trial Wilcoxon
  test cannot reach p < 0.05.
- Automated verification of every table and quantitative claim against its data
  source: 200 values, no discrepancies.
- Committed result files for the blind, partial-posterior and composite
  unknown-channel experiments, so those prose claims are machine-checked too.

### Fixed — technical

- AF end-to-end SNR expression: added attribution and stated its CSI-assisted
  variable-gain assumption, which determines the convention-dependent `+1` term.
- Universal-approximation overclaim: exact representation of the posterior holds
  only at fixed noise variance, not across the trained SNR family.
- Mamba selectivity derivation: stated the Hurwitz parameterization
  `A = -exp(A_log)`, without which the contraction step is unjustified.
- Relay observation equation: rewritten in post-compensation form with the random
  effective SNR, instead of showing the AWGN calibration case.
- DF error-composition formula: two inconsistent forms reconciled to the exact
  odd-flip expression.
- Pilot-budget collapse: recharacterised as an estimation-variance effect rather
  than a loss of channel identifiability, since the least-squares problem remains
  overdetermined.
- Renamed the channel operator to stop it colliding with the fading coefficient.
- Qualified two claims that outran the evidence: the VAE result and a uniqueness
  claim about short blocks.

### Fixed — data and references

- One reference had the wrong author, title and page range; corrected to Samuel,
  Diskin & Wiesel, *Learning to Detect*, IEEE Trans. Signal Processing, 2019.
- Two works appearing twice under different keys merged.
- The Mamba architecture had been attributed to the earlier S4 paper; repointed.
- Six table cells corrected against their data sources.
- One figure regenerated from the committed results file, and the corresponding
  table reset from the same source, after both were found to derive from a
  superseded run.
- Discussion text contradicting the canonical-channel results table corrected;
  timing figures and units unified with the table they cite.

### Fixed — build

- **Package ordering fault that broke compilation on Overleaf.** `polyglossia`
  (which loads `bidi`) preceded fifteen packages it must follow, producing errors
  that `latexmk` treats as fatal; every cross-reference and citation rendered as
  `??`. The language block now loads last. The document builds clean: 0 undefined
  references, 0 undefined citations.
- Restored the English abstract, which was silently absent because `main.tex`
  included a nonexistent file.

### Fixed — presentation

- Replaced 142 em dashes with conventional punctuation; six remain, all legitimate
  (table placeholders and bibliography author-repetition markers).
- Fixed seven asides delimited by paired colons, a pandoc-conversion artifact.
- Fixed a truncated figure caption, a malformed section heading, a broken
  enumeration, a stale section label, and a claim that results span "six
  configurations."

### Known issues

- The VAE relay decodes stochastically at inference; the deterministic variant was
  not evaluated, so its weak result may be an inference-time artifact rather than
  evidence about generative relaying. Stated as open in the text.
- Blind CMA was not tested at the shortest block length; the corresponding claim is
  narrowed accordingly.
- Two references (Bergel 2023 / ICASSP 2024, Akdemir et al. 2024) are unverified
  against the primary sources.
- Six hardcoded cross-references ("Table 15", "Table 24", "Table 13",
  "Section 3.7.6") and two bare `\ref` pointers predate chapter-based numbering and
  now name tables that do not exist. Identified, not yet fixed.
- Chapters 5, 6 and 7 all live in `chapters/ch05_experiments.tex`, and
  `ch06_discussion.tex` / `ch07_summary.tex` hold Chapters 8 and 9. File names do
  not match chapter numbers.
