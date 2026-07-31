# Thesis revision — changelog

**To:** Dr. Anatoly Khina
**Subject:** M.Sc. thesis — major revision (changelog attached)

Dear Dr. Khina,

Attached is a **major revision** of my thesis. The changes are listed below;
Appendices E and F in the document itself carry the full detail.

---

## 1. Structure and scope

- **Fixed a single canonical setup** and varied only the relay function: two-hop
  SISO, i.i.d. Rayleigh fast fading, complex baseband, BPSK, uncoded BER.
  *(Your comment: "define one model that you've tackled and stick to it.")*
- **Removed** from the thesis and redesignated as future work: AWGN and Rician
  robustness studies; 2x2 MIMO second hop and its ZF/LMMSE/SIC equalization;
  16-PSK; CSI-injection ablation; end-to-end autoencoder; diversity–multiplexing
  tradeoff discussion.
- **Retained as two separate chapters,** each with its own stated configuration:
  QPSK/16-QAM extension (Ch. 6) and the unknown-channel study (Ch. 7, the
  principal contribution).
- **Removed the MIMO equalization section** from Ch. 2, which had survived the
  above deletion and described machinery the thesis never uses.
- **Removed from Ch. 2** a validation subsection duplicating the Ch. 5 experiment,
  a Rician fading-distribution subsection (Rician is future work), two figures
  printed identically in Ch. 2 and Ch. 5, and a positioning table duplicating Ch. 3.
  Chapter 2 is now a literature review only, with no results in it.

## 2. Your comments — all 17 addressed (Appendix E)

- **Symbol-wise vs. block DF.** Now a formal remark. The thesis studies symbol-wise
  (slicing) DF throughout and explicitly does not claim its high-SNR optimality
  carries over to coded block-DF, which is named as the leading future-work item.
- **"Replace classical strategies with a neural net."** Reframed. On a matched,
  known channel DF is at least as good at zero parameter cost; the learned relay's
  value is family-agnostic mitigation under unknown or mismatched channels. The
  claim is now "characterise when learning helps."
- **Non-causal relay window.** Formalised as a remark: realizable in the half-duplex
  store-and-forward protocol, at a fixed w-symbol processing latency.
- **Missing abstract; compilation errors; unusual citations; MMSE vs. LMMSE; time
  axis; fading introduced late.** All resolved, each documented with its resolution.

## 3. Technical corrections — 24 findings (Appendix F)

- Attributed the AF end-to-end SNR expression and stated its CSI-assisted
  variable-gain assumption.
- Corrected an overclaim that a single-hidden-layer network represents the
  posterior *exactly*; this holds only at fixed noise variance.
- Stated the Hurwitz parameterization the Mamba selectivity derivation depends on.
- Rewrote the relay observation equation, which showed the AWGN calibration case
  rather than the canonical Rayleigh channel.
- Recharacterised the 5-pilot collapse as an estimation-variance effect, not a loss
  of channel identifiability (the least-squares problem remains overdetermined).
- Qualified two claims that outran the evidence: the VAE result and a uniqueness
  claim about short blocks.

## 4. Data and results integrity

- **Corrected one reference** recorded with the wrong author, title and pages
  (now Samuel, Diskin & Wiesel, *Learning to Detect*, IEEE TSP, 2019); **merged two
  duplicated bibliography entries.**
- **Corrected six table cells** that disagreed with the underlying data.
- **Regenerated one figure** that had been produced from a superseded run, and reset
  the corresponding table from the committed results file.
- **Automated verification:** every table and quantitative claim is now checked
  against the data files that produced it — currently 200 values, no discrepancies.

## 5. Presentation

- Replaced 142 em dashes with conventional punctuation, and fixed seven asides that
  had been delimited by paired colons.
- Fixed a truncated figure caption, a malformed section heading, a broken
  enumeration, and a stale claim that results span "six configurations."

## 6. Build

- **Fixed a package-ordering fault that broke compilation on Overleaf,** where every
  cross-reference and citation rendered as "??". The document now builds clean:
  0 undefined references, 0 undefined citations.

---

## Open items

- The VAE relay's weak result may be an inference-time artifact — it decodes
  stochastically, and the deterministic variant was not evaluated. Flagged in the
  text as open; I can run it if you prefer it closed.
- Blind CMA was not tested at the shortest block length; the corresponding claim is
  correspondingly narrowed.
- Could you confirm the Bergel and Akdemir references are as you intended? I could
  not verify their details independently.

A copy with inline annotations marking each change in the text is available if that
is easier to review.

Best regards,
Gil Zukerman
