# Final-submission review against Khina's round-2 feedback

**Current submission review:** See [the final merged-snapshot readiness review](submission_readiness_review_2026-10-17.md) for the submission PDF, completed reviews and current validation. The report below is a preserved historical pass.


**Historical first pass:** superseded by [the skeptical re-audit](skeptical_reaudit_2026-10-17.md), which records further fixes, current verification and the author’s subsequent push/merge authorization.

**Reviewed:** PR #183 at `e58e94d9d6136963ef56a9b3586f5b437d8075fd`, then the local corrections on `fix/khina-round2-codex-review`, 17 October 2026. The PR head was reread through GitHub. This is an independent examiner-style assessment against the supplied email, not a statement of Dr. Khina's approval. No changes have been pushed.

## Verdict

**The published PR needed further corrections. The revised local manuscript addresses all six round-2 requests, and I found no remaining scientific blocker within its stated scope.** I would treat it as ready for the author's final approval, rather than request another experimental study. The claim is a bounded simulation comparison of implemented relay strategies, not a general advantage of learned relaying, an architecture lower bound, or a demonstrated hardware benefit.

Submission should use the rebuilt local manuscript, not the PDF currently attached to the remote PR. At the author's subsequent request to fix the remaining wording, the Hebrew abstract was reconciled with the English abstract: the 1.23-times ratio, real-tap factorization and hard/soft MLP scope are explicit. Final author reading remains recorded in `thesis/HEBREW_ABSTRACT_SYNC.md`. Publishing or pushing these corrections requires the author's explicit confirmation.

## Acceptance criteria

| Khina's request | Assessment of the revised manuscript | Evidence |
|---|---|---|
| Identify the actual canonical runs | Addressed, with unavoidable historical uncertainty explicit. Architectures, AWGN/BPSK training, QPSK/Rayleigh evaluation, I/Q scaling, window counts, 100 epochs, optimizer settings/defaults, checkpoint tolerance, and evaluation budgets are distinguished. Missing weights, training RNG state, loss histories and library versions are not reconstructed by assumption. | Sections 4.2–4.4, Table 4.2, ledger entries C4–C5. AF/DF per-trial outputs reproduce bit for bit; this does not prove exact reproducibility of the learned columns. |
| Compare penalties on common targets | Addressed. The reference fixes the target set, missed targets remain explicit lower bounds, and rankings, initialization spreads and minimum-size conclusions are rechecked. Mamba-2's misleading +4.69 dB is replaced by the best common-target result +5.83 dB; Mamba-S6 and Mamba-2 exchange order in that case. MLP remains best among the tested configurations. | Table 6.7, Tables 9.4–9.6, Appendix 9.6, stored per-seed BER curves. |
| Restrict matched-decoding conclusions | Addressed. The matched comparison covers the hard and soft MLP read-outs. The absence of a matched Mamba run is explicit. Neither missing Mamba results nor a learned relay supplied with channel reliability are implied. | Section 5.5.1, Table 5.4, abstract and conclusions. |
| Make the formal problem consistent | Addressed. A local output is followed by a block normalization operator, including the actual separate I/Q gains and Hybrid switch. Empirical hard-decision loss and noise-averaged expected BER are distinguished. | Section 4.1.5, Remark 4.1, canonical block diagram. |
| Make corrected comparisons principal | Addressed. Matched results lead the coded discussion and tables; factorized counts lead the main cost tables and Figure 6.7. Five original tables and three unweighted plots are retained in Appendix 9.7 with short main-text diagnostics. | Section 5.5; Tables 5.4–5.8, 6.8–6.9; Figure 6.7; Appendix 9.7. |
| Resolve scope and presentation | Addressed. Local Mamba causality is separated from block buffering; shared window length is not described as shared context; the implementations are identified as simplified; the ten-pilot difference is unresolved; an existing immutable code/data snapshot is identified; Chapter 5 headers read “Core Experiments.” | Remark 4.1, Section 4.4, Figure 6.5, Section 4.3.4 and rendered PDF. |

## Additional findings resolved locally

1. **Causality and normalization still contradicted each other.** The remote remark called AF/DF strictly causal and inferred that canonical results did not depend on look-ahead because neighbors contain no additional information. A trained function may nevertheless use those inputs, and block normalization changes transmitted-symbol dependence. The revised wording distinguishes local input dependence, detector look-ahead and block buffering; it makes no unperformed ablation claim.
2. **Corrected numbers still had misleading presentation.** Secondary coded tables placed matched results last and bolded winners from the older unweighted comparison. Columns and emphasis now follow the principal matched comparison. The cost figure now uses the same explicit multiplication-count convention as the main factorized comparison; the joint trellis remains a diagnostic. Numerical verifications were redirected to the new column positions without reducing coverage.
3. **Censored penalties were still interpreted as exact measurements.** The loss–penalty plot now marks three failed Transformer runs with lower-bound arrows, and the discussion no longer treats a correlation of substituted bounds as a correlation of true penalties. A VAE spread of at least 0.156 dB cannot establish that its spread is below 0.18 dB. The unsupported five-of-nine size-effect count was removed.
4. **Two conclusions exceeded the comparisons.** Tracking DF does not isolate regeneration as the causal explanation for a learned relay's improvement over AF. Also, 256 factorized multiplications versus 208 network MACs at seven taps does not support the blanket assertion that an arithmetic budget cannot exclude the classical method. The revised conclusions give the observations and avoid an unmeasured mechanism or hardware feasibility claim.
5. **The claimed release tag did not exist.** The manuscript now links the existing immutable experimental code/data snapshot at the reviewed PR head and identifies the historical canonical code commits. A future release tag is optional, not represented as already archived.

6. **Hebrew/English precision.** Both abstracts now state the real-tap factorization condition and identify the evaluated hard and soft MLP outputs. The ambiguous Hebrew “פי 1.23 יותר” is replaced by an explicit ratio to the relay's multiplication count. The canonical Hebrew sentence also states the identical-distribution assumption.

## Limitations that are disclosed, not reasons to invent new results

- Original learned canonical weights cannot be regenerated exactly. The supplied feedback explicitly permits known configurations with uncertainties stated.
- Matched Mamba was not evaluated. Restricting the conclusion to the two MLP read-outs follows the alternative Khina explicitly offered.
- Three-seed summaries, exploratory uncorrected paired tests and nominal bit-level intervals are not universal guarantees. The thesis states their scope and treats unresolved differences as unresolved.
- Real-tap I/Q factorization and selected arithmetic counts do not establish optimized runtime, energy savings or deployment feasibility. The matched destination metric also does not model relay decision errors.
- Stronger classical comparators and further generalization tests remain future work. Their absence limits the claims; it does not require extending this thesis once those limits are respected.

## Verification and submission state

- Numerical verifier: **873 checked values, zero inconsistencies**, with coverage floors retained.
- Provenance audit: clean.
- Unit tests: **347 passed, four optional PyTorch/Mamba-2 skips** in this environment. The reported historical 351-test run with PyTorch is distinguished from this local check.
- Rebuilt PDF: **119 countable pages, 135 total**, within the 120-page limit; no LaTeX errors, undefined references or overfull boxes. Revised tables, figures, formal statement, running headers and appendix layout were visually inspected.
- Both Overleaf bundles are rebuilt from the active source manifest. Original experimental result data are preserved; no training or BER experiment was rerun for these local corrections.
- Remote PR remains unchanged. The final author checks are the Hebrew sentence and approval of the prepared local revision before any push.
