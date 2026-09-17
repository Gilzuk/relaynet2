# Final professor-handover review

**Review date:** 17 September 2026  
**Disposition:** Ready to hand to the professor for academic review. The rebuilt PDF meets the measured 120-page limit and the checked result tables match their source artifacts. The professor remains the final authority on academic framing and any submission-specific requirements.

The reusable implementation instructions are in [the fix-guidelines prompt](codex_professor_handover_fix_prompt.md). This report records what was changed, what was verified, and what remains outside the evidence.

## Review scope and checks

The active manuscript was traced from `thesis/main.tex`; inactive drafts and review-response appendices were excluded. The source graph contains 204 labels, 234 internal references, 51 distinct cited keys, and 18 included figures. The audit found no missing or duplicate labels, missing citation keys, missing figure files, or visible TODO/FIXME/review-block markers.

The result-table verifier checked 526 displayed numeric cells against their declared data sources: zero inconsistencies. The complete Python test suite passed: **299 passed**. The main PDF rebuilt successfully with XeLaTeX. A clean Overleaf bundle, including the local Hebrew style file, was extracted and compiled independently.

The fresh PDF contains **120 physical pages**, including front matter, appendices, and both inner/outer title-page sets. The printed body uses 12-point type and the existing margins; the table-of-contents depth is limited to chapters, sections, and subsections. The final rendered pages, tables, figures, bibliography, Hebrew abstract, and title pages were inspected. These checks follow the available [Tel Aviv University master's essay instructions](https://engineering.tau.ac.il/sites/engineering.tau.ac.il/files/media_server/Engineering/Tagel/masters_essay_guidelines.pdf). The title pages are dated September 2026.

## Corrections made

- Corrected the complex adaptive-Rayleigh noise scaling and added variance and seed-regression tests. Re-ran only the affected composite AF comparison under the existing protocol, with ten trials at each SNR and source hashes, in the separately named `codex_composite_af_validation.json`. Historical arrays and checkpoints remain intact.
- Unified the original unknown-channel figure and central table on the matched-protocol Viterbi artifacts. The figure now shows actual zero-error budgets with nominal rule-of-three bounds, and omits MLP estimates above the validated 16 dB range.
- Corrected the AF noise convention and separated the textbook CSI-assisted expression from the simulated relay. The QPSK BCJR control is described as relay-output detection; it is not used as an end-to-end destination comparison. These distinctions are consistent with the original [BCJR paper](https://doi.org/10.1109/TIT.1974.1055186) and [Tse and Viswanath's wireless-communication text](https://web.stanford.edu/~dntse/wireless_book.html).
- Reconciled the abstract, objectives, methods, results, discussion, summary, appendices, and Hebrew abstract. Claims are limited to tested channel families, budgets, endpoints, and SNR ranges. The text no longer treats a proposed error-propagation or overfitting explanation as experimentally established, and it makes no measured hardware-latency or energy-saving claim.
- Corrected the BPSK relay and coded-QPSK relay parameter counts and arithmetic: 170 parameters / 156 MACs for the former; 220 parameters / 208 MACs per symbol for the latter. The 16 dB MLP point is reported from its fixed-budget validation, while the higher-SNR tail is left unclaimed.
- Regenerated the original study figures from their declared sources as 300-DPI PNG and vector PDF. Tightened table headers and long appendix entries; the final TeX log contains no overfull boxes, unresolved references, or bookmark warnings.
- Added direct identifiers to bibliography entries checked during the review, including the correct DOI for [*Network Information Theory*](https://doi.org/10.1017/CBO9781139030687), the original [Kingma–Welling VAE preprint](https://arxiv.org/abs/1312.6114), and the DOIs for Nosratinia et al. and Cybenko.

## Citation and theory review

The reference-identity pass retrieved identifier metadata for 33 cited works and manually checked the remaining 18 against publisher, author, or library records. It found no current title/year mismatch after correcting the references. That is a metadata and targeted claim-support review; it does not mean every cited source was read cover to cover.

One substantive citation-context correction concerns Akdemir et al. The cited study uses learned channel estimates and evaluates relay-selection protocols; it does not establish a learned forwarding classifier. The manuscript now describes that work at the level supported by the [publisher record](https://link.springer.com/article/10.1007/s11277-024-11269-y).
The publisher's article title itself spells “Leaning-Aided”; the bibliography preserves that source spelling rather than silently correcting the record.

The baseline repository file [bibliography-verified.json](research/bibliography-verified.json) had labeled two title-only matches as verified: *Network Information Theory* matched an unrelated 2001 chapter DOI, and *Auto-Encoding Variational Bayes* matched unrelated 2024 authors. That file is now marked superseded and retains the historical data with an explicit warning. The manuscript bibliography instead points to the correct Cambridge book DOI and Kingma–Welling source above.

## Remaining evidence limits

- Two timing tables remain machine-dependent historical measurements; the verifier reports them as informational and they were not re-run in this handover pass.
- The coded-relay comparisons are implementation- and decoder-metric-specific. They do not establish a general error-propagation mechanism or show that relay decoding is inherently inferior.
- The 120-page count is for the rebuilt local PDF and includes its title pages. The author should update the September 2026 date if the actual submission date changes and confirm any professor- or department-specific formatting requests.
- Citation identity was checked across all active keys, while detailed claim support was targeted to the theory and reference contexts changed in this pass. This is a professor-handover review, not a substitute for the supervisor's final scholarly judgment.
