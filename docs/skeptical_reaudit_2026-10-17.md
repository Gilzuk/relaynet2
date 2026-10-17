# Skeptical re-audit of PR 183

This pass follows `final_submission_review_2026-10-17.md` and supersedes its approval, page-count and verification status. The author subsequently authorized “Merge and push”, requested another audit, then asked to publish this revision before starting reproduction experiments on a new branch. This is an independent technical review against the supplied round-2 email, not approval by Dr. Khina.

## Verdict

The additional inconsistencies below are corrected. Within the stated experimental scope, the revised manuscript addresses all six round-2 requests. Exact regeneration of the historical learned canonical columns remains impossible from the archived record; the manuscript identifies the missing information instead of inventing it. The separately requested new experiments are follow-on evidence, not experiments claimed to have generated the published thesis.

## Findings and repairs

| Finding | Repair and evidence |
|---|---|
| Sequence architectures omitted their input skip; describing the complete output as tanh-bounded was incorrect. | Sections 4.2–4.3 and 5.2 now describe raw center input plus the tanh head, center read-out, head widths/activations, internal normalization, Transformer positional encoding, feedforward width and dropout. Checked against the checkpoint implementations. |
| VAE inference and architecture details were incomplete or inconsistent. | State stochastic latent sampling during training, deterministic mean decoding during evaluation, KL averaging, and the complete native and 3K encoder/decoder widths. |
| Canonical provenance overstated what was known about saved weights and randomness. | Identify absent archived canonical checkpoints, unseeded PyTorch sources, seeded NumPy shuffling for the VAE, unknown historical library versions, CPU float32 execution and optional versus internal LayerNorm. Updated `canonical_run_provenance.py` and its JSON note. No stored BER values changed. |
| Table 6.7 did not use the canonical family-specific training protocol. | Section 6.4.2 now specifies its separate 40-epoch final-checkpoint loop, Adam settings, batch size, three training SNRs, 19,998/39,996 windows, initialization/source seeds, evaluation grid and budget. Checked against `seq_models_on_memory.py` and its imported evaluator. |
| Training/inference padding, CPU/CUDA labels and ledger commands conflicted with code. | Distinguish zero-padded training from edge-repeated inference where applicable; correct Appendix 9.1 architecture/device entries, ledger references, matched-metric/trial-count environment settings, coded-size scope and adaptation-envelope derivation. |
| Two derived median penalties were marked exact despite a lower-ranked censored observation that could exceed the middle observation. | Correct `penalty_common_targets.py` and add odd/even/censored-rank regression cases. Only the exactness flags for `table_9_6/Transformer-3K` and `transformer_instability/Transformer-3K` change; no numerical penalty changes. |
| Causal explanations exceeded the evidence. | Remove sole attribution of the scalar-MAP gap to context; hard/soft forwarding also differs. Figure 6.5 now states that estimation variability and nonlinear-model mismatch were not separated. Ten pilots are a chosen budget, not an established minimum. |
| Scope and reproduction instructions still overclaimed. | Qualify canonical CSI, channel detection versus code decoding, full-frame buffering, nominal parameter budgets, QAM axis targets and initialization evidence. Correct `REPRODUCE.md`, Makefile guidance and dependency-version provenance. |
| The previous archive reference omitted the revised figure generators. | Cite immutable experimental/code/data snapshot `218a1c550cb8b917a95d9545c5211e109985bf38`, including the corrected generators, derived metadata and verifier. The final manuscript clarifications follow that snapshot. No nonexistent tag is claimed. |

## Checks performed

- Thesis verifier: **873 checked values, zero inconsistencies**; coverage floors retained. The scalar-MAP paragraph boundary was updated after its interpretation changed, retaining all seven checked cells.
- Test suite: **348 passed, four skipped** because the optional PyTorch/Mamba-2 import is unavailable in this environment. One existing warning concerns rejection of complex taps. This is distinct from historical test runs in other environments.
- Verifier mutation checks: **10 passed**. Provenance audit: clean.
- Independent interpolation and interval aggregation over **762 saved curves**, including **104 with missed targets**: no numerical discrepancy. Reproduced all 12 Table 6.7 entries and all ten minimum-size budget rows from the stored curves.
- Rebuilt PDF: **120 countable pages / 136 total** (Chapter 1 starts at physical page 14; final body folio 120 is physical page 133; Hebrew back matter starts at 134). No LaTeX errors, undefined or multiply-defined references, overfull boxes or missing characters. Existing duplicate PDF table-destination and Courier font warnings remain. Revised methods, tables, pilot caption, ledger and final body/Hebrew pages were visually checked.
- PDF, preview and both source bundles are rebuilt together. No training or BER experiment was rerun during this audit; the curve reanalysis and document checks are not presented as new simulation evidence.

## Remaining limits and next work

Matched Mamba was not evaluated, historical canonical learned checkpoints/RNG state are unavailable, three-seed statistics do not characterize all training variability, and arithmetic counts do not establish hardware performance. These limitations remain explicit. The author has requested a separate reproduction branch with fresh outputs, logs, reference-file links and discrepancy reporting. Published files must be preserved, and cross-environment stochastic differences must not be silently relabeled as exact reproduction or tuned away.

The reply remains a draft. Neither this review nor repository publication represents advisor approval or sending the thesis to the advisor.
