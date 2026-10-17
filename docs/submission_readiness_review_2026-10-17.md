# Submission readiness review — 17 October 2026

Reviewed merged snapshot: **`5eb34ab3ed01bc9df17498358656965af2fa916b`**,
after PRs #183–#187. The submission PDF is
[archived at that revision](https://github.com/Gilzuk/relaynet2/blob/5eb34ab3ed01bc9df17498358656965af2fa916b/thesis/main.pdf).
This is the current readiness review; earlier review reports describe their own
starting revisions and findings. This is a technical assessment, not Dr. Khina's
approval or evidence that the thesis or email has already been submitted.

## Verdict

**Ready for the author's submission within the stated experimental scope.**
All six supplied round-2 requests are addressed. This final targeted review found
no further blocking manuscript correction. It does not establish exact historical
replay for missing learned checkpoints or complete execution of every registered
experiment. The advisor remains the authority on acceptance.

The changes in this submission-document commit update the response drafts,
review record and handover. The reviewed thesis source, PDF, figure inputs,
experimental results and model code are unchanged.

## Feedback closure

| Request | Final disposition and location |
|---|---|
| Actual canonical configurations | §4.3.3 and Table 4.2 identify the runs behind Tables 5.2–5.3, architecture/parameter/window settings, BPSK/AWGN training, I/Q scaling at QPSK evaluation, training sample/window counts, 100 epochs, Adam settings, checkpoint selection and evaluation budgets. Historical software versions, learned weights, full training RNG states and per-epoch losses are explicitly unavailable. Ledger C4/C5 and `results/canonical_run_provenance.json` provide the trace. |
| Comparable SNR penalties | Table 6.7 and Appendix 9.6 use reference-defined targets and retain failed targets as bounds. Rankings, spreads and the size table are recomputed from archived seed curves. The historical Mamba2 +4.69 dB entry cannot win by omitting 10⁻³; the corrected best value is +5.83 dB, with failures still flagged. Minimum size means the smallest tested configuration under the declared budgets. |
| Matched-decoding scope | Principal §5.5.1 and Table 5.4 cover the hard and soft MLP read-outs. Matched Mamba was not evaluated, and no result for it is implied. The section number differs from the original feedback because the corrected comparison was moved first. |
| Formal problem | §4.1.5 separates local output from block normalization, including per-axis gains and the Hybrid block switch. Empirical hard-decision loss is distinguished from potentially differentiable noise-averaged BER. |
| Principal comparisons | Matched decoding leads §5.5. Factorized trellis counts lead Tables 6.8–6.9 and Figure 6.7; the real-tap factorization condition and arithmetic convention are stated. Unweighted/joint comparisons remain diagnostics. |
| Scope and presentation | Remark 4.1 and §4.4 distinguish local Mamba causality from whole-block transmission and a shared window from equal context. Simplified implementations are identified. Figure 6.5 leaves the ten-pilot difference unresolved. §4.3.4 cites immutable experimental snapshot `80b4545adcc66e4be1f435364f7ec927e87f6414`; the final manuscript is pinned separately above. Chapter 5's running headers are correct. |

## Final audit corrections included

- **Paired signed ranks:** PR #187 tests integer error-count differences, avoiding
  floating-point splitting of ties. Denominators are 10,000 canonical bits and
  100,000 coded information bits. All 418 saved ten-pair contrasts agree with
  independent sign enumeration. Historical 3K Transformer/DF at 20 dB changes
  p = 0.05078125 to 0.04296875; additional native S6/DF at 16 dB changes
  p = 0.03515625 to 0.05468750. No learned canonical DF advantage is resolved;
  all 24 additional matched-coded contrasts remain below 10⁻⁵. These are
  exploratory uncorrected tests conditional on trained checkpoints.
- **VAE inference:** PR #186 reconciles Chapter 2 with the code: sampling in
  training, decoding the latent mean at inference. A historical pointwise equality
  is not called equivalence, and the comparison does not identify a causal effect
  of variational regularization.
- **Reproduction runner:** PR #185 records a seed-42 global bootstrap while
  preserving driver-local seeds, checks output freshness, retains per-step files
  and saved weights, and propagates failures. Three historical records are archive
  references. The full 73-entry plan has not been executed in full; the retained
  cheap ISI-penalty integration used archived BER inputs and is an analysis check.

The [post-merge audit](postmerge_184_audit_2026-10-17.md),
[VAE review](vae_inference_review_2026-10-17.md) and
[runner review](full_runner_review_2026-10-17.md) retain the detailed evidence.

## Evidence checked for submission

Fresh checks against the reviewed merged snapshot:

- Numerical verifier: **892 checked values, zero inconsistencies**.
- Archived-run assessment: saved hashes, recalculated comparisons and saved
  assessment agree.
- Provenance audit: all declared outputs committed; no data predates its script.
- PDF SHA-256:
  `9be23d2c30ad49d793ec5ab029c12d669d4f91a02164ab7f6fea70314406207c`.
  The preview and the PDF embedded in each Overleaf bundle are byte-identical.
  Active chapter files in the annotated source bundle match the source tree.
- Page accounting rechecked: Chapter 1 begins at physical page 14; body folio
  120 is physical page 133; Hebrew back matter starts at 134. Thus **120
  numbered body pages and 136 physical pages**. All Chapter 5 running headers
  on physical pages 64–76 (folios 51–63) read “Core Experiments”; its opening
  page has the correct title.
- Visually inspected physical pages **24, 55, 69, 92, 102 and 133**: VAE
  interpretation, training table, principal coded discussion, ten-pilot caption,
  additional-run interpretation and final numbered page. No clipping, overlap or
  broken layout was observed on these pages. This was a targeted visual pass.

Recent unchanged-code evidence retained in the same snapshot:

- **386 tests passed**, including 19 runner regressions; one existing
  ComplexWarning is recorded. This suite was not redundantly rerun for these
  prose-only submission documents.
- Python **3.10, 3.11 and 3.12 CI passed** at head
  `6cbd90dd5f70bb77a99cba30f4e4a87184ff0bd8`
  ([run 39045052001](https://github.com/Gilzuk/relaynet2/actions/runs/39045052001)).
  The merged `5eb34ab3ed01` tree is identical to that reviewed head's tree.
- The retained-model audit verifies **48 checkpoints and 70 output hashes**;
  its declared limits are retained. Saved test, model, assessment, provenance and
  integration logs are in
  [`results/reproduction/validation/full_runner_20261017`](../results/reproduction/validation/full_runner_20261017).

## Disclosed limits and submission state

The four hosted training replications supplement the historical evidence. Their
classical arrays match; learned discrepancies are preserved and discussed in
§7.1.4. Sequence seed IDs 0/1/2 and coded initialization 0 repeat, so a change in
initial seed alone is not an established explanation. Small differences, unchanged
best-of-three order and failure to reject do not establish equivalence. The
same-host seed-42 control demonstrates repeatability of one native MLP protocol.

Historical canonical weights/RNG states/software versions/loss histories and
trained weights for the new matched-coded classifier remain unavailable. The
minimum-size training grid, pilots, adaptive tails and matched Mamba were not
rerun in the focused October validation. These limitations are disclosed and
bound the claims; none is silently replaced with invented evidence.

The [short email](../thesis/submission/email_to_khina_round2_resubmission_2026-10-17.md)
and [detailed response](../thesis/submission/reply_to_khina_round2_2026-10.md)
now reference the exact reviewed PDF, include the final statistical and VAE
corrections, distinguish historical measurements from corrected summaries, and
state the missing records and execution limits. Both remain **unsent drafts**.
No experiment was started, stopped, restarted or modified during this final check.
