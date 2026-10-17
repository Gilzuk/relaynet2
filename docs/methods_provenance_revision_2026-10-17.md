# Methods and appendix revision

The author requested that the thesis body emphasize the configuration actually used, with run bookkeeping in the appendix. This revision is based on source main `8734afb5e1421188237fe06fcc7c258ec6ab2a43`.

## Changes

- Section 4.3.3 now opens with the canonical training and evaluation design. Table 4.2, including sample counts, epochs, learning rates, batch sizes and checkpoint rules, is unchanged. Architecture descriptions, scaling, optimizer and evaluation settings remain in Methods.
- Run dates were removed. Commit identifiers, code paths, execution records and software details are consolidated in Appendix 9.2 and the existing Appendix 9.3 ledger. Section 4.3.4 explicitly directs readers to the Git repository for reproduction steps.
- Detailed missing-RNG commentary was removed from the main chapters. Appendix 9.2 retains a concise account of unavailable checkpoints, software versions and training states, as required by the request to disclose settings that cannot be established. The additional-replication discussion retains its scientific limits: the comparisons do not isolate seed-only effects, and repeated seed IDs are not independent seeds.
- The numerical verifier follows the relocated runtime statement and still checks both numbers. Two mutation cases confirm that an incorrect runtime is detected in its new appendix location.
- The PDF, preview and both Overleaf bundles are synchronized. Both README sources reflect the new page count.

## Validation

- Numerical verifier: **892 checked values, zero inconsistencies**.
- Focused document/verifier suite: **24 passed, 10 deselected**, including both new runtime-mutation checks.
- All active table and figure environments are identical to the base revision. Every label is retained; no raw results, experiment scripts, model code or figure inputs changed.
- The XeLaTeX build completed with no undefined references/citations, missing characters or overfull boxes. Existing font-feature and duplicate longtable-anchor warnings remain. The local build used the compatible bidi package and flushed generated files between TeX passes after an incomplete auxiliary-file failure; no manuscript workaround was introduced.
- Visual review covers the revised Methods, replication discussion, reproducibility appendix and start of the ledger. The appendix no longer spills a two-line tail onto an otherwise empty page.
- The PDF has **119 Arabic-numbered body pages and 135 physical pages**: 13 front-matter and three Hebrew back-matter pages. These remain separate counts; the earlier administrative qualification about the faculty counting convention is unchanged.
- PDF SHA-256: `f55ac2e3553af73f53bd241959a14e7395a4c4c1e5bbddc4ff2ddc67a9a0660e`.

Evidence: `results/reproduction/validation/methods_provenance_20261017/`. The advisor-draft quotation inventory is updated separately to pin this reviewed PDF and its new page references. No experiment was started, stopped or rerun. Publication of a review branch does not merge it, update the generated thesis mirror or send the email.
