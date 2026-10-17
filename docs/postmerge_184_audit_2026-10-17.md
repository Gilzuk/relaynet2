# Post-merge skeptical audit — 17 October 2026

PR [#184](https://github.com/Gilzuk/relaynet2/pull/184) merged at
`a714061cbc6abd111beba037a4a955cfeac27b2e`, preserving the reviewed tree
`9fb1018262c6f18cfdf68cf2a286a0a2693f9dc8` exactly. This audit started from
that merged revision. The author requested the merge and another audit.
This is a technical review, not approval by Dr. Khina or a separate professor.

## Verdict and changes

The post-merge check found a real numerical-analysis defect and a scope
contradiction. Both are corrected in this follow-up branch. The principal
conclusions remain supported within the stated experiments: no tested canonical
learned relay has a resolved advantage over DF, and matched block DF beats both
coded MLP read-outs. This is not an equivalence or population-ranking claim.
Raw historical and new BER measurements, checkpoints, plots and archived worker
statuses are unchanged. Original measurements are not invalidated merely because
another trained realization differs.

### Integer-error ties in paired tests

Subtracting floating-point BER arrays can turn equal absolute count differences
into slightly unequal numbers, assigning incorrect signed ranks. The helper now
reconstructs integer error counts, validates their common denominator, and tests
the count differences. Canonical trials contain 10,000 bits; the coded denominator
is read from `n_frames * frame_info_bits` (500 * 200). A missing denominator is an
error, and BERs incompatible with it are rejected. Mean BER differences are retained
unchanged. SciPy 1.17.0 uses two-sided `wilcoxon`, `zero_method="wilcox"`,
`correction=False`, `method="auto"`; no multiplicity correction is introduced.
Zeros are omitted and all-zero differences return p=1. Thirty-trial coded tests
can use the asymptotic method; they are not described as exact tests.

The corrected code and derived JSON files are pinned at
[`80b4545adcc66e4be1f435364f7ec927e87f6414`](https://github.com/Gilzuk/relaynet2/tree/80b4545adcc66e4be1f435364f7ec927e87f6414).
The thesis archive link now identifies that revision, which also contains the
historical raw records and the reproduction evidence. Original run commits and
the hosted evidence-archive commit remain separately identified.

| Comparison | Previous p | Corrected p | Consequence at uncorrected 0.05 |
|---|---:|---:|---|
| Historical 3K Transformer vs DF, 20 dB | 0.05078125 | 0.04296875 | DF advantage becomes resolved; mean BER gap +0.00026. |
| Additional native S6 vs DF, 16 dB | 0.03515625 | 0.05468750 | DF advantage is no longer resolved; mean BER gap +0.00054. |

The native S6 error-count differences are `[9, 2, 8, 5, -1, 29, -2, 1, 4, -1]`.
Its corrected two-sided probability is 56/1024 under exhaustive sign enumeration.
The historical summary changes 61 p-values; the additional-run assessment changes
18 (14 canonical, four coded). Exactly the two classifications above cross 0.05.
All stored mean differences and BER measurements remain unchanged. All 24 coded
comparisons still have p < 1e-5 (maximum 1.7333066441991056e-6).

Evidence and executable checks:

- [`canonical_paired_tests.py`](../canonical_paired_tests.py) and
  [`results/canonical_paired_tests.json`](../results/canonical_paired_tests.json).
- [`scripts/assess_round2_reruns.py`](../scripts/assess_round2_reruns.py) and
  [`results/round2_rerun_assessment.json`](../results/round2_rerun_assessment.json).
- [`tests/test_canonical_paired_tests.py`](../tests/test_canonical_paired_tests.py)
  independently enumerates the sign distribution with integer doubled midranks
  for all **418** saved ten-pair contrasts, including ties and zero differences.
  All agree exactly. It also tests the reported S6 case, identical observations,
  and rejection of a coded BER under the wrong denominator.

### Constellation scope

The claim that 16-QAM appeared only in the adaptation ladder contradicted the
fixed-MCS equal-throughput control in Table 5.5. The active scope statements in
Chapters 3–5 now name both uses. The learned/classical relay comparison is QPSK;
no standalone learned-relay architecture comparison for 16-QAM is claimed.
Superseded drafts outside the compiled manuscript are left as historical records.

## Additional skeptical checks

- All ten common-target analysis blocks regenerate identically to the merged
  JSON, excluding descriptive/git-revision metadata. Across 750 distinct input
  curve/reference combinations there are 1,568 attained and 122 unreached target
  instances; none exercise the below-range or unbracketed cases. Failed targets
  remain explicit bounds. This check supports the current reported dataset,
  not every possible future censoring pattern.
- Model audit on the merged snapshot strictly loads all 48 retained checkpoints
  and verifies 70 inventoried output hashes. No model or checkpoint changes are
  included in this follow-up. The historical canonical states/weights and the
  new coded classifier weights remain unavailable, as disclosed.
- The six-point round-2 acceptance matrix from the
  [preceding review](final_skeptical_model_triage_2026-10-17.md) remains applicable
  with the statistical correction above. The main matched-coded subsection is
  §5.5.1; that older review's §5.5.2 reference is a documentation typo, not the
  location of the principal matched comparison.
- The original learned differences cannot all be attributed to different
  initialization seeds. Sequence IDs and coded seed policy repeat. Same-host
  seed-42 identity is demonstrated only for the retained native MLP control.
- No full-budget experiment was started, cancelled, restarted or modified. No
  advisor email was sent. No broader statistical-equivalence claim is made.

## Validation

- Full pytest suite after the numerical fix: **366 passed**, one existing
  complex-cast warning in the complex-tap rejection test, 98.16 seconds.
  The subsequently added exhaustive enumeration check also passed; the final
  focused file has **four passing tests** (three overlap the full suite).
- Numerical verifier: **892 checked values, zero inconsistencies**.
- Archived assessment `--check` passes; provenance audit passes after the
  code-and-derived-results commit. The model audit passed on the merged snapshot.
- Rebuilt XeLaTeX PDF: **120 numbered body pages / 136 total**. Revised methods,
  constellation scope, archive reference and headers inspected. No undefined
  references/citations, missing characters or overfull boxes. Existing duplicate
  table-destination and Courier feature warnings remain.
- Thesis PDF, preview and both Overleaf bundles synchronized in the document
  commit. This follow-up is separate from merged PR #184 and does not itself
  claim a hosted CI result or final advisor acceptance.

To repeat the focused checks in the declared environment:

```sh
python -m pytest -q tests/test_canonical_paired_tests.py
python scripts/assess_round2_reruns.py --check
python verify_thesis_tables.py
python provenance_audit.py
```

`python canonical_paired_tests.py` regenerates the historical derived statistical
summary; `python scripts/assess_round2_reruns.py` regenerates the additional-run
assessment. Neither command trains models or overwrites the raw measurements.
