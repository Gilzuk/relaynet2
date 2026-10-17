# Additional training replications and seed sensitivity — 17 October 2026

All four hosted jobs completed with worker exit 0. Their scientific comparison status is `RETRAINING_DIFFERENCE_REQUIRES_REVIEW`; red workflow checks do not indicate crashed training. The archive job succeeded, with no missing jobs, in commit `dc7c02c00cbebdfb566362a810ab4b45cb6abbe0`. Run source is `c0394f0da5dac449594b89a9794fe251fad75e76`. These runs predate the later strict-RNG additions in `c192560`.

## Interpretation and seed design

These are additional training replications and seed-sensitivity evidence. Numerical differences between trained realizations do not by themselves establish that the original measurements were incorrect. The original tables continue to report their original runs; the new results supplement them. The archived comparator status flags numerical disagreement for review, not invalidity of the earlier experiment.

| Study | Original and additional seed evidence | Interpretation |
|---|---|---|
| Native / 3K canonical | Historical Torch state unknown; new bootstrap seed 20261017 | Additional realization for each protocol. Initialization variability is a plausible contributor, but seed alone is not isolated from environment differences. Do not pool native and 3K as one seed sweep. |
| Sequence memory | Both executions use IDs 0, 1, 2 for each architecture and channel | A repeated three-seed study with results grouped by execution environment, not six independent seeds. |
| Matched coded | Initialization 0, training-data seed 0 and shuffle seed 42 retained | Same-protocol replication, not a deliberately changed-initialization-seed experiment. |
| Native MLP seed-42 control | Two full-budget repeats with all RNGs set to 42 and deterministic operations | Same-seed repeatability control, not two independent seed realizations. |

The seed policies are documented in `docs/rng_reproducibility_audit_2026-10-17.md`; `seed_design` in the assessment JSON records the canonical bootstrap values and checks all sequence seed groups directly against the archived outputs. Historical versus additional statistics are kept separate; no pooled training-seed confidence interval or universal seed-only explanation is claimed.

## Manuscript changes

- Historical Tables 5.2, 5.3, 5.9 and 6.7, their original inputs and plots remain historical measurements. They are not selectively overwritten with favorable retraining outcomes.
- The canonical discussion and figure caption retain the observed 0.004 feedforward–DF BER bound for its original training realization. The new native VAE gap is 0.00640 at 4 dB. The historical VAE equality with DF at 20 dB is also scoped to that run.
- A new discussion subsection, `sec:round2-rerun-validation`, reports the new outcomes separately. Canonical pointwise rankings change. The sequence-memory best-of-three ordering is unchanged across three channels, without a claim of population superiority or equivalence.
- QPSK ISI Transformer best common-target penalty is 5.959459352 historical versus 6.635541522 dB new. Mamba2 initialization spread is historically greater than 3.281577614 dB versus 0.972542107 dB new. Its seed 1 reaches BER 1e-3 in the new run; the historical failed target remains an unsuccessful case in the historical table.
- Earlier same-host/cross-machine repeatability observations are identified as observations of those particular configurations. The new control establishes exact seed-42 repetition for one full-budget native MLP on one host, not every model or environment.
- The minimum-size training grid, pilot study, adaptive tails and matched Mamba remain outside the rerun scope. Historical seeded sequence/coded discrepancies are unresolved; missing canonical Torch seeding is not a sufficient explanation for those paths.

## Independent calculations and evidence

Run `python scripts/assess_round2_reruns.py --check`. Without `--check` it writes only `results/round2_rerun_assessment.json`; it does not retrain or replace historical experiment inputs. The script verifies archived reference/output/checkpoint/log hashes, recomputes the job comparisons, checks exact classical arrays, recomputes common-target penalties from old and new BER curves, and calculates fresh paired Wilcoxon tests from the new trials. The JSON links the source artifacts at the immutable archive revision.

The test is two-sided Wilcoxon signed-rank, uncorrected for multiplicity, following the historical reporting convention. No learned canonical relay has a resolved advantage over DF in either new canonical table. On the coded grid, matched block DF beats both MLP read-outs in all 30 paired trials at all six SNRs with either MLP destination metric; all 24 recomputed p-values are below 1e-5. These tests condition on the trained networks. They do not resample training or establish statistical equivalence.

The thesis numerical verifier adds 19 checks on the new subsection, taking coverage from 873 to 892 values. A mutation test changes the VAE gap to 0.00400 and must fail the new check. The reproducibility methods distinguish local generator seeds, global RNG states, historical recovery and checkpoint reuse for evaluation; saved weights alone do not imply resumable optimizer state.

The advisor drafts are updated but remain unsent. No new experiment was started or cancelled, and no original result was changed.

## Validation of the revision

- Assessment recomputed from the archived files and matched the committed JSON.
- Numerical verifier: 892 values, zero inconsistencies; 64 focused tests passed, including the new drift mutation.
- XeLaTeX/latexmk build: 120 Arabic-numbered pages, 136 total. Repeated discussion was shortened without changing historical tables or data. Revised pages, running headers, ledger and both abstracts were visually inspected.
- Final log has no undefined references/citations, missing characters or overfull boxes. Existing Courier font-feature and duplicate table-destination warnings remain; they were also present in the prior build.
- Both Overleaf bundles and both PDF copies are regenerated. The publication commit uses `[skip ci]` to avoid another full-budget reproduction launch; no new hosted CI pass is claimed. PR #184 remains draft for author review; numerical disagreement alone is not a finding that historical measurements are incorrect.
