# Round-2 reproduction evidence

The thesis corrections were merged in [PR 183](https://github.com/Gilzuk/relaynet2/pull/183). Follow-on experiments are kept separately in [PR 184](https://github.com/Gilzuk/relaynet2/pull/184). Original published result files are preserved.

**Recovery snapshot: 17 October 2026.** The local run [`20261017-round2-v2`](20261017-round2-v2/REPORT.md), frozen source `43ce4a6fbfe2a147a8c77c0fb06e44fcd5d7d5e2`, lost its execution session after its last heartbeat at **09:07:25 UTC**. It is marked `INTERRUPTED_SESSION_LOST`; it is not still running. The exact termination mechanism and time are unknown. All 16 completed sequence seeds and checkpoints were recovered. The original last-live status is preserved in the [interruption record](20261017-round2-v2/sequence_memory/interruption/recovery.json).

The [GitHub Actions workflow](../../.github/workflows/round2-reproduction.yml) runs the four unfinished experiments independently at their full original budgets. It retains failure artifacts and commits evidence to this reproduction branch. It reruns the complete sequence study on the new host, preserving the interrupted attempt separately; it does not combine seeds from different environments. See the [Actions runs](https://github.com/Gilzuk/relaynet2/actions) for live status. New evidence paths use `gha-<run-id>-<attempt>-<job>` and include their own frozen source/environment.

| Completed check | Result | Authoritative published JSON |
|---|---|---|
| Common-target penalty and minimum-size reanalysis | MATCH, zero differing scientific fields | [penalty_common_targets.json](../penalty_common_targets.json) |
| Canonical paired analysis | MATCH, zero differing fields | [canonical_paired_tests.json](../canonical_paired_tests.json) |
| Fresh canonical AF/DF | MATCH; per-trial BER arrays exactly equal in both tables | [native canonical](../modulation/qpsk_rayleigh.json), [3K canonical](../normalized_3k/3k_rayleigh.json) |
| Joint/factorized MLSE | MATCH; 264,000 decisions, zero disagreements | [factorized_mlse_equivalence.json](../factorized_mlse_equivalence.json) |

The common-target and paired jobs reanalyse saved data. AF/DF and factorization execute fresh simulations. Each job folder contains raw new output, reference snapshots/hashes, command, environment overrides, log and comparison; `matching_reference` directs matches to the corresponding published JSON.

## Flagged training differences

The [recovered comparison](20261017-round2-v2/sequence_memory/interruption/comparison.json) covers all 12 ISI (BPSK) seeds, all three ISI (QPSK) MLP seeds, and the first QPSK Transformer seed. It flags **53 changed seed/SNR BER cells**. The classical baseline curves match exactly for both reached channels. The completed BPSK architecture order is unchanged, but the full three-channel study is unfinished.

| BPSK architecture | Published best penalty (dB) | New best penalty (dB) | Published spread (dB) | New spread (dB) |
|---|---:|---:|---:|---:|
| MLP | 1.973793596 | 1.965266649 | 0.391174891 | 0.399144385 |
| Transformer | 3.519872627 | 3.723647162 | 0.330452266 | 0.235665197 |
| Mamba S6 | 4.298017768 | 4.298017768 | 1.148739914 | 1.148739914 |
| Mamba2 | 4.552580734 | 4.552580734 | 0.135312241 | 0.135312241 |

All targets in this BPSK subset are reached. These are best-of-three point estimates, not evidence of statistical superiority across training populations. Mamba S6's three saved BER curves match; Mamba2 seed 1 differs even though its best and spread are unchanged. Maximum absolute BER differences are 0.0003075 (BPSK MLP), 0.001535 (BPSK Transformer), 0.0005675 (BPSK Mamba2), 0.001565 (QPSK MLP) and 0.0007425 (the single completed QPSK Transformer seed).

An independent [original-driver control](20261017-original-control/comparison.json), retaining the original MLP loop and all three BPSK seeds, produced **zero differences against the instrumented replay**. This rules out the new checkpoint/logging instrumentation for that control. It does not establish which historical environment or other condition caused the archived/new difference. The earlier [four-seed snapshot](20261017-round2-v2/sequence_memory/partial/comparison.json) remains preserved as a dated intermediate record.
These are recorded as `RETRAINING_DIFFERENCE_REQUIRES_REVIEW`, not silently accepted under a wide tolerance. Published numbers are not replaced or tuned toward. Current environment details are archived: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, CPU PyTorch 2.8.0, two numerical-library threads. Historical canonical RNG/checkpoints remain unavailable; the forthcoming canonical jobs are new seeded protocol replications.

## Failures and repairs retained

The [initial attempt](20261017-round2/REPORT.md) failed before simulation: a sparse worktree created without checkout had an empty index, so no worker script existed. All eight failure logs are retained. Commit `43ce4a6` fixes population of the checkout and adds a passing integration test covering fresh output, reference preservation and cleanup. The corrected run's first four jobs now match. A detached child first failed to establish a heartbeat; the later managed session also became unavailable across turns. Neither a PID nor an old `RUNNING` record proves continued execution. The new workflow moves execution off that session, uploads artifacts even for failed jobs, and archives them with a separate evidence job. The original four matching jobs are not rerun by that workflow.

Dependency guidance, overly broad environment-doctor claims and missing native canonical checkpoint retention were also corrected; see [the suite documentation](../../docs/round2_reproduction.md). Historical sequence JSON `db`/`best_db`/`spread` fields use the superseded target-dropping rule. Fresh results use per-target common-set penalties and explicit failure bounds.

The complete minimum-size training grid, adaptive rare-event tails, pilot study, other secondary studies and matched Mamba are outside this initial suite. The matched-coded and two canonical training jobs never started in the interrupted local attempt. The four unfinished studies are independent jobs in the hosted workflow; a changed learned result still returns a review-required failure status while preserving its evidence. The email to Dr. Khina is [drafted](../../thesis/submission/email_to_khina_round2_resubmission_2026-10-17.md), not sent.
