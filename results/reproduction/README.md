# Round-2 reproduction evidence

The thesis corrections were merged in [PR 183](https://github.com/Gilzuk/relaynet2/pull/183). Follow-on experiments are kept separately in [PR 184](https://github.com/Gilzuk/relaynet2/pull/184). Original published result files are preserved.

**Evidence snapshot: 17 October 2026, 08:57 UTC.** The active run is [`20261017-round2-v2`](20261017-round2-v2/REPORT.md), simulation source `43ce4a6fbfe2a147a8c77c0fb06e44fcd5d7d5e2`. Check its [status/heartbeat](20261017-round2-v2/status.json) and logs for subsequent progress. A committed GitHub snapshot does not update itself while local computation continues.

| Completed check | Result | Authoritative published JSON |
|---|---|---|
| Common-target penalty and minimum-size reanalysis | MATCH, zero differing scientific fields | [penalty_common_targets.json](../penalty_common_targets.json) |
| Canonical paired analysis | MATCH, zero differing fields | [canonical_paired_tests.json](../canonical_paired_tests.json) |
| Fresh canonical AF/DF | MATCH; per-trial BER arrays exactly equal in both tables | [native canonical](../modulation/qpsk_rayleigh.json), [3K canonical](../normalized_3k/3k_rayleigh.json) |
| Joint/factorized MLSE | MATCH; 264,000 decisions, zero disagreements | [factorized_mlse_equivalence.json](../factorized_mlse_equivalence.json) |

The common-target and paired jobs reanalyse saved data. AF/DF and factorization execute fresh simulations. Each job folder contains raw new output, reference snapshots/hashes, command, environment overrides, log and comparison; `matching_reference` directs matches to the corresponding published JSON.

## Flagged training differences

The full Table 6.7 retraining is running. A [partial snapshot](20261017-round2-v2/sequence_memory/partial/comparison.json) retains three completed ISI MLP seeds and the first ISI Transformer seed, their complete curves and four new checkpoints. The snapshot is explicitly partial; it cannot establish the full architecture ranking.

- ISI MLP best common-target penalty: **1.965266649 dB** in the new run versus **1.973793596 dB** published. Across-seed spread: **0.399144385 dB** versus **0.391174891 dB**. Maximum absolute BER difference over those MLP seed/SNR cells: **0.0003075**. The MLSE baseline matches exactly.
- The first ISI Transformer seed also differs (maximum absolute BER change **0.0001925**). It is not used alone to infer the best-of-three result or architecture ordering.
- An independent [original-driver control](20261017-original-control/comparison.json), retaining the original MLP training/evaluation loop and all three seeds, produced **zero differing fields against the instrumented replay**, while reproducing its differences from the archived run. This rules out the new checkpoint/logging instrumentation as their cause for this control. It does not establish which historical library, thread setting or other condition caused them.

These are recorded as `RETRAINING_DIFFERENCE_REQUIRES_REVIEW`, not silently accepted under a wide tolerance. Published numbers are not replaced or tuned toward. Current environment details are archived: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, CPU PyTorch 2.8.0, two numerical-library threads. Historical canonical RNG/checkpoints remain unavailable; the forthcoming canonical jobs are new seeded protocol replications.

## Failures and repairs retained

The [initial attempt](20261017-round2/REPORT.md) failed before simulation: a sparse worktree created without checkout had an empty index, so no worker script existed. All eight failure logs are retained. Commit `43ce4a6` fixes population of the checkout and adds a passing integration test covering fresh output, reference preservation and cleanup. The corrected run's first four jobs now match. The hosted execution uses a managed asynchronous session because a detached child did not establish a heartbeat after its launcher exited; a PID alone is not evidence of a running experiment.

Dependency guidance, overly broad environment-doctor claims and missing native canonical checkpoint retention were also corrected; see [the suite documentation](../../docs/round2_reproduction.md). Historical sequence JSON `db`/`best_db`/`spread` fields use the superseded target-dropping rule. Fresh results use per-target common-set penalties and explicit failure bounds.

The complete minimum-size training grid, adaptive rare-event tails, pilot study, other secondary studies and matched Mamba are outside this initial suite. The matched-coded and two canonical training jobs remain queued behind the sequence study in this snapshot. The email to Dr. Khina is [drafted](../../thesis/submission/email_to_khina_round2_resubmission_2026-10-17.md), not sent.
