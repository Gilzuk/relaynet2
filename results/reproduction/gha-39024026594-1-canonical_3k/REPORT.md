# Round-2 reproduction evidence

Run: `gha-39024026594-1-canonical_3k`; source: `c0394f0da5dac449594b89a9794fe251fad75e76`.
Status: **REVIEW_REQUIRED**; updated 2026-10-17T10:25:58.011282+00:00.

Original published files are preserved. Each job has reference snapshots/hashes, output hashes, commands, logs and a comparison. Training differences require review; an unchanged ranking is not equivalence.

| Job | Status | Scope | Evidence |
|---|---|---|---|
| canonical_3k | RETRAINING_DIFFERENCE_REQUIRES_REVIEW | Original 3K protocol, 100 epochs; new declared PyTorch seed, absent historical checkpoints/RNG | [record](canonical_3k/job.json), [log](canonical_3k/run.log) |

## Limits

Canonical learned checkpoints and historical PyTorch RNG state are absent. New seeded canonical training is protocol replication, not recovery of the original weights. The suite reanalyses the complete saved minimum-size grid but does not retrain that entire grid. Unknown-channel adaptive rare-event tails, pilot studies, full-window/bidirectional Mamba variants and other secondary studies are outside this initial suite. Matched Mamba remains unevaluated.

The frozen source, manifest and environment define this run. No numerical result is replaced automatically. Failures, partial outputs and timeouts remain evidence.
