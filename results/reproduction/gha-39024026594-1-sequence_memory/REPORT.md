# Round-2 reproduction evidence

Run: `gha-39024026594-1-sequence_memory`; source: `c0394f0da5dac449594b89a9794fe251fad75e76`.
Status: **REVIEW_REQUIRED**; updated 2026-10-17T11:00:50.848415+00:00.

Original published files are preserved. Each job has reference snapshots/hashes, output hashes, commands, logs and a comparison. Training differences require review; an unchanged ranking is not equivalence.

| Job | Status | Scope | Evidence |
|---|---|---|---|
| sequence_memory | RETRAINING_DIFFERENCE_REQUIRES_REVIEW | All 3 channels × 4 architectures × seeds 0,1,2; original 40 epochs and 20×20,000-bit budget | [record](sequence_memory/job.json), [log](sequence_memory/run.log) |

## Limits

Canonical learned checkpoints and historical PyTorch RNG state are absent. New seeded canonical training is protocol replication, not recovery of the original weights. The suite reanalyses the complete saved minimum-size grid but does not retrain that entire grid. Unknown-channel adaptive rare-event tails, pilot studies, full-window/bidirectional Mamba variants and other secondary studies are outside this initial suite. Matched Mamba remains unevaluated.

The frozen source, manifest and environment define this run. No numerical result is replaced automatically. Failures, partial outputs and timeouts remain evidence.
