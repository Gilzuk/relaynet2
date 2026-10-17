# Round-2 reproduction evidence

Run: `gha-39024026594-1-coded_matched`; source: `c0394f0da5dac449594b89a9794fe251fad75e76`.
Status: **REVIEW_REQUIRED**; updated 2026-10-17T11:22:00.746303+00:00.

Original published files are preserved. Each job has reference snapshots/hashes, output hashes, commands, logs and a comparison. Training differences require review; an unchanged ranking is not equivalence.

| Job | Status | Scope | Evidence |
|---|---|---|---|
| coded_matched | RETRAINING_DIFFERENCE_REQUIRES_REVIEW | Full matched coded comparison: 30 trials × 500 frames at six SNRs; hard/soft MLP, no Mamba | [record](coded_matched/job.json), [log](coded_matched/run.log) |

## Limits

Canonical learned checkpoints and historical PyTorch RNG state are absent. New seeded canonical training is protocol replication, not recovery of the original weights. The suite reanalyses the complete saved minimum-size grid but does not retrain that entire grid. Unknown-channel adaptive rare-event tails, pilot studies, full-window/bidirectional Mamba variants and other secondary studies are outside this initial suite. Matched Mamba remains unevaluated.

The frozen source, manifest and environment define this run. No numerical result is replaced automatically. Failures, partial outputs and timeouts remain evidence.
