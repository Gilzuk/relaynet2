# Round-2 reproduction evidence

Run: `20261017-round2`; source: `634ec5c3572b0b6848bf5adb3f7cc61251c8c733`.
Status: **REVIEW_REQUIRED**; updated 2026-10-17T08:49:09.598818+00:00.

Original published files are preserved. Each job has reference snapshots/hashes, output hashes, commands, logs and a comparison. Training differences require review; an unchanged ranking is not equivalence.

| Job | Status | Scope | Evidence |
|---|---|---|---|
| common_target_analysis | FAILED | Table 6.7, Appendix 9.6 and minimum-size conclusions; saved-curve reanalysis | [record](common_target_analysis/job.json), [log](common_target_analysis/run.log) |
| canonical_paired_analysis | FAILED | Tables 5.2–5.3 paired comparisons; saved-curve reanalysis | [record](canonical_paired_analysis/job.json), [log](canonical_paired_analysis/run.log) |
| canonical_af_df | FAILED | Fresh AF/DF evaluation: 11 SNRs, 10 trials × 10,000 bits, trial seeds 0–9 | [record](canonical_af_df/job.json), [log](canonical_af_df/run.log) |
| factorized_mlse | FAILED | Full joint/factorized control: L=3,5,7; six SNRs; 2,000 symbols per block; original depths | [record](factorized_mlse/job.json), [log](factorized_mlse/run.log) |
| sequence_memory | FAILED | All 3 channels × 4 architectures × seeds 0,1,2; original 40 epochs and 20×20,000-bit budget | [record](sequence_memory/job.json), [log](sequence_memory/run.log) |
| coded_matched | FAILED | Full matched coded comparison: 30 trials × 500 frames at six SNRs; hard/soft MLP, no Mamba | [record](coded_matched/job.json), [log](coded_matched/run.log) |
| canonical_native | FAILED | Original native protocol, 100 epochs; new declared PyTorch seed, absent historical checkpoints/RNG | [record](canonical_native/job.json), [log](canonical_native/run.log) |
| canonical_3k | FAILED | Original 3K protocol, 100 epochs; new declared PyTorch seed, absent historical checkpoints/RNG | [record](canonical_3k/job.json), [log](canonical_3k/run.log) |

## Limits

Canonical learned checkpoints and historical PyTorch RNG state are absent. New seeded canonical training is protocol replication, not recovery of the original weights. The suite reanalyses the complete saved minimum-size grid but does not retrain that entire grid. Unknown-channel adaptive rare-event tails, pilot studies, full-window/bidirectional Mamba variants and other secondary studies are outside this initial suite. Matched Mamba remains unevaluated.

The frozen source, manifest and environment define this run. No numerical result is replaced automatically. Failures, partial outputs and timeouts remain evidence.
