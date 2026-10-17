# Full-runner review validation — 17 October 2026

- Executed source: `fef5916ad7c7049fe5814a9a825ef69de38d5dd3`.
- Runner regressions: **19 passed**. These use tiny committed fixture repositories, including a small Torch layer for seed-42 repetition; they are not thesis-scale learned experiments.
- Full test suite: **386 passed**, one existing ComplexWarning.
- Thesis verifier: **892 cells checked, 0 inconsistencies**.
- Retained-model audit: **48 checkpoints**, documented absence of coded weights; archived output hashes agree.
- Archived assessment and provenance audit pass.
- The source files tested before publication have the same Git blob hashes as the executed source above; merging PR #186 only added its reviewed prose/PDF changes.
- Integration: `python scripts/reproduce_all.py --run --seed 42 --threads 2 --only 'ISI required-SNR' --run-id 20261017-final-review-isi`.
- Integration evidence: [`results/reproduce_all/20261017-final-review-isi`](../../../reproduce_all/20261017-final-review-isi). Freshly generated penalty JSON is `BYTE_MATCH`; the verifier passes; the worktree was removed. The numerical input is archived matched-protocol BER data, not a new simulation. Source/input hashes and before/after RNG states are retained there.
- The complete 73-entry plan is saved as `full_plan.json`. It was **not executed in full**. Three entries are historical archive references and two analyze immutable archived evidence.

These logs establish the stated code and analysis checks. They do not recover historical missing RNG/checkpoints or establish statistical equivalence across learned realizations.
