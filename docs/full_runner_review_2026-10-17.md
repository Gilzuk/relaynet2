# Full-runner review — 17 October 2026

Reviewed PR #185 at `277718407b4620657625661c8195b2414f644174`
against the current round-2 registry and retained evidence. This is a technical
review, not advisor approval or an assertion that every experiment was rerun.
PR #187's statistical corrections and PR #186's VAE prose correction are separate
completed reviews; neither original BER measurements nor historical checkpoints
were changed by this runner review.

## Findings and repairs

| Finding | Reproduced consequence | Repair |
|---|---|---|
| Committed outputs remain in the disposable checkout | A no-op producer returned `MATCH` | Remove the first producer's old file; later shared writers must change the fresh file |
| Verifier return code ignored | Verifier exit 7 still yielded CLI exit 0 | Propagate verifier failure to exit 1; regression exercises the CLI |
| Historical archives interpreted as runnable commands | Dependency cycle; missing required arguments; potential duplicate full-suite execution | Three historical entries are explicit archive references, never fresh reproductions |
| Rare-event replicate lacked its own command | Existing replicate file could be accepted without generation | Separate original/replicate commands with source bases 1000/7000 and distinct paths |
| Shared outputs overwritten in the evidence directory | Earlier comparison hash had no corresponding retained version | Numbered per-step snapshots and hashes retain each version |
| Global RNG and inherited configuration not controlled | Unseeded initialization or inherited matched settings could change a purported repeat | Seed Python/NumPy/Torch before imports, deterministic Torch, recorded before/after states, clear study overrides |
| Existing evidence directory reused | Earlier evidence could be overwritten | Exclusive directory creation; no overwrite |
| Failed producer followed by dependent execution | Later results could mix stale and fresh inputs | Block subsequent producers and verification after failure |
| Top-level text search used for Torch/dependencies | Comments created dependencies; indirect Torch imports missed | AST literal-path hints, explicit shared-writer ordering and conservative local import traversal |
| Dirty working-tree plan executed committed code | Planned and executed versions could differ | Reject tracked uncommitted changes before execution |
| Only declared final JSON retained | Saved weights and auxiliary results could be discarded | Preserve changed files under results, weights and reproduction_checkpoints, including ignored files |

The plan has 73 entries: 68 experiment/derived-result producers, two analyses of
immutable archived evidence, and three archive references. Plan coverage is not
execution coverage. Literal-path dependency inference is explicitly incomplete
for dynamic paths; partial runs disclose omitted upstream producers as committed
inputs. Unsupported nondeterministic Torch operations fail rather than silently
relaxing the RNG contract. Driver-local seeds remain unchanged.

## Interpretation and remaining limits

`BYTE_MATCH` means a successful producer wrote the committed file bytes. It does
not establish statistical equivalence. Changed bytes require inspection because
they may represent scientific results, timing or metadata. A worker error, missing
output, stale output or verifier error cannot produce an overall success exit.
Historical archive references cannot be described as newly executed experiments.

The default bootstrap is 42 and is configurable; the before/after RNG states,
package versions, source SHA, arguments, environment and Torch build are retained.
These controls do not reconstruct missing historical RNG states, establish
cross-platform bitwise reproducibility, or prove that all observed discrepancies
were caused by seeds. Original drivers that keep trained weights only in memory
still do not supply a replay checkpoint. The focused round-2 runner remains the
source of richer model-specific checkpoints and per-field scientific comparisons.

No full-budget training, rare-event or pilot experiment was launched for this
review, and no existing background task was stopped or modified. Validation uses
small committed fixture repositories plus one inexpensive analysis over archived
BER inputs. Its evidence must not be presented as a new BER simulation.

## Validation

The 19 runner regression tests pass, including no-op/stale-output rejection,
shared-file version preservation, partial failure, verifier exit propagation,
archive classification, seed-42 NumPy/Python/Torch repetition on this host,
checkpoint retention, dirty-source rejection and evidence-overwrite rejection.
The complete test suite passes: **386 tests**, one existing ComplexWarning.
The thesis verifier checks **892 values with zero inconsistencies**. Retained-model,
archived-assessment and provenance audits pass with their documented limits.

A real inexpensive CLI integration at immutable source
`fef5916ad7c7049fe5814a9a825ef69de38d5dd3` recomputed ISI penalties from the
committed matched-protocol BER input. The output is byte-identical to the committed
JSON, the verifier passes, and the worktree was removed. It is an analysis check,
not a new simulation or full-budget training run. Source/input hashes, logs,
output, package versions and RNG states are retained in
[`results/reproduce_all/20261017-final-review-isi`](../results/reproduce_all/20261017-final-review-isi).
Other checks and the complete dry-run plan are in
[`results/reproduction/validation/full_runner_20261017`](../results/reproduction/validation/full_runner_20261017).
No remaining blocking defect was found within this reviewed scope; the explicit
historical and full-execution limitations above remain.
