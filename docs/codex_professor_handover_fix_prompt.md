# Professor-handover fix guidelines

## Reusable implementation prompt

Review and correct the active thesis compiled by `thesis/main.tex` against the
committed simulation code and result artifacts. Work on a fresh `codex_` branch
from current `main`. Preserve historical data, checkpoints, and unrelated work.
Do not merge the PR or publish the thesis on the author's behalf.

1. Trace each substantive conclusion to its experimental configuration,
   measurement endpoint, information supplied to the detector, and uncertainty.
   Distinguish a theoretical result, an observation, and a proposed explanation.
   Remove unsupported causal or universal conclusions; do not add new claims.
2. Use one explicit noise convention: for unit symbol energy and gamma=Es/N0,
   real AWGN has variance 1/(2 gamma) and complex AWGN has total power 1/gamma.
   Keep textbook CSI-assisted AF formulas separate from the simulated relay.
3. Correct the adaptive complex-Rayleigh noise scaling and add regression tests.
   Trace its callers. Any affected results must be rerun under the existing
   protocol into separately named `codex_` artifacts, or explicitly excluded from
   the revised quantitative claims. Never relabel old measurements as reruns.
4. Use the matched-protocol Viterbi artifacts for BOTH the central table and its
   original figure. Keep the fixed-budget MLP tail and omit unconfirmed tail
   estimates. Zero observed errors mean insufficient data to estimate a positive
   BER, not zero true BER. State budgets and qualify rule-of-three bounds as
   nominal independent-Bernoulli references where bit errors can be correlated.
5. Treat the QPSK BCJR study as a relay-output detector control, not an end-to-end
   BCJR relay experiment. Compare methods only at the same endpoint. Do not use
   overlapping confidence intervals as an equivalence test. Do not infer an
   error-propagation mechanism, minimum architecture, or hardware speedup from
   experiments that did not isolate or measure it.
6. Reconcile abstract, objectives, methods, results, discussion, summary,
   appendices, and Hebrew abstract. Correct parameter/MAC/FLOP terminology and
   training/test SNR ranges. Retain limitations rather than treating them as
   unresolved implementation promises. Compress redundant exposition without
   deleting unique evidence, references, or required labels.
7. Check cited metadata against primary records and check whether each inspected
   source supports the attached claim. Remove internal citation-verification
   notes from the printed bibliography. Report the actual citation-audit scope;
   do not claim that every source was read if it was not.
8. Rebuild the original data figure reproducibly in vector PDF and 300-DPI PNG,
   using distinguishable markers and colorblind-safe colors. Rebuild main.pdf
   from the changed sources; inspect actual rendered pages for clipped tables,
   unresolved references, unreadable plots, title/abstract ordering, and length.
   Follow current faculty instructions without shrinking mandated type/margins.
9. Strengthen numerical checks so censored values and scientific notation cannot
   pass with inappropriate absolute tolerances. Run regression tests, table and
   provenance audits, and the complete relevant test suite. Record exact results.
10. Update project memory last, commit source and generated PDF together, push,
    and open a PR with the review findings, changes, validation, and any residual
    author decisions. Call it merge-ready only after actual required checks pass.

## Initial review evidence (2026-09-16)

Baseline: `origin/main` at `e303c37`; clean tree. Existing checks passed: 285
tests; 527 checked table cells; no unresolved active-source labels/citation keys.
Those checks did not establish scientific or PDF readiness.

| Priority | Finding | Required disposition |
|---|---|---|
| High | Complex AdaptiveRayleighChannel noise power is half the declared N0 | Fix/test; audit affected composite AF measurements |
| High | Main text retains superseded 8-dB Viterbi comparison | Use matched 0.001354/0.001460; remove old crossover claim |
| High | Central figure and table consume different Viterbi data | One authoritative source and explicit zero-error budgets |
| High | Relay-output BCJR control is compared with destination BER | Separate endpoints and narrow interpretation |
| High | CSI-assisted AF BER equation conflicts with BPSK noise convention | Correct formula and distinguish simulated AF |
| High | Coded-relay explanations are written as established causes | Report observed ordering; label untested mechanisms |
| High | Committed PDF is stale and clips the main results table | Rebuild and inspect actual PDF |
| Medium | Statistical bounds, interval overlap, replication and cost claims overreach | Qualify scope and correct arithmetic/terminology |
| Medium | Table verifier accepts misleading censored/scientific values | Add failing mutation tests and enforce budgets/precision |
| Medium | Submission formatting/date and page-count checks incomplete | Verify current faculty rules and rebuilt document |

Primary references used for targeted checks include the
[TAU thesis instructions (December 2025)](https://engineering.tau.ac.il/sites/engineering.tau.ac.il/files/media_server/Engineering/Tagel/masters_essay_guidelines.pdf),
[Tse and Viswanath, Chapter 3](https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter3.pdf),
[BCJRNet source record](https://arxiv.org/abs/2401.12645),
[neural BCJR source record](https://arxiv.org/abs/2006.01125),
[bursty-channel BCJR source record](https://arxiv.org/abs/2405.10814), and
[MambaCSP source record](https://arxiv.org/abs/2604.21957).
The latter first appeared after the manuscript's old March 2026 title date.

This is a fix specification, not a claim that implementation or final validation
has already completed. The PR records the final validation status.
