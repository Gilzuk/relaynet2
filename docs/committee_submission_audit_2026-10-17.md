# Committee-style submission audit — 17 October 2026

## Recommendation

**The six supplied round-2 scientific requests are addressed within the stated
experimental scope.** This new review found additional documentation and presentation
corrections, now prepared in this change. It found no new numerical contradiction in
the principal thesis results and no basis to call the historical measurements
incorrect merely because additional training realizations differ.

My technical recommendation is to submit the corrected document once the reviewed
changes are integrated and the page-count convention is confirmed. This is a
skeptical technical assessment, not an appointment as an examiner, advisor approval,
a predicted grade, proof of acceptance, or evidence of submission. The remaining
administrative uncertainty below prevents an unconditional submission clearance.

## Frozen scope

- Source baseline: `Gilzuk/relaynet2` main at
  `5eb34ab3ed01bc9df17498358656965af2fa916b`.
- Published thesis mirror reviewed: `Gilzuk/relaynet2-thesis` main at
  `0278959d9e981f0e0b8d8e7dcef0c8d3dab1c1a9`, generated from that source baseline.
- Baseline PDF SHA-256:
  `9be23d2c30ad49d793ec5ab029c12d669d4f91a02164ab7f6fea70314406207c`.
- Corrected PDF SHA-256:
  `4262e711607c6cdf1e0cd877754b7883ee3300305118ddb45e0c1aeea7c206c6`.
- Both main READMEs were reviewed. The mirror correction is made in its authoritative
  generator, `scripts/overleaf_project.py`, and included in both rebuilt bundles.
  Publication of that generated content must follow source integration; this report
  does not claim that an unmerged branch is already on either repository's main.

The review covered the abstracts and conclusions; the six feedback topics; model,
normalization and bit-label conventions; evidence/retraining interpretation; main
README claims, examples and links; document structure and selected rendered pages.
The numerical verifier has declared coverage, not exhaustive sentence coverage.
The bibliography's earlier identifier review was consulted, not rerun as a new
systematic literature search. No full-budget training was launched or stopped.

## New findings and corrections

| Severity / location | Finding and evidence | Disposition |
|---|---|---|
| Major documentation — source README | Claimed an inherent 16-QAM I/Q-classification error floor, infinite improvements from zero observations, statistical matching to DF and a literature-wide first. A finite-SNR point cannot establish a floor or equivalence; separable square-QAM likelihoods permit independent-axis decisions. The README's MLP 4/16-class 20-dB values 0.00811/0.00002 do not match the current JSON's 0.05491/0.03840, and its zero DF value differs from 0.03748 in that file. The historical run behind those README values was not established. | Removed those stale numerical summaries from the current entry point and linked the preserved historical data with explicit limits. No raw JSON, trial, result figure or thesis BER table changed. |
| Major documentation — source README | Matched coded conclusion said “both learned relays,” although only two MLP read-outs were matched; the original comparison also included Mamba. | Names hard/soft MLP explicitly and states matched Mamba was not evaluated. |
| Reproduction — source README | Example passed unsupported `channel` and `num_bits` arguments and unpacked the three-array return as rows of four values. The old example raises a TypeError. Generic dependencies and the legacy 18-job runner were presented as the whole reproduction route. | Executable CPU example uses the current signature and a separate interval calculation. Two runs match all returned arrays. Added pinned validation environment and distinct saved-evidence, focused-round-2 and broader-plan commands. |
| Scientific scope — both READMEs | Legacy scope omitted faded QPSK / coded 16-QAM controls, overstated shared context and full BPSK/QPSK generalization, and treated simplified Mamba descriptions as reference architectures. The mirror overview lacked the later rerun limits and evidence links. | Aligns scope with the PDF, names simplified implementations, six-sample centre readout and block normalization, and separates historical results from additional training realizations. |
| Reproduction — both READMEs | Seed 42 could be mistaken for full historical determinism; the full plan could be mistaken for completed execution. | Distinguishes focused bootstrap 20261017 from broad-runner bootstrap 42 and driver-local seeds; discloses missing historical states, missing coded weights, repeat-seed limits and the unexecuted full plan. |
| Minor manuscript — §4.1.1, p. 27 | General BPSK formula labels zero as +1, while `relaynet/modulation/bpsk.py` labels zero as −1. QPSK and the fixed-ISI training generator use the former convention. Both are valid, but a fixed sample sequence must follow its driver. | Explicitly states both maps and decision rules. No model or experiment code changed. |
| Minor format — front matter | Acknowledgments followed the English abstract. Faculty instructions place acknowledgments before the first-language abstract. | Reordered the two pages and rebuilt the contents/PDF; body pagination is unchanged. |
| Minor repository metadata | The source README linked to a nonexistent LICENSE file, despite the MIT classifier in setup.py. | Removed the broken badge/link and states the metadata/file discrepancy. No license text or permission grant was invented. |

Historical implementation checkpoints and exploratory results remain in the repo
and git history. Removing an unsupported summary does not erase those records.
The source README's compact proof-of-claims section remains synchronized with the
thesis and covered by its existing drift checks.

## Round-2 closure matrix

| Dr. Khina request | Current evidence / decision |
|---|---|
| Actual canonical training and evaluation | §4.3.3/Table 4.2, pp. 42–43, and `results/canonical_run_provenance.json`: architecture, sample/window counts, train/test channels, scaling, epochs, optimizer and checkpoint rule, seeds and budgets. Historical library versions, full training RNG states, weights and epoch losses are explicitly unknown. **Addressed by reconstruction plus disclosure**, not an exact learned replay claim. |
| Comparable SNR penalties | Table 6.7 and Appendix 9.6 retain reference-defined targets and failed-target bounds. Historical QPSK Mamba-2 best is +5.83 dB; the failing initialization is >+9.11 dB and spread >3.28 dB. Ranking/spread and finite-grid minimum-size conclusions were rechecked from archived curves. **Addressed.** |
| Matched-decoding scope | §5.5.1/Table 5.4 and the diagnostic subsection distinguish two MLP read-outs from unrerun Mamba. New README wording now agrees. **Addressed without inventing a missing Mamba comparison.** |
| Formal problem and BER objective | §4.1.5 separates the local output and whole-block normalization, including per-axis and Hybrid dependencies; empirical hard-decision loss is distinguished from noise-averaged expected BER. **Addressed.** |
| Corrected results lead | Matched coded decoding leads §5.5. Factorized trellis counts are in main Tables 6.8–6.9, with the real-tap/separability condition stated. Unweighted/joint comparisons remain diagnostics. **Addressed.** |
| Causality, context, implementation, caption, archive, headers | Remark 4.1 and §4.4 distinguish causal local scans from full-block transmission and common length from common context. Simplified models, unresolved ten-pilot result, immutable code snapshot and all Chapter 5 headers were checked. **Addressed.** |

## Committee criteria and scientific limits

The faculty's examiner guidance asks reviewers to assess scientific content/method,
structure and presentation. Under those criteria, the defensible contribution is a
bounded empirical study of when small learned relays help specified receiver
pipelines, together with corrected classical comparisons. It is not a new universal
optimal detector or a demonstration that neural relays displace matched MLSE.
The objectives/hypothesis outcomes, methods and discussion now consistently state
that narrower contribution.

The following remain limitations of the evidence, not missing claims to fill with
assumptions:

- Historical canonical learned weights/RNG/software versions are missing. Exact
  learned replay is not established. The known configuration and unknowns are stated,
  as the feedback explicitly permits.
- The four additional hosted jobs are completed training replications with learned
  differences and matching classical controls. Sequence IDs 0/1/2 and coded seed 0
  repeat; these are not six independent sequence initializations or an isolated
  changed-seed experiment. No seed-only causal attribution is justified.
- Forty-eight new canonical/sequence checkpoints are retained. Coded classifier
  weights were not saved. The independent native MLP seed-42 repeat validates only
  that configuration on that host.
- Paired tests use integer error-count differences, remain exploratory and have no
  multiplicity correction. Fixed-checkpoint intervals omit retraining variation.
  Similar values or unchanged rankings do not establish equivalence.
- Pilot studies, the complete minimum-size training grid and adaptive rare-event
  tails were not newly rerun by the focused suite. Matched Mamba remains unevaluated.
  The 73-entry broader plan has not been executed end to end.
- Unknown-channel benefits remain within training families. BCJR is relay-output
  only; operation counts and interpreter timings are not hardware/energy results.

These are also the points the author should be ready to defend orally: why the
matched classical results changed the interpretation, what seed controls prove,
where temporal information enters, which side information each detector receives,
and why arithmetic counts do not by themselves establish a deployment advantage.

## Formal submission uncertainty

The December 2025 faculty instructions specify a maximum of 120 pages including
appendices and separate Roman/front-matter from Arabic/body numbering. They do not
explicitly state whether front/back matter is excluded from that maximum. The repo's
local rule assumes exclusion. The actual corrected PDF is **120 numbered body pages,
136 physical pages**: Chapter 1 begins at physical 14, body folio 120 is physical
133, and Hebrew back matter occupies 134–136. No written faculty clarification of
the exclusion was found in the reviewed evidence.

Therefore this review neither declares a 16-page violation nor certifies the
exclusion. Obtain or retain the secretary/advisor's confirmation of the accepted
count before formal submission. The earlier local readiness verdict should be
read subject to this explicit administrative qualification. Margins and font size
were not reduced to force a count.

The English/Hebrew covers identify the school and degree, the two abstracts are
present, the contents/symbol/figure/table lists are present in order, and the body
ends with references and appendices before the Hebrew matter. Acknowledgments now
precede the English abstract. Advisor approval and actual submission remain human
steps; no email was sent in this review.

Primary faculty sources, checked 17 October 2026:

- [Thesis instructions, December 2025](https://engineering.tau.ac.il/sites/engineering.tau.ac.il/files/media_server/Engineering/Tagel/masters_essay_guidelines.pdf), §§A–D.
- [Examiner guidance](https://engineering.tau.ac.il/tezot3).
- [Oral examination procedure](https://engineering.tau.ac.il/tezot4).

## Verification and retained evidence

Fresh audit checks: **892 numerical values / zero inconsistencies**, provenance
pass, archived comparison/hash assessment pass, and strict model audit pass with
48 checkpoints / 70 output hashes and its documented coded-weight limitation.
After the edits, **32 document/verifier regression tests pass**. The earlier
386-test full-suite result is unchanged-code evidence, not a new full-suite run.

A clean staged XeLaTeX/latexmk build succeeded with no undefined references/citations,
missing characters or overfull boxes. Existing font-feature and repeated longtable
anchor warnings remain. The first in-place build attempts lacked local TeX package
paths and left incomplete auxiliary files; the final build used a clean source
stage with the already installed compatible TeX paths. No manuscript workaround
was made for those environment errors.

Only physical pages **3, 4, 5 and 40** have changed extracted text relative to main:
acknowledgments, abstract, contents and BPSK convention. All result-page text remains
unchanged. Visually inspected pages **1, 3, 4, 40, 55, 69, 92, 102, 133 and 136**.
No clipping, overlap or broken layout was observed on those pages. Chapter 5 headers
on physical pages 64–76 are correct. PDF, preview and both bundled PDFs are identical;
both bundled READMEs match the authoritative generator.

The corrected README example repeats exactly on the CPU; all its relative links
resolve. All **21 existing advisor-email quotations** still match the corrected
PDF on the same printed body pages. The email's PDF pointer/hash must identify this
corrected artifact when it is published; the draft remains unsent.

Logs, numerical mismatch evidence, page checks, SHA-256 identities and quote checks
are retained in [committee validation evidence](../results/reproduction/validation/committee_20261017/).
This audit performs no new learned training and changes no historical experimental
result, figure input, model, runner or background-job state.
