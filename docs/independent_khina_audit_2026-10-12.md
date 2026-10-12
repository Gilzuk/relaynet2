# Independent recheck of the Khina response — 12 October 2026

Baseline: `abed3748779857341ec3ac95aa170fbef872398b` (`main`).

## Verdict and scope

The nine requests as summarized in `HANDOVER.md` have corresponding changes in the active thesis, code or declared limitations. This pass found further wording and documentation defects, corrected by the accompanying changes. It does **not** certify that the thesis is issue-free or that all simulations reproduce from scratch.

The original October nine-comment email is not committed here. This mapping is against the repository's paraphrase and reply. `thesis/ak_comments.json` is an earlier review of a different manuscript configuration; it is not the source of those nine comments. An exact check of every original sub-request requires the original email.

The active manuscript is the include graph of `thesis/main.tex`, not the root manuscript variants, underscored chapter drafts, `docs/thesis_feedback.md`, or `professor_feedback.md`. The latter retains a placeholder signature and cannot establish external academic approval.

## Comment-by-comment evidence

| Comment | Evidence inspected | Verdict and remaining boundary |
|---|---|---|
| 1: relay-design objective and surrogates | Ch. 4 `sec:relay-design-problem`; Ch. 2 conditional-mean discussion | Addressed: information, power, look-ahead, destination rule and BER objective are specified. Local MSE is distinguished from destination BER. Hard-output error composition is limited to the stated uncoded setting. |
| 2: question/hypothesis/experiment hierarchy | Ch. 3 `sec:two-objectives`, `tbl:rq-map`; Ch. 8 answers O1/O2 | Addressed: H1–H4 are the canonical control, H5 belongs to the principal extension; separate studies do not measure one common boundary. |
| 3: AF implementation and conventional reference | `relaynet/relays/af.py`; `af_block_and_conventional.py`; `results/af_block_and_conventional.json`; Ch. 4/7 | Addressed: empirical block normalization and heavy-tailed zero-forced noise are disclosed. Recomputed maximum difference between saved variable-gain simulation and fade-averaged analytical BER: **0.0005573808**, below the claimed 0.001 tolerance. This is an aggregate-data check, not a fresh million-bit simulation. |
| 4: matched coded metrics | `relaynet/coding/convolutional.py`, `relaynet/coding/bcjr.py`, `coded_matched_metrics.py`; Ch. 5 `sec:coded-matched-metrics`; saved trial arrays | Addressed with inferential limitations: per-symbol reliability is included and the original ordering reverses. Recomputed strict block-DF wins: **30/30 against each learned hard/soft relay under both destination metrics at all six tested SNRs**. The first 30 published trials reproduce exactly. The 30-trial budget was retained after inspecting running means; p-values are exploratory. Destination weighting for learned relays models Hop 2 alone. |
| 5: QPSK factorization and crossover | `FactorizedViterbiQPSKRelay`; factorization tests; `results/factorized_mlse_equivalence.json`; Ch. 2/6 | Addressed under real taps and separable noise. Saved equivalence record: **264,000 decisions, zero disagreements**; implementation tests pass locally. Arithmetic conventions are stated; operation counts do not establish hardware timing. |
| 6: delay accounting | Ch. 5 `tbl:table41`; Ch. 6 `tbl:joint-latency`; Ch. 7 resource discussion; reply item 6 | Addressed by separating detector look-ahead, whole-block normalization, scheduling and end-to-end delay. Streaming normalization remains unmeasured; operational readiness is not inferred for normalized soft relays. |
| 7: model and penalty specifications | Appendix 9.4/9.6; `e6_isi_penalty.py`; flat-channel corrected source and committed output | Addressed in the active specifications and declared verifier coverage. The flat-control shortfall at 0 dB is reported. Not every model law was independently rerun in this pass. |
| 8: architecture/training comparisons | Ch. 4 training and parameter-budget sections; Ch. 7 limitations; reply item 8 | Addressed by qualification. **Exact canonical-data regeneration remains unavailable** because initialization and historical training provenance are incomplete. Equal parameter count does not isolate architecture, and held-out accuracy is not the checkpoint-selection objective. |
| 9: prior-work positioning | Ch. 2 `sec:positioning-closest-work`; Ch. 3 `tbl:table17`; Ch. 6 scalar MAP comparison | Addressed as a bounded empirical contribution, not a new forwarding principle. No exhaustive novelty search was performed, so priority claims remain limited to the reviewed literature. |

## Defects corrected by this review

1. **Trial-count ambiguity:** Figure 5.3's caption could assign 100 trials to the matched table. It now distinguishes the 100-trial unweighted figure from the separate 30-trial matched comparison.
2. **Unstated stopping decision:** the generating script records retention of 30 trials after running means were inspected. That detail is now disclosed in the thesis and reply, with the paired tests described as exploratory.
3. **Overstated zero-error conclusion:** the reply's “block DF reaches the oracle” is replaced by equal observed zero-error counts at 24, 28 and 32 dB in six million information bits per point. This does not prove equal underlying BER. No binomial confidence bound is added because coded bit errors need not be independent.
4. **Overgeneralized code-length statement:** the reply now states the measured K=3/5/7 ordering from 4 dB upward and its 0-dB exception, rather than invoking universal monotonic improvement with constraint length.
5. **Documentation authority:** stale manuscript, planning and unattributed endorsement drafts are explicitly marked historical. The older plan's LOS-only description of Cover–El Gamal is flagged as incorrect.
6. **README page count:** the baseline PDF has folio 1 on physical page 14 and folio 119 on page 132; the remaining three pages are Hebrew back matter. Thus the baseline is **119 countable pages / 135 physical PDF pages**, not 120.

## Verification and interpretation

- Baseline numerical verifier: **678 checked cells, zero inconsistencies**. This validates only its declared subset against saved outputs.
- Baseline provenance audit: successful; script/data age exceptions are explicitly reviewed in its registry. It is a history check, not a proof of full reproducibility.
- Local test run: **343 passed, 4 skipped**, with one existing complex-cast warning in the factorized detector's rejected-input path. All four skips concern the unavailable Mamba-2 wrapper in this environment. The repository's earlier “347 tests pass” statement is not an independent result from this environment.
- Recomputed matched block-DF errors at 0/4/8/12/16/20 dB: **805488 / 141862 / 9705 / 610 / 30 / 3**, each over three million information bits. Best learned result at 20 dB: **2667 errors**. Maximum stored unweighted replay difference: **0**.
- The baseline PDF was text-inspected: no `??` markers were found. The XeLaTeX/latexmk rebuild succeeds with 135 physical pages and 119 countable pages, zero LaTeX errors, zero undefined references and zero overfull horizontal boxes. A 6.28-point overfull vertical-box warning occurs on unchanged Ch. 4 folio 43; that page was visually checked with no clipping, as were changed folios 58–59. The local TeX Live 2023 build also reports duplicate table destinations from the existing table setup. These warnings are retained transparently rather than represented as a completely warning-free build. The PDF, preview and both Overleaf bundles were regenerated.

No experimental output or simulator behavior was changed by these corrections. The reply remains a draft; it has not been sent. The author still needs to review the Hebrew abstract, submission date and final reply, and the generated Overleaf mirror must be republished after this PR is merged.

## Primary-source cross-checks

- Cover and El Gamal, “Capacity Theorems for the Relay Channel,” IEEE TIT, 1979: the general formulation is discrete memoryless, not restricted to a LOS physical channel. [Author-hosted paper](https://isl.stanford.edu/~cover/papers/transIT/0572cove.pdf), Sections I and VI.
- Gomadam and Jafar, “Optimal Relay Functionality for SNR Maximization in Memoryless Relay Networks,” IEEE JSAC, 2007: estimate-and-forward optimizes a specified generalized SNR criterion; this does not make local MSE training a general destination-BER optimum. [Author preprint](https://arxiv.org/abs/cs/0510002).

These checks support the scoped framing. They do not establish exhaustive novelty or validate every bibliography entry.

## Publication status

The initial automatic approval review rejected publication pending explicit authorization. On 13 October 2026 the author explicitly approved publishing the corrections, audit, rebuilt PDF and bundles to a new branch and draft pull request in `Gilzuk/relaynet2`. The review remains a draft for the author; the reply has not been sent and the generated mirror still needs publication after merge.
