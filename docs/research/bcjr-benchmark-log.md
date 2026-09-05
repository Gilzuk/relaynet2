# Research log — BER-optimal (BCJR/APP) benchmark for the unknown-ISI channel

Tracking log for the open item the thesis names in three places:
Chapter~7 (`sec:qpsk-unknown-channel`), Appendix F, and Future Work item 4.
Append one dated entry per session. Newest last, so the file reads as a
chronology.

**Status: Phase 1 complete, Phase 2 blocked on network egress, experiment not
started.**

**Question.** On the 3-tap unknown-ISI relay channel, does the genie-CSI
classical benchmark's margin over the learned relay change when the benchmark
is made BER-optimal (BCJR/APP over the same trellis) rather than
sequence-optimal (Viterbi MLSE), at BPSK and at Gray-coded QPSK?

The question is the thesis author's own Future Work item 4, scoped. It is not
a generated research question.

---

## 2026-09-05 — Phase 1: scoping

Ran the `deep-research` skill's Phase 1 (RQ brief, methodology blueprint,
Devil's Advocate checkpoint).

### Scope

| | |
|---|---|
| In scope | 3-tap ISI channel of Ch.7; BPSK and QPSK; uncoded; relay-output BER; same seeds and trial protocol as `qpsk_trellis_controls.py` |
| Out of scope | turbo / iterative relay-destination exchange; reduced-state BCJR approximations; the canonical memoryless channel (MAP and MLSE coincide there); any coded chain |

### FINER

| Criterion | Score | Note |
|---|---|---|
| Feasible | High | Trellis, channel classes, rare-event estimator and CI machinery already exist in-repo |
| Interesting | Medium | Interesting to examiners; the answer is likely undramatic |
| Novel | **Low** | BCJR is Bahl et al. 1974. Novel *within this thesis*, not to the field |
| Ethical | n/a | Synthetic simulation |
| Relevant | High | Closes a gap the thesis itself names three times |

The low Novel score is the load-bearing one: this is completeness work, not a
publishable contribution. Budget it accordingly.

### Methodology blueprint

Positivist, quantitative. Monte Carlo simulation, not literature synthesis —
the question is settled by running code; a literature phase only establishes
whether anyone has already answered it for this configuration.

Reuse `ComplexISIRayleighChannel` and the existing trellis classes so the new
comparator differs from the published one in exactly one respect: the branch
metric marginalizes per-bit posteriors instead of maximizing over paths.

**The trap, which has already caught this thesis once.** The withdrawn QPSK
reversal happened because the benchmark was built from `h` alone on a channel
that also fades per symbol. A BCJR built the same way would reproduce that
error and look like a finding. The detector must take the per-symbol gains, as
`FadingAwareViterbiQPSKRelay` does.

**First thing to run is a control, not the comparison:** BCJR and Viterbi must
agree to within Monte Carlo error once the fading is removed. If they do not,
the implementation is wrong, not the theory.

Validity criteria, taken from this repo's existing standards:

- Hierarchical CIs across seeds, `t(2,0.975)=4.303`, not pooled 1.96
- Rare-event estimator with error counting at 16/18/20 dB; rule-of-three where no error is found
- Results written into `results/`, never `/tmp/`
- A row added to `provenance_audit.py` REGISTRY
- A verifier check for any number that reaches the thesis, negative-tested before being trusted

### Devil's Advocate — Checkpoint 1: PASS with one scope revision

1. **The answer is probably "no".** MAP over MLSE on a short trellis is
   typically a fraction of a dB. At 20 dB the fading-aware trellis already
   reaches 0.0001 against the MLP's 0.0508. A BER-optimal detector can only
   widen that. Expect a null result that closes a gap, not a reversal.
2. **Measurability.** At BER ~1e-4 the MAP-vs-ML difference may be smaller than
   the confidence interval. *Scope revision accepted:* state the minimum
   detectable difference before running; if the CI cannot resolve it, report
   the bound, not a point estimate.
3. **Wording.** BCJR is BER-optimal *given the true channel model*. The claim
   is conditional on genie CSI. Appendix F already words this correctly; keep
   that wording.

---

## 2026-09-05 — Phase 2: investigation — BLOCKED

### What was attempted

Systematic search for prior work comparing learned/neural detectors against
BCJR on ISI channels, to establish whether this comparison already exists in
print for a configuration close enough to matter.

### Outcome: cannot complete to the skill's evidence standard

`WebSearch` works. Independent verification does not: `arxiv.org`,
`en.wikipedia.org`, `api.semanticscholar.org` and `api.crossref.org` are all
refused by this environment's egress proxy (403 on CONNECT / EGRESS_BLOCKED).
No candidate source could be confirmed against its own record.

The skill's IRON RULE #4 is that a source which cannot be confirmed is a
**FAIL**, not an "uncertain". So nothing below may be cited, in the thesis or
anywhere else, until someone opens these on an unrestricted network.

### Candidates — verification deferred to CI

The list lives in `docs/research/bcjr-candidates.json`. It is checked by
`scripts/verify_citations.py`, run by the `verify-citations` GitHub Actions
workflow on a runner with open network access, which writes the results back
into the block below. Until that block says otherwise, nothing here is citable.

<!-- VERIFICATION:BEGIN -->
_Not yet run on CI._
<!-- VERIFICATION:END -->

Titles as reported by search, none confirmed at the time of writing:

Returned by search, titles and identifiers as reported, none confirmed:

| Identifier | Title as reported | Why it may matter |
|---|---|---|
| arXiv 2006.01125 | Neural Network-Aided BCJR Algorithm for Joint Symbol Detection and Channel Decoding | Closest analogue: neural augmentation *of* BCJR |
| arXiv 2405.10814 | Data-Driven Symbol Detection for ISI Channels with Bursty Impulsive Noise | Trellis-based soft detection without full CSI |
| arXiv 2411.01517 | Enhancing LMMSE Performance with Modest Complexity Increase via Neural Network Equalizers | Reported to frame BCJR as the optimal ISI equalizer |
| arXiv 2203.16417 | Low-complexity Near-optimum Symbol Detection Based on Neural Enhancement of Factor Graphs | Near-optimum framing against a MAP baseline |
| arXiv 2401.12645 | On the Robustness of Deep Learning-aided Symbol Detectors to Varying Conditions and Imperfect Channel Knowledge | Mismatch regime, the thesis's own framing |

### The one finding that survives non-verification

Four independent search results converge on the same convention: **in this
literature, neural and learned detectors are benchmarked against BCJR**, and
papers describe closing "the BER gap to BCJR" as the standard target.

Convergence across independent results is weaker evidence than a verified
citation, and it is not admissible in the thesis. But it does bear on
expectations: benchmarking against MAP appears to be the field's normal
practice, which raises the priority of Future Work item 4 from
thesis-completeness to something an examiner familiar with the area may
actively expect. That conclusion needs no citation, because it is a statement
about what to expect from a reviewer, not a claim about the world.

### Next actions

- [ ] Re-run Phase 2 from an unrestricted network; verify or drop each candidate above
- [ ] Decide whether the experiment proceeds without the literature phase (it can — the question is settled by simulation)
- [ ] If proceeding: implement the fading-aware BCJR, run the fading-removed control **first**
- [ ] State the minimum detectable difference before running the comparison
- [ ] Register the new experiment in `provenance_audit.py` REGISTRY
