# Final skeptical review and model triage — 17 October 2026

Reviewed starting revision: **d29f8ee96664e5df8baa21a7e02e2a91ca9c665d** on PR #184.
The author authorized committing, then requested another skeptical/model audit and
explicit scrutiny of unsupported statements and missing reproduction details.
This is a technical audit, not approval by Dr. Khina or a claim that a separate
professor/model independently reviewed the manuscript.

## Verdict

All six supplied round-2 requests are addressed within the evaluated scope after
the corrections below. No additional numerical-table correction was identified
in this pass. **Exact reproduction is not established for every learned result**:
the historical canonical RNG/checkpoint/environment gaps remain, and the new
matched-coded job did not retain its trained classifier weights. Those are now
explicit in the methods and researcher protocol, rather than hidden by a blanket
reproducibility claim. Scientific causation of all learned differences remains
unresolved. The original measurements are preserved, not declared wrong because
an additional training realization differs.

This verdict covers the identified claims, methods, archived evidence and stated
checks. It is not an exhaustive certification of every sentence, reference,
secondary experiment, software backend or possible scientific interpretation.

## Findings corrected

| Finding at the audited starting revision | Evidence and correction |
|---|---|
| Early Hop-2 description called the local map the transmitted symbol, conflicting with block normalization. | Ch. 4 now distinguishes local `u`, transmitted block `N(u)`, equalized input and Hybrid block switching consistently with §4.1.5. |
| Overview generalized perfect CSI and Hop-2 fading to studies with different information/channel assumptions. | Scope perfect CSI to canonical receivers; identify genie taps versus learned observations and refer to the per-study channel table. |
| Xavier initialization wording covered parameters it did not initialize. | Linear layers are distinguished from S6 random-normal `A_log`/`D`, SSD log-arange/ones, and library-initialized attention/LayerNorm/VAE parameters. Checked the implementation and pinned environment. |
| Coded input was described as interleaving I/Q windows. | `MLPQPSKClassifierRelay._extract_windows` concatenates the entire I window then Q. Corrected that order and the appendix's overgeneralization of joint input to axis-wise models. |
| Main matched-coded discussion lacked a compact, complete training recipe. | Added 42→16→4 topology, activations, 1,212,000 examples, padding boundaries, code generators/tails, loss, precision, Adam options, epochs/batch, final selection and all relevant seeds. Source: coded driver, training-data generator, NumPy classifier and convolutional encoder. |
| Checkpoint-system capability read as an assurance that all evaluated weights were retained. | Actual inventory: 12 canonical and 36 sequence checkpoints; coded archive has trial JSON only. Missing historical and coded weights, optimizer/resume limitations are explicit. |
| Nominal equal parameter count was claimed to remove capacity as an explanation. | Caption/goal now state approximate control of nominal count, retaining the width/depth/window/state/optimization confounds. S6's unused C projection is explicitly included in its nominal count. |
| cGAN rationale asserted reconstruction-loss dominance and convergence compensation without measurement. | State design intent/settings and absence of reported ablations or measured convergence benefits; cGAN was not evaluated in the reported comparison. |
| Weighted-MLP explanation was written as an identified causal mechanism. | Mark it as consistent with trusting relay errors on strong Hop 2, not an ablation-established cause. |
| Rate-adaptation prose generalized finite-grid optimality and ruled out relay contributions to the remaining goodput gap. | Restrict uncoded 16-QAM preference to tested MCSs/SNRs. Separate the constellation's 4-bit ceiling from residual FER and unmeasured alternatives; retain the single-link/two-hop distinction. |
| Reproduction guide implied universal same-host seeded identity and advised comparing only best-of-three values. | Require complete seed curves/censoring/spreads by environment; same-host evidence is configuration-specific. Differences need not be last-digit noise. |
| Guide failure advice could encourage overwriting history to match a new run. | Preserve both records, investigate configurations and uncertainty, and require evidence for changes. Added an exact researcher protocol with Linux/fork requirement, dependency-set distinction, source versions, commands, budgets, RNG call order and missing-record inventory. |

## Round-2 acceptance matrix

| Dr. Khina request | Current evidence / disposition |
|---|---|
| Actual canonical configurations and explicit uncertainties | §4.3.3, Table 4.2 and `results/canonical_run_provenance.json`; architecture inventory cross-checked against retained new weights. Historical Torch states/weights/versions/losses unavailable and disclosed. |
| Comparable SNR penalties, rankings, spreads and size conclusions | Fixed reference target sets, per-target censoring, corrected `results/penalty_common_targets.json`, Tables 6.7 and Appendix 9.6. The new assessment preserves old/new values separately; no dropping failed targets. Minimum size means smallest tested within the stated grid, not a mathematical bound. |
| Matched-Mamba scope | §5.5.1 explicitly covers hard/soft MLP only; no matched-Mamba result is invented. New coded training details and missing-weight disclosure added. |
| Formal problem and differentiability | Local map plus normalization consistently specified; empirical hard loss separated from potentially differentiable noise-averaged BER. |
| Correct comparisons lead | Matched coded results lead; factorized trellis counts are in the main complexity comparison. Unweighted/joint results remain diagnostics. |
| Remaining scope/presentation | Past-only Mamba local scan distinguished from block normalization; common input context qualified; simplified implementations named; ten-pilot difference unresolved; immutable archive identified; rendered Chapter 5 headers corrected. |

## Model-by-model triage

[`scripts/audit_round2_models.py`](../scripts/audit_round2_models.py) produces
[`results/round2_model_checkpoint_audit.json`](../results/round2_model_checkpoint_audit.json).
It verifies all 70 inventoried output-artifact hashes, strictly loads all 48
retained model state dictionaries, matches architecture representations and
parameter/window inventory, checks tensor finiteness and repeated evaluation on
fixed inputs. It does not train or recompute BER. Input hashes pin the inspected
source. The coded classifier is constructed only for an untrained inventory check.

| Model | Native / 3K nominal parameters | Triage |
|---|---|---|
| MLP | 169 / 3,004 | Shapes and count match; source confirms canonical AWGN training and per-axis deployment. Sequence uses its separate channel-trained protocol. |
| Hybrid | 169 / 3,004 | Distinct MLP weights; block-wise power estimator chooses branches. It does not receive the nominal SNR as an oracle. |
| VAE | 1,777 / 3,037 | Compatible weights; evaluation decodes latent mean. Training sampling and default layer initialization remain distinct. |
| Transformer | 17,697 / 3,007 | Unmasked attention, center readout, train-only dropout, optional input normalization off; future context differs from the causal scans. |
| Simplified S6 | 24,001 / 3,027 | Center output has zero change in the fixed future-perturbation probes. Backpropagation confirms C-projection disconnection: 2,080 native / 198 3K parameters included in the nominal totals. No model surgery or retroactive count substitution. |
| Simplified SSD | 26,179 / 3,004 | Compatible weights, center-readout future probes unchanged; no disconnected parameters in those backward probes. Chunk/gate implementation is not a reference Mamba2 kernel. |
| Matched-coded MLP | 756 | Untrained topology/count/float64 and 1,212,000-example recipe checked. No trained coded weights in the archived artifact inventory. Hard/soft rows reuse one learned classifier. |
| cGAN | Not evaluated here | Implementation description only; no performance/convergence endorsement from these runs. |

The causality probes cover all 22 retained S6/SSD checkpoints; source inspection
explains their causal scan. Finite probes are not a general proof, and local
causality never implies causality of block-normalized transmission. Loading
weights is compatibility evidence, not proof of historical identity or BER
agreement. The existing seed-42 experiment supplies stronger same-host
repeatability evidence for one native MLP only.

## Remaining limits relevant to a professional reproduction

See [`round2_reproduction_protocol.md`](round2_reproduction_protocol.md) for
one-script execution and the full data/seed/checkpoint map. A future coded run
should export its trained classifier; the lost realization cannot be recovered
by adding a save operation after it has ended. The four completed hosted jobs
remain at frozen source c0394f0, with no cancellation, retraining or parameter
changes made during this audit. Later deterministic flags are not retroactive.

All classical controls agree exactly; learned discrepancies remain additional
realization evidence. Sequence uses the same seed IDs 0/1/2 and coded initialization
stays 0, so differences cannot all be attributed to changed initial seeds.
Repeated IDs are not pooled as independent new seeds. Small differences,
unchanged order and nonsignificant tests do not establish equivalence. The
minimum-size training grid, pilots, adaptive tails and matched Mamba remain
outside these four reruns. Historical records and figure inputs remain intact.

## Validation and delivery

- Saved-rerun assessment and model-audit `--check` commands passed.
- Numerical verifier: 892 checked values, zero inconsistencies. Focused evidence,
  drift, tolerance and document tests: 64 passed in 80.23 seconds.
- PDF rebuilt with XeLaTeX/latexmk; 120 Arabic-numbered body pages, 136 total.
  Chapter 1 starts on physical page 14; body folio 120 is physical page 133;
  Hebrew back matter starts on physical page 134.
- Revised methods, matched-coded recipe, capacity table/discussion, appendix input
  convention and Chapter 5 headers visually inspected. No undefined references,
  citations, missing characters or overfull boxes in the final build. Existing
  duplicate table-destination and Courier font-feature warnings remain.
- PDF, preview and both source bundles rebuilt together. New audit output is
  registered in the provenance ledger, which passes after committing it.
- Publication uses `[skip ci]` to avoid relaunching full-budget experiments from
  the cumulative PR diff. No new hosted CI pass is claimed. PR #184 remains draft;
  no merge or advisor email is part of this audit.
