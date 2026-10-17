# Handover — Khina revision round (task chain)

**Full runner review (2026-10-17):** PR #185's stale-output false match, ignored verifier failure, archive-driven dependency cycle, rare-event replicate command, shared-file evidence loss and missing RNG bootstrap are fixed. The 73-entry plan separates 3 archive references and 2 archive analyses from 68 producers. Default Python/NumPy/Torch seed 42, deterministic Torch, local seed preservation, per-step snapshots, checkpoint retention and failure blocking are documented. 19 runner regression tests pass; full suite 386 passed (one existing warning), numerical verifier 892/0, retained-model and archived-assessment checks pass. PR #186 merged at 200fdaeb7f614743510ea100ec3f706169f0e50a after Python 3.10–3.12 CI passed; its current thesis/PDF are included here. No full-budget experiments launched or stopped. Frozen-source integration at fef5916ad7c7049fe5814a9a825ef69de38d5dd3 regenerated ISI penalties from archived BER inputs: BYTE_MATCH, verifier passed, worktree removed. Evidence and source/input hashes are under results/reproduce_all/20261017-final-review-isi; test/audit logs under results/reproduction/validation/full_runner_20261017. Full-plan execution remains unperformed; missing historical states remain disclosed. Final evidence-only publication and CI are next. Details: docs/full_runner_review_2026-10-17.md.


**VAE review completion (2026-10-17):** PR #187 merged at 97ee4c72f7c09154efaeffa1ea38def073565927. Reviewed PR #186 against that main: Torch infer and NumPy _infer decode the latent mean; sampling is confined to training. Ch. 2 now agrees with §4.3.3 and limits its historical equality/paradigm claims to measured evidence, without implying equivalence or a causal generative-bias effect. Merged current main into the existing PR branch, preserving its history, all evidence and the statistical corrections. Rebuilt PDF/preview/bundles: 120 numbered / 136 total pages; VAE page inspected; verifier 892/0, no critical LaTeX warnings. No experiment or raw result changed. Report: docs/vae_inference_review_2026-10-17.md. PR #185 runner review is in progress; original runner falsely accepts untouched outputs and ignores verifier failure, and its plan cycles against the new registry.

**PR #184 merged; post-merge audit (2026-10-17):** Author requested merge and another audit. PR #184 merged at a714061cbc6abd111beba037a4a955cfeac27b2e with reviewed tree unchanged. New audit/postmerge-184-20261017 branch fixes a real signed-rank tie defect: use integer error-count differences (10,000 canonical / 100,000 coded bits), requiring and validating denominators. Historical 3K Transformer/DF at 20 dB changes p 0.05078125 -> 0.04296875; additional native S6/DF at 16 dB changes 0.03515625 -> 0.0546875. Exactly two significance classifications change; no learned canonical DF advantage, coded DF lead and raw BER data unchanged. Corrected analysis code/derived results pinned at 80b4545adcc66e4be1f435364f7ec927e87f6414; thesis archive now points there. Active 16-QAM scope prose now includes the equal-throughput control as well as adaptation. All 418 ten-pair results agree with independent sign enumeration; full suite 366 passed plus the later exhaustive test (final focused file 4 passed), numerical verifier 892/0, assessment and provenance pass. Merged-snapshot model audit passes all 48 checkpoints / 70 output hashes. PDF remains 120 numbered / 136 total; revised pages inspected and preview/bundles synchronized. Report: docs/postmerge_184_audit_2026-10-17.md. Publish the follow-up branch as a separate PR, not to the merged reproduction branch. No full-budget experiments launched/cancelled/modified, no advisor email, no exact historical reproduction or equivalence claim. Prior draft/unmerged status entries below are historical.

**Final skeptical/model/reproduction audit (2026-10-17):** The author approved commit, requested final audits, then explicitly added unsupported statements and missing researcher reproduction details. Audited base d29f8ee96664e5df8baa21a7e02e2a91ca9c665d. Corrected early normalization/CSI statements, initialization exceptions, I/Q input order, coded training recipe, nominal-budget/cGAN/decoder-causation/finite-grid claims and reproduction instructions. New scripts/audit_round2_models.py strictly loads all 48 retained checkpoints and verifies 70 inventoried output hashes; S6 nominal counts include unused C projections (2,080 native / 198 3K parameters). Newly explicit gap: the hosted coded job retained trial JSON but no trained classifier weights. Historical canonical RNG/weights/versions remain unavailable. No historical data or figure inputs changed, and no training was launched/cancelled. Researcher guide: docs/round2_reproduction_protocol.md; final findings/acceptance matrix: docs/final_skeptical_model_triage_2026-10-17.md. Assessment/model checks pass, numerical verifier 892/0, focused tests 64 passed (80.23s), PDF 120 numbered/136 total; revised pages inspected, no critical LaTeX warnings, preview/bundles synchronized. Commit/push to existing draft PR #184 is authorized; use [skip ci] to avoid duplicate full-budget jobs. This is technical closure within disclosed scope, not exact reproduction of every learned result, advisor approval, a merge, or email delivery. Do not infer that different seeds caused all discrepancies.


**Author-directed interpretation update (2026-10-17):** Treat the October runs as additional training replications and seed-sensitivity evidence. Numerical variation alone does not invalidate original measurements. Section 7.1.4, advisor drafts and reports now say so; all historical tables, plots, archived statuses and prior assessment values are preserved. Canonical bootstrap 20261017 supplies new realizations with unknown historical Torch states; sequence repeats IDs 0/1/2, and coded initialization stays 0. Do not label repeated IDs as independent new seeds or claim all differences are caused by changed initial seeds. The assessment adds seed_design metadata and validates complete sequence seed groups. No new training is launched. PDF remains 120 numbered / 136 total pages; numerical verifier 892/0; all 64 focused tests passed. Existing PR #184 remains draft for author review; this wording update is authorized for publication under the session's standing push instruction. No merge or advisor email requested in this turn.


**Completed rerun review (2026-10-17; supersedes live-status notes below):** All four hosted workers in run 39024026594 completed at frozen source c0394f0, exit 0, scientific status RETRAINING_DIFFERENCE_REQUIRES_REVIEW. Classical arrays match exactly. Archive job succeeded; commit dc7c02c00cbebdfb566362a810ab4b45cb6abbe0 contains every job (index missing_jobs=[]). No experiment was stopped or restarted. Section 7.1.4 now reports the new VAE gap, sequence penalty/spread changes, fresh canonical/coded paired tests and seed-42 control limits; historical tables and plots are unchanged. New scripts/assess_round2_reruns.py checks hashes and produces results/round2_rerun_assessment.json. Verification: 892 values / 0 inconsistencies, 64 focused tests passed; rebuilt PDF 120 numbered / 136 total pages, bundles and preview synchronized, revised pages visually inspected. Existing font-feature/duplicate table-anchor warnings remain, but no undefined refs/citations, missing characters or overfull boxes. Advisor drafts updated and unsent. Publish to existing draft PR #184 with [skip ci] to prevent duplicate experiments; do not claim a new CI pass or merge while discrepancies remain unresolved. Details: docs/round2_rerun_manuscript_update_2026-10-17.md.


**Current state (2026-10-17):** PR #183 is merged at `42e3e386e0b6d1c137a84cf609b51e9581e7e69a`; CI passed at the verified head `b09b20373312823f6a05fcd0efcb0db8048b51a9`. Git CLI push lacked credentials, so the GitHub connector published the same verified tree (`9e1ead5941dae798b238ce3e2571d297d0aca0af`) with a guarded fast-forward. The thesis cites archived code/data/tooling commit `218a1c550cb8b917a95d9545c5211e109985bf38`. PDF 120 countable / 136 total pages; audit 873/0 and 348 passed / 4 optional skips before installing Torch.

**Active task:** draft PR #184 on `experiments/round2-reproduction-20261017`. Local run `20261017-round2-v2` at source `43ce4a6` is **INTERRUPTED_SESSION_LOST**: last heartbeat 09:07:25 UTC, managed session 87408 no longer exists. No process was stopped by the assistant. Recovered all 16 completed seeds and checkpoints, last-live status and logs. Four completed jobs MATCH. Recovered learned curves flag 53 changed BER cells; classical curves match in both reached channels. All BPSK architectures completed with unchanged best-of-three ordering; Transformer best moved 3.519872627 to 3.723647162 dB. The independent original-driver MLP control matches the new replay; historical cause remains unresolved. See `results/reproduction/README.md` and the interruption comparison.

The new `Round-2 reproduction` Actions workflow runs sequence/matched-coded/native/3K as four full-budget jobs, using immutable head SHA and declared CPU environment. It retains failure artifacts and automatically commits evidence to this branch, with no main merge or result substitution. Inspect the latest Actions run and `gha-*-index.json`; do not reuse old session 87408 or infer running status from its old PID. Earlier local training outputs remain evidence; the new host repeats the complete sequence study to avoid mixing environments. Next: inspect hosted status, investigate completed comparisons, and report unresolved differences. Workflow YAML and embedded Bash/Python were syntax-validated; underlying simulation code/budgets are unchanged. Prior validations: full tests358, focused7 after sparse-checkout fix, CI green at43ce.

**RNG audit update (2026-10-17):** historical canonical seed 42 controls training data, not Torch initialization/shuffling. Four full-budget native MLP controls at b4059e6 are archived under results/reproduction/20261017-rng-seed42: legacy repeats differ in 102/110 trial BERs; explicit Python/NumPy/Torch 42 plus deterministic algorithms match initial/final weights and all 110 BERs exactly on this host. Seeded vs published maximum mean-BER difference 0.00150 remains unresolved. Future suite runs declare all seed policies, require deterministic Torch algorithms and save global RNG states; preserve original study-local seeds. Ten focused tests pass, 47 evidence hashes verified. Hosted run 39024026594 stays at frozen c0394f0; no active jobs changed/restarted. Canonical_native completed 10:49:37 UTC with worker exit 0 and RETRAINING_DIFFERENCE_REQUIRES_REVIEW (635 fields, classical controls exact); artifact 12099095123 verified. Sequence/coded still running at the check; automated archival pending. PR #184 remains draft. New commit uses [skip ci] to avoid launching duplicate full-budget jobs during the active run; no CI pass is claimed for that commit. See docs/rng_reproducibility_audit_2026-10-17.md.

**Email:** concise draft `thesis/submission/email_to_khina_round2_resubmission_2026-10-17.md` prepared at the author's request, alongside the detailed response. Neither was sent. The draft performs the resubmission when the author sends it; GitHub publication alone is not an email to the advisor. Earlier status/approval notes below are historical.

**Independent recheck (2026-10-12):** draft #171 (branch `fix/khina-independent-audit-20261012`, now combined with #172 in #173) records additional wording/documentation corrections and a nine-comment evidence mapping in `docs/independent_khina_audit_2026-10-12.md`. The saved coded reversal is confirmed independently, but exact canonical training reproduction and verification against the absent original October email remain limited. That environment ran 343 passes and 4 Mamba-2 skips; where the Mamba-2 module imports, as in the session that combined it with #172, all 347 pass. Republish the generated mirror after this PR merges.

This file is the resume point if a session ends mid-task (out of credits, container reset,
interruption). A new session should read it **after** `CLAUDE.md`, `context.md`,
`memory-bank/activeContext.md` and `memory-bank/progress.md`, then pick up the first task below
that is not marked DONE. Update this file — status, PR number, page count — in the same PR as each
task, or in a small follow-up PR if the task's PR has already merged.

## Source of the work

Dr. Anatoly Khina (advisor) replied to the final-revision email with 9 comments plus presentation
notes. Summary of each comment (paraphrased; page numbers refer to the 120-page PDF he reviewed):

1. State the relay-design problem explicitly (observations/CSI, look-ahead, power constraint,
   destination rule, objective) and explain MSE / cross-entropy as surrogates. Local MMSE is not
   destination-BER optimal.
2. Reconcile RQs, hypotheses and experiments: O1/O2 are about memory, channel information and
   resources, while H1–H4 are about the memoryless architecture comparison, yet Ch. 6 is called the
   principal contribution. Make the hierarchy explicit and **map each RQ to the experiments that
   answer it**. ("Cumulative relaxations" and "distance between boundaries" are already fixed, #128.)
3. AF: zero-forced y/h with whole-block normalization; E[1/|h|²] = ∞. Specify clipping, check block
   length, add a conventional finite-power AF.
4. Coded classical baselines use unweighted distances after zero forcing; implement
   **reliability-weighted (matched) metrics** at the relay and destination Viterbi/BCJR and rerun
   the principal coded comparisons. Inflating the noise variance by a constant does not fix it.
5. QPSK MLSE factorizes exactly into two binary trellises; verify and update the crossover.
6. Delay accounting: separate detector look-ahead, normalization buffering, protocol scheduling and
   end-to-end delay; the 202-symbol buffer belongs to the full-frame decoder (finite traceback
   possible). If 5- or 10-symbol readiness is claimed, specify and evaluate a causal normalization.
7. Specify models: amplifier map and strength, branch-asymmetry distributions, composite-channel
   parameter laws, size-sweep configurations, both hop models, impairment ordering, redraw
   intervals, normalization, pilot construction, window-boundary handling; DBPSK (bits or
   differential symbols? how forwarded?); define the single-number SNR penalties in Tables 6.6–6.7
   and 9.3–9.5 (BER target, interpolation scale, aggregation, out-of-range and zero-error handling,
   checkpoint selection).
8. Equal parameter counts do not equalize training or isolate inductive bias; keep conclusions tied
   to the protocols. Transformer parameter count, ReLU/tanh protocol and 0.07 dB claim (done in
   Phase A). The equal-budget discussion was made consistent with the protocols in B4 (#147).
9. Positioning: explain what the thesis adds beyond estimate-and-forward, Farsad–Goldsmith
   (learned detection without CSI) and ViterbiNet; Table 3.1 (`tbl:table17`) should compare
   **named studies** and properties, not "prior work: ad-hoc / none". Scalar-MAP reference (done, C2).

Presentation notes: running headers (#124), ledger labels (#135), verification-coverage wording
(#134), figures vs principal tables using different values (C5, #151).

## Task chain

| ID | Task | Status | PR |
|----|------|--------|----|
| A  | Quick text fixes (Transformer width, 0.07 dB, tanh/ReLU, ledger labels, verifier coverage, AF disclosure, cost convention, delay wording) | DONE | #131–#138 |
| C1 | Factorized QPSK MLSE implemented + equivalence check (0 disagreements / 264k symbols) | DONE | #139 |
| C2 | Scalar MAP relay on unknown ISI (`e6_scalar_map.py`) | DONE | #140 |
| C3 | AF block-length sensitivity + conventional variable/fixed-gain AF | DONE | #141 |
| —  | Publish mirror after C1–C3 | DONE (f174a3b) | — |
| B0 | Canonical training described as implemented (BPSK/AWGN-trained, no √2 pre-scale) | DONE | #142 |
| B1 | Relay-design problem + surrogates, new Sec. 4.1.5 (comment 1) | DONE | #143 |
| B2 | RQ hierarchy + RQ→experiment mapping table (comment 2); O1/O2 answered by name in Ch. 9 | DONE | #145 |
| B3 | Model specifications + SNR-penalty definitions (comment 7): Appendix G; ISI penalties now derived by `e6_isi_penalty.py` and verified | DONE | #146 |
| B4 | Equal-budget wording pass (comment 8, last bullet) | DONE | #147 |
| B5 | Positioning vs estimate-and-forward (Gomadam–Jafar 2007, verified via DBLP/arXiv), Farsad–Goldsmith, ViterbiNet; named-study table (now Table 3.2) (comment 9) | DONE | #148 |
| C4 | Reliability-weighted (|h|²) Viterbi/BCJR at relay + destination; coded Tables 34/37/39 rerun (`coded_matched_metrics.py`, 30 paired trials, published seeds, unweighted rows reproduce exactly). **Result: matched block DF beats both learned relays at every SNR 0–20 dB, 30/30 trials, either destination metric; the published high-SNR learned advantage was a metric artifact.** New Sec. 5.5.2 + Table `tbl:coded-matched`; Ch. 5/8/9 and both abstracts updated. Secondary coded studies (Tables 38, 44, throughput/latency/rate adaptation, coded size sweep) NOT rerun | DONE | #150 |
| C4b | Secondary coded studies rerun with matched metrics (`CODED_METRIC=matched`, 30 trials, published seeds; outputs `results/*_matched.json`, envelopes `coded_matched_envelopes.py`). Matched block DF stronger everywhere: relay symbol errors ~1e-5/1e-6 (Table 38); larger K now helps (8 dB: 0.0033/0.00069/6.7e-5); reliable regime = oracle (0 errors 24–32 dB); equal-throughput coding wins from 8 dB (12.3× at 16 dB); rate-adaptation envelope far above denoise 8–32 dB; 150-symbol budget: block DF leads ≥20 dB; 100-symbol structural lock-out survives; coded size-sweep penalties +7.6–9.9 dB. Matched values added beside published ones (Tables 38/40/42/43, prose), Ch. 8/9 updated; verifier check `coded-matched-secondary` (43 cells) | DONE | #154 |
| C5 | Corrected-figure vs superseded-table consistency (Khina presentation note). The ISI→Rayleigh (S2) and control (S4) rows had already been removed from `tbl:tableE6` and Fig. E6 (#122/#123), so no rerun was needed. Remaining instance fixed: Table 39's 20 dB soft block-DF cell now shows the doubled-variance run Fig. 5.x plots (0.001376), nominal 0.002348 in the footnote; verifier maps the cell to `coded_soft_df_calibration.json` | DONE | #151 |
| C6 | Causal/streaming normalization. Not needed: no 5/10-symbol operational-readiness claim remains. #138 separated detector look-ahead from normalization buffering (Ch. 5 Table 41 text), Ch. 8 separates protocol scheduling, readiness and computation, and the streaming rule is stated as not evaluated; the joint-latency hard MLP and MLSE relays forward exact points with no normalization, so their delays are readiness delays | CLOSED (no code) | #155 |
| C7 | E6_FLAT Hop-2 noise fixed (1/γ → 1/(2γ) for real forwarded signals, three sites in `e6_flat_ported.py`) and rerun on the original seeds. DF/MLP BERs roughly halve; DF-vs-MLP similarity holds from 8 dB; below 8 dB the unknown-phase MLP trails differential detection by up to 0.0106 at 0 dB (exceeds the 0.0036 control target in CLAUDE.md; reported, not tuned). Table `tbl:tableE6flat`, Ch. 6 conclusion and Appendix G updated; new verifier check `prose:flat-conclusion` | DONE | #152 |
| C8 | Training protocol now described from the committed code: all relays train with Adam (MLP/Hybrid lr 0.01 batch 32 — the NumPy SGD path is a no-PyTorch fallback the canonical run excludes; VAE/sequence 1e-3 batch 64; cGAN 1e-4, β=(0,0.9)); samples 25k (MLP, Hybrid) / 50k (VAE, cGAN, sequence), 100 epochs (cGAN 200). Exact reproduction is not possible (unseeded torch init, global-RNG channel draws; data predate repo history) — stated once in Ch. 4. Fixed #133's "canonical SGD" and Ch. 4's "trained by the same protocol" | DONE | #153 |
| M  | Memory bank updated for the round (`activeContext.md`, `progress.md`); kept current through the audit and recheck | DONE | #155, #160, #161 |
| P  | Republish mirror: done after Phase B (46ad144), after C4–C8 (f3eabf6, 133 pages), after the audit (4d3e992, 136 pages), after the second recheck (69a3386 = main e26361b, 138 pages), after #162 (162fb85 = main 1eaeb08), after the final review #164 (3d45423 = main eee7895, 119 countable pages), after the README pass #166 (5b1da55 = main c8f7710; only the generated README.md changed) after the Overleaf-guide fix #168 (2a3fd16 = main c4b6a0c; only OVERLEAF.md changed) and after the figure check #169 (4b0e1e1 = main 6809399; two chapters, six figures and thesis.pdf, 119 countable pages) and after the combined PR #173 (748962e = main 82875bc); author Gil Zukerman, `thesis.pdf` byte-identical to main each time | DONE | — |
| R  | Reply to Khina: drafted 2026-10-12, one paragraph per comment with the thesis location of each fix (no PR numbers); C4 reversal stated; ×2 multiplication-count convention; C7/C8 disclosures. Committed at the author's request: `thesis/submission/reply_to_khina_2026-10.md` (not sent); updated after the point-by-point audit (#158) and the second recheck (RC), section/table numbers re-checked against the 138-page build | DRAFTED, COMMITTED (awaiting author) | #156, #157, #159, #160 |
| RC | Second recheck of the rendered PDF against every sub-request: factorization caveat wherever M^L is stated (Fig. 6.7, Sec. 6.5, 2.6, O2, H5, Appendix); Table 5.11 "Look-ahead" caption; Sec. 4.1.5 matched destination metric; Ch. 1 contribution 3; Sec. 7.1.1, 7.6 wording; Hybrid described as coded (`hybrid_branch_selection.py`, verifier `prose:hybrid-branch`); sequence-model holdout; Appendix B Table 9.1; test counts | DONE | #160 |
| RC2 | Found while updating the documents: Ch. 7 (comparator scope) and the Ch. 8 closing paragraph still listed "fully reliability-weighted decoding" as unmeasured, contradicting Sec. 5.5.2; both now say a destination metric that also models relay decision errors is what remains unmeasured | DONE | #162 |
| FR | Final page-by-page review of the rendered PDF: body was 122 countable pages against the 120 limit (#160 added two) - now 119 by cutting redundancy where chapters ended with 2-3 lines; Table 5.13 ran off the page and five tables overflowed - 0 overfull boxes now; Sec. 4.2 still described a sqrt(2) axis rescaling (contradicting Eq. 4.16 and the code); Transformer rationale said d_ff 128 (code 64); cGAN lambda_L1 100 (code 20); Eq. 2.9 notation; DF closed-form percentages; Table 5.12 caption; boundary wording vs Sec. 3.1.1; appendix letters (no Appendix A) dropped; revision language removed; bibliography arXiv IDs and casing | DONE | #164 |
| D  | Tracking documents brought up to date: thesis, rerun and framework changelogs, README findings, REPRODUCE, context, Overleaf-sync guide (publish identity), Hebrew-abstract note, superseded cover letter marked, memory bank (techContext SNR trap, systemPatterns), table provenance regenerated; reply final pass | DONE | #163 |
| RM | Every README checked against the thesis and code: root `README.md` (Ch. 6 table restated with the thesis's own numbers plus the scalar MAP row and the factorized trellis count; Hybrid snippet shows the fixed 5 dB threshold on a per-axis power estimate, switch at 6-8 dB; Transformer d_ff 64; cGAN lambda_L1 = 20; Overleaf mirror row); `e6_unknown_channel_results/README.md` (backs Ch. 6 on main, S2/S4 not reported, October reruns, scripts write into the repo); `e6_multi_training_results/` and `e6_qpsk_rescaled_results/` marked historical/exploratory; `results/_archive_precorrection/` stale appendix reference removed; mirror README template in `scripts/overleaf_project.py` (AF qualifier, matched-metric coded reversal, factorized trellis); `thesis_overleaf*.zip` and `thesis_preview.pdf` rebuilt from the current thesis (both were September builds). Then the mirror's other doc, `thesis/OVERLEAF.md` (shipped in the project and the zips): Hebrew font is David CLM, not Arial; the `hebrewcal.sty` stub main.tex loads since #123 is explained; `thesis/OVERLEAF_SYNC.md`: 50-file project (18 figures, `hebrewcal.sty`) + README + thesis.pdf = 52, route B named, tooling on main, `origin/overleaf-dist` (2026-09-04) marked not current. Memory bank: techContext/projectbrief build recipe now `thesis/main.tex` via latexmk, not the old root `thesis_tau.tex` | DONE | #166, #168, #167 |
| FG | Figures checked against their data by regenerating the 16 reproducible ones from committed files (12 matched). Figs. 5.3-5.5 were drawn from earlier runs than Tables 5.8/5.12/5.13 (5.3 and 5.4 printed 20 and 10 trials; redrawn from the 100-trial runs); Table 5.8's 20 dB soft block-DF cell showed the doubled-variance run while its footnote said Fig. 5.3 plots it (cell back to the nominal 0.002348, doubled-variance value footnoted, reply corrected); Fig. 6.3 still had the removed Rayleigh-hop-2 panel (single AWGN panel, Fig. 6.2's style, title names the per-symbol fading, caption no longer says the MLP 'separates' from 2 dB); Fig. 6.7(a) had no script and read MLP-169 (new `scripts/plot_e6_complexity_cost.py`, MLP-170 reference line; panel (b) timed the 169-parameter I/Q relay, now stated); Fig. 5.5 axis said latency (now buffer-readiness budget); Fig. 5.1 title had an internal '(§7.10)'. 119 countable pages | DONE | #169 |
| AC | Author's corrections applied as given: Sec. 4.1.5 says the BER-optimal hard-relay target is bit-wise MAP (= symbol-wise MAP only for BPSK or a factorized QPSK posterior; a joint four-class arg-max targets symbol error); Sec. 6.4.1 says beating zero-threshold DF does not show a temporal-context benefit (the scalar MAP control removes the floor without a window); reply comment 1 likewise, and comment 8 now says d_ff 128 gives 26,017 for the full model, 25,408 being the two encoder blocks only (609 projection parameters) -- verified on the instantiated model; README page count 119 (135 PDF). 119 countable pages | DONE (merged 82875bc) | #172, combined in #173 |
| FA | Pre-submission audit of #173 (no merge): clean rebuild 0 errors/undefined/multiply defined/missing glyphs/overfull; BibTeX 0 warnings; 135 PDF pages, 119 countable; fonts embedded; no ??/placeholders/LaTeX residue/old appendix letters/revision language; 53 cited entries complete; abstracts and headline numbers consistent; reply cross-checked (all cited numbers and labels right). Fixed: five British spellings; Sec. 4.3.3 now states the unseeded PyTorch initialization the reply cites; Sec. 5.5.2 'all six SNRs'. Open: title-page date (September 2026, four places) | DONE (merged 82875bc) | #173 |
| CA | Committee-level audit of main 82875bc, report only (`docs/committee_audit_2026-10-13.md`). Substantive: A1 the ten-pilot "lead" in both abstracts is not resolved (0.0545 +/- 0.0108 vs 0.0486 +/- 0.0017; panel b finds no difference); A2 Mamba-S6/Mamba-2 are causal scans read at the window centre, so zero look-ahead (confounds their Table 6.7 rows); A3 Sec. 7.1.3 still says training uses Rayleigh; A4 the initialization paragraph says He/LeCun but the code is Xavier; A5 Mamba block descriptions (Conv1D, gate) do not match the code; A6 Table 6.6's learned-relay column has no committed source; A7 H5 names a nonlinear-bias channel with no reported result. Plus B1-B17 and C1-C23. Bibliography (Gomadam-Jafar, Tsai et al.) verified | DONE | #174 (report, superseded by the CF PR) |
| CF | Committee-audit fixes, all findings (report has an Outcome section): ten-pilot lead now "not resolved at ten, leads at five" in both abstracts, Table 6.1, Sec. 6.3, Ch. 8; Mamba zero look-ahead disclosed (Remark 4.1, Sec. 4.2, 6.4.2, App. 9.6); Sec. 7.1.3 training; Xavier init (reverses 7e3f933, which had relabelled it LeCun); Mamba blocks as implemented; Table 6.6 learned-relay column sourced (best-seed best size-sweep config) and verified; S3 nonlinear-bias result reported for H5; H2 title neutral, verdict "Supported" (adjusts 9dde972's scoping); plain-QPSK ISI acknowledged; composite/ISI(QPSK) channel notes; three sequence models; MLP's 0.007; goodput eta (Fig. 5.4 relabelled); Table 5.5 99/0; size-sweep reference; unsourced FER 20.23%/10.58% -> committed 19.98%/10.64%; bib venues. 11 new verifier cells (689). Reply, README, REPRODUCE, mirror template, Hebrew note updated. 120 countable pages | DONE (awaiting approval) | CF PR |
| MR | Mamba look-ahead rerun (audit A2 [data]), `docs/mamba_lookahead_rerun_2026-10-13.md`: both Mamba relays gained an optional `readout` (centre default, bit-identical); full-window and bidirectional readouts at the same parameter count both made Mamba worse (Table 6.7 conclusion stands); MLP/Transformer rerun in the same run; published ISI spreads do not reproduce; seeded training bit-exact on one machine but not across machines (Mamba-S6 0.21 dB, Transformer 0.89 dB, MLP identical); the published Mamba-2 QPSK cell's best seed never reaches 1e-3. Folded into the CF PR: Sec. 6.4.2 look-ahead sentence, "reproducibility inverts" paragraph replaced, Table 6.7 dagger and the 3.14 dB margin note; new verifier check prose:seq-rerun (19 cells, 708 total); five experiments registered. 120 countable pages | DONE | #175 |
| RT | Red-team review of main 4bac934, report only (`docs/red_team_review_2026-10-13.md`): 2 high (R1 blind lead resolved only at 0-12 dB; R2 H1 margin quoted against the weaker conventional AF), 5 medium (R3 matched coded decoder gets \|h1\|^2; R4 training on BPSK/AWGN; R5 H5 baselines fail analytically; R6 Table 9.6 cross-machine caveat; R7 Fig. 6.7(b) and "30-90x" had no committed data), 4 low (R8-R11), 7 attacks cleared | DONE, superseded by RF | #177 |
| RF | Red-team fixes (branch `fix/red-team-findings`): blind intervals rerun (`e6_blind_intervals.py`, every mean reproduces; MLP-CMA lead resolved 0-12 dB, p <= 0.003, not 14-20 dB, p >= 0.40); H1 margin 0.003-0.007 in abstracts (Hebrew number re-drafted; approved by the author 2026-10-14), Ch. 5, 7, 8; R3/R4 disclosed; H5 restated; Fig. 6.7(b) redrawn from committed timing data, 58-82x single-threaded; R6, R8, R10, R11 applied; R9 not applied (MambaCSP venue not yet confirmable). Verifier 732 cells; 120 countable pages | DONE | #178 |
| K2 | Khina round 2 (branch `fix/khina-round2`): (1) canonical runs identified, Sec. 4.3.3 + new Table 4.2 from `canonical_run_provenance.py` (AF/DF of Tables 5.2/5.3 reproduce bit for bit; learned columns not reproducible: unseeded init, weights not saved); (2) comparable-target penalties (`ber_metrics.penalty_common`, `penalty_common_targets.py`): Table 6.7 Mamba-2 ISI(QPSK) +5.83 (>3.28) was +4.69; Table 9.4 coded strict budgets now none; Table 9.5 pooled 148/0.176/>=1.226/>=1.555 was 152/0.180/1.229/3.182; Table 9.6 Transformer spread >=0.892 was 0.906; window gain 'more than 8 dB'; coded sweep +1.24-+1.92 (printed +1.86 was an earlier error), matched >= +9.75 (was +7.6-+9.9); (3) matched-decoding statements scoped to the MLP read-outs, Mamba not rerun; (4) Sec. 4.1.5 local f vs block alpha, differentiability restated; (5) matched decoding leads Sec. 5.5, factorized counts primary in Tables 6.8/6.9 (crossover 6.70, joint 3.35 diagnostic); (6) Remark 4.1, Sec. 4.4 context, simplified Mamba, Fig. 6.5 caption, archive tag, Ch. 5 header. Verifier check `prose:khina-round2`; 858 cells; 351 tests; 120 countable pages. Follow-up at the author's request: script and result-file names removed from the running text and cited as [C1]-[C40] (`\coderef`), the ledger (Table 9.2) rebuilt as the code reference. Then a Khina-checklist pass (`docs/khina_checklist_review_2026-10-17.md`): 22 findings, 19 fixed (paired tests show DF ahead of the MLP/VAE at 7 of 11 SNRs, never behind; Table 6.6 on a common MLSE curve, lead 1.8-2.3 dB; matched/factorized-first ordering; contradictions), 3 judgement items (Fig. 6.7(a) joint bars, matched-column order, Table 9.5 'five of nine'). Verifier 873 cells. An external review of fa2e1b4 was then applied: per-axis normalization map (complex power 2 vs 1, Hybrid per-axis branch), checkpoint rule (min_delta 1e-5, no early stopping), MLP scope in the last coded summaries, reply ranking statement. Tag thesis-2026-10 still to be created on the merge commit before the thesis is sent | DONE (awaiting approval) | #183 |

## Session chain (work these in order)

Each step is one branch off `origin/main` and one PR. A step is finished when its PR is merged
and its row above says DONE with the PR number. If a session stops mid-step, the branch name in the row
tells you where the work is; rerun any experiment script from scratch rather than trusting partial logs.

1. ~~**C5**~~ done (see row). Find which rows of `tbl:tableE6` (Ch. 6, ISI → Rayleigh, S2) come
   from data produced under the old SNR convention and which figures already use the corrected data
   (`provenance_audit.py`, `e6_sim_ported.py` setup S2, `results/e6_matched_*`). Rerun S2 under the
   current convention (γ = 10^(SNR/10), σ² = 1/(2γ) per real dimension), commit the data into the repo,
   update the table and the verifier. Report any number that moves.
2. ~~**C7**~~ done (see row). Was: **C7: E6_FLAT Hop-2 noise.** Fix `e6_flat_ported.py` so real forwarded signals get σ² = 1/(2γ)
   (match the AF branch), rerun, commit `e6_flat_ported_results.npy`, update `tbl` flat rows (Ch. 6,
   `sec:flat-unknown-channels`), and replace the disclosure paragraph in Appendix G with the corrected
   statement.
3. ~~**C8**~~ done (see row). Was: **C8: training-protocol provenance.** Retrain the canonical relays with `train_base_relays` defaults
   (seed from `run_experiments.py`) and compare with `results/modulation/qpsk_rayleigh.json`; if they
   reproduce, rewrite the Ch. 4 sample counts/optimizer to the code; if not, say what is unknown.
   Re-check Phase A #133's "canonical ReLU/SGD" wording.
4. ~~**C4b**~~ done (see row). Was: **C4b: secondary coded studies with matched metrics** (row above). Expect orderings involving block DF
   to change; update Tables 38/44 and the throughput/latency/rate-adaptation text accordingly.
5. ~~**C6**~~ closed (see row). Was: **C6: causal normalization** (comment 6) — only if the 5/10-symbol readiness claims are kept;
   otherwise remove those claims and close C6.
6. ~~**M**~~ done. Was: **M: memory bank** — `memory-bank/activeContext.md` and `progress.md` entry for the whole round.
7. ~~**P**~~ done. Was: **P: republish the mirror** with the author identity (see Environment gotchas); verify fast-forward
   and that `thesis.pdf` is byte-identical to `thesis/main.pdf`.
8. ~~**R**~~ drafted. Was: **R: reply to Khina** — draft only (do not send): one paragraph per comment with the PR that answers
   it, the C4 reversal stated plainly, the ×2 multiplication-count convention, and C7/C8 disclosures.

### Next (author decisions)

0. ~~**Committee audit fixes**~~ done: #175 merged as 4bac934 on the author's approval (2026-10-13), mirror republished as a401ee4 ("Thesis at 4bac934 (clean)", author identity). Hebrew abstract approved, H2 verdict "Supported as matching", title pages October 2026.
1. **Send the reply** in `thesis/submission/reply_to_khina_2026-10.md` (drafted, not sent), with the
   revised PDF (`thesis/main.pdf`, 120 countable pages) or the Overleaf project. The Hebrew abstract is
   approved by the author (2026-10-13; the 0.003–0.007 margin change on 2026-10-14; `thesis/HEBREW_ABSTRACT_SYNC.md`).
2. **Overleaf:** pull once from GitHub (Menu → Sync → GitHub). If Overleaf still points at a pre-rewrite
   commit, unlink and relink first (no Overleaf-side edits exist).
3. **Mirror commit f174a3b** was authored "Claude"; rewriting it to the author's identity needs the
   author's OK (force-push to `relaynet2-thesis`).
4. ~~**Title-page date**~~ done: October 2026 / אוקטובר 2026 on all four title pages (`thesis/main.tex`), the author's choice
   of 2026-10-13.
5. ~~If Khina replies with further comments~~ he did (round 2, 2026-10-17): see row K2.
6. **Round 2 publication:** push and merge authorized on 2026-10-17; see the current state above. Final author reading of the Hebrew wording remains distinct from technical verification. The reply `thesis/submission/reply_to_khina_round2_2026-10.md` is drafted, not sent.

### Task notes

**B2 (done).** Files: `thesis/chapters/ch03_objectives.tex` (O1/O2 at `sec:two-objectives`, H1–H5 at
`sec:research-hypotheses`, Specific Objectives). Add a short "hierarchy" paragraph: main objective →
O1/O2 → H5 + Ch. 6 studies answer O1; joint latency–memory study (`sec:joint-latency-memory`) and
pilot sweep (`sec:partial-posterior`) answer O2; H1–H4 are the canonical control that calibrates the
framework, not the principal contribution. Add a compact table: RQ / hypothesis → section(s) →
experiment script(s) → verdict location (Ch. 9 table). Use `\ref` only, never raw labels. Do not
claim the studies are cumulative relaxations or measure a common boundary.

**B3 (done).** Mostly `ch07_unknown_and_mismatch_channels.tex` (renders as Ch. 6) and the appendix. Read
the generating scripts (`e6_*_ported.py`, `e6_sim_enhanced_multimod.py`, `relaynet/channels/e6_channels.py`)
and transcribe the actual laws — never invent a distribution. The SNR-penalty definitions must
match the code that computes them (find it via `provenance_audit.py` REGISTRY / `verify_thesis_tables.py`).

**B5 (done).** Needs bib entries checked: Farsad–Goldsmith and ViterbiNet (Shlezinger et al.) are already
cited in the appendix and Ch. 6 — reuse `FarsadGoldsmith2018NNDetection` and `Shlezinger2020ViterbiNet`; estimate-and-forward is not cited yet (Khojastepour/
Sabharwal/Aazhang "estimate-and-forward" / Cover–El Gamal compress-forward lineage). Verify any new
reference's identity before adding it. (Abou-Faycal & Médard, "Optimal uncoded regeneration for binary
antipodal signaling", ICC 2004, could NOT be verified and is not cited.) Watch the page budget.

**C4 (done).** Coded pipelines: `relaynet/relays/coded_df.py`, `soft_coded_df.py`, `relaynet/coding/`.
Matched metric: weight each zero-forced squared residual by |h|² (i.e., use the un-equalized metric
|y − h x|²). Rerun the principal coded comparisons into `results/`, add REGISTRY rows, update the
verifier. **This can reverse coded conclusions — report whatever comes out.**

**C5 (done).** `tbl:tableE6` Rayleigh rows vs the corrected figures; rerun with the current SNR convention
(γ = 10^(SNR/10)), commit data, update table + verifier.

## Audit (2026-10-12)

Every sub-request of Khina's nine comments and the presentation notes was checked against the
thesis; eight remaining gaps were fixed in #158 (Sec. 5.3 goal wording; size-sweep configuration in
Appendix E; penalty-table pointers to Appendix G; conventional-AF qualifier in Sec. 5.2, abstracts and
Ch. 8; Table 6.8 delay caption; coded figure/table captions name the metric; Appendix C coverage
sentence; ledger channel column). Thesis 136 pages.

A second recheck of the rendered PDF (RC) found nine more gaps and two code-vs-text mismatches, all fixed: see the RC row above and the facts below. A final page-by-page review (FR row) fixed the page count (122 -> 119 countable), the overflowing tables and further contradictions.

## Facts discovered this round (keep)

- Canonical 7.10 relays are trained **once on BPSK ±1 over AWGN** at 5/10/15 dB (Hybrid 2/4/6 dB)
  and applied per axis to the zero-forced QPSK/Rayleigh signal with **no √2 rescaling**
  (`run_experiments.py::train_base_relays`, `generate_training_targets(use_rayleigh=False)`,
  `simulation/runner.py::_process_relay`).
- Hard-output relays: DF, `MLPQPSKClassifierRelay`, Viterbi relays, `ScalarMAPISIRelay`.
  Soft-output: AF, `MinimalGenAIRelay` (169p), `MLPRelay` (E6 170p), VAE, cGAN, Transformer, Mamba;
  the Hybrid is soft below its SNR threshold and DF above it.
- Against conventional variable-gain AF the 0 dB margin is 0.007, not 0.073 (C3); against fixed-gain AF it is 0.003 (RF, R2).
- The blind-regime intervals stored by `e6_blind_ported.py` pool triplicated classical draws (sqrt(3) too narrow) and ignore MLP training variance; `e6_blind_intervals.py` gives the corrected ones (RF, R1).
- The ISI required-SNR penalties (0.26/1.18/1.88/2.53 dB) had no committed derivation; `e6_isi_penalty.py`
  reproduces them exactly from `results/e6_matched_protocol.json` (B3).
- The canonical Hybrid compares a **fixed 5 dB** threshold with `estimate_snr`, a per-axis received-power
  estimate that assumes unit signal power (the per-axis QPSK power is 1/2, the zero-forced noise heavy-tailed).
  Its switch falls between 6 and 8 dB (`hybrid_branch_selection.py`, exact replay: `Source(seed)` reseeds
  NumPy's global RNG before the bits and Hop-1 draws). The nominal-SNR variant `codex_hybrid.py` is unused.
- Transformer/Mamba wrappers (`checkpoints/checkpoint_18/20/23`) hold out `val_split=0.15` and restore the
  lowest-training-loss epoch (`patience=0`, no early stopping); `seq_models_on_memory.py` keeps the final epoch.
- Script comments in `e6_blind_ported.py` call Hop 2 "always-complex AWGN"; it is compensated Rayleigh
  (`ComplexISIRayleighChannel(taps=[1])`).

## Procedure (every task)

1. `git fetch origin && git checkout -B <new-branch> origin/main` — **confirm `git branch --show-current`
   before any reset**; never `git reset --hard` on `claude/porting-md-file-l6xzsr`.
2. Edit. Results go into the repo (`results/`), never `/tmp/`. New experiment → REGISTRY row in
   `provenance_audit.py` + a check in `verify_thesis_tables.py`.
3. Build: `cd thesis && latexmk -xelatex -g -interaction=nonstopmode main.tex` three times; check exit
   code, `grep -c '^!' main.log` = 0, `grep -c undefined main.log` = 0, `grep -c 'Overfull \\hbox' main.log`
   = 0. Report the countable page count (the folio of the last appendix page; the faculty limit is 120),
   never the `pdfinfo` total, which adds 13 front-matter and 3 Hebrew pages (`.clinerules/90-safety.md`).
4. `python3 verify_thesis_tables.py` (873 cells now, 0 inconsistencies), `python3 provenance_audit.py`,
   `python3 -m pytest -q tests` (347 pass).
5. Commit chapters **and** `thesis/main.pdf` together; push the branch; open a PR into `main`;
   report the page count. Merge only PRs this session opened, with the full 40-char head SHA.
6. When rebasing onto a moved main: resolve `thesis/main.pdf` with `--theirs`, then **rebuild** and amend.

## Environment gotchas

- After a container reset, install TeX: `apt-get install -y latexmk texlive-xetex texlive-fonts-recommended
  texlive-latex-extra texlive-lang-other texlive-plain-generic texlive-science texlive-lang-arabic
  texlive-luatex texlive-publishers poppler-utils`. Python: `python3 -m pip install -r requirements.txt torch`
  (`pip` alone targets the wrong interpreter).
- Mirror publish: `git remote add thesis-repo https://github.com/Gilzuk/relaynet2-thesis`; delete any
  stale local `overleaf-dist`; publish with the author's identity, e.g.
  `GIT_AUTHOR_NAME="Gil Zukerman" GIT_AUTHOR_EMAIL=gil.zukerman@gmail.com GIT_COMMITTER_NAME="Gil Zukerman" GIT_COMMITTER_EMAIL=gil.zukerman@gmail.com python3 scripts/overleaf_sync.py --repo`
  (mirror commits must carry the author's identity; keep the repo's own config as the agent identity). Never `--force` without inspecting what the
  guard reports. Verify the pushed `thesis.pdf` is byte-identical to `thesis/main.pdf`.
  Note: f174a3b (2026-10-12) was pushed authored as Claude; rewriting it needs the author's OK.
- PDF timestamp-only diffs after a verification rebuild: `git checkout -- thesis/main.pdf`.
- PR #126 (advisor letter) is closed unmerged by request; the letter stays on `claude/advisor-update-letter`.
