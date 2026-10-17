**Author-directed interpretation update (2026-10-17):** Treat the October runs as additional training replications and seed-sensitivity evidence. Numerical variation alone does not invalidate original measurements. Section 7.1.4, advisor drafts and reports now say so; all historical tables, plots, archived statuses and prior assessment values are preserved. Canonical bootstrap 20261017 supplies new realizations with unknown historical Torch states; sequence repeats IDs 0/1/2, and coded initialization stays 0. Do not label repeated IDs as independent new seeds or claim all differences are caused by changed initial seeds. The assessment adds seed_design metadata and validates complete sequence seed groups. No new training is launched. PDF remains 120 numbered / 136 total pages; numerical verifier 892/0; all 64 focused tests passed. Existing PR #184 remains draft for author review; this wording update is authorized for publication under the session's standing push instruction. No merge or advisor email requested in this turn.

**PR #184 merged; post-merge audit (2026-10-17):** Author requested merge and another audit. PR #184 merged at a714061cbc6abd111beba037a4a955cfeac27b2e with reviewed tree unchanged. New audit/postmerge-184-20261017 branch fixes a real signed-rank tie defect: use integer error-count differences (10,000 canonical / 100,000 coded bits), requiring and validating denominators. Historical 3K Transformer/DF at 20 dB changes p 0.05078125 -> 0.04296875; additional native S6/DF at 16 dB changes 0.03515625 -> 0.0546875. Exactly two significance classifications change; no learned canonical DF advantage, coded DF lead and raw BER data unchanged. Corrected analysis code/derived results pinned at 80b4545adcc66e4be1f435364f7ec927e87f6414; thesis archive now points there. Active 16-QAM scope prose now includes the equal-throughput control as well as adaptation. All 418 ten-pair results agree with independent sign enumeration; full suite 366 passed plus the later exhaustive test (final focused file 4 passed), numerical verifier 892/0, assessment and provenance pass. Merged-snapshot model audit passes all 48 checkpoints / 70 output hashes. PDF remains 120 numbered / 136 total; revised pages inspected and preview/bundles synchronized. Report: docs/postmerge_184_audit_2026-10-17.md. Publish the follow-up branch as a separate PR, not to the merged reproduction branch. No full-budget experiments launched/cancelled/modified, no advisor email, no exact historical reproduction or equivalence claim. Prior draft/unmerged status entries below are historical.

**Final skeptical/model/reproduction audit (2026-10-17):** The author approved commit, requested final audits, then explicitly added unsupported statements and missing researcher reproduction details. Audited base d29f8ee96664e5df8baa21a7e02e2a91ca9c665d. Corrected early normalization/CSI statements, initialization exceptions, I/Q input order, coded training recipe, nominal-budget/cGAN/decoder-causation/finite-grid claims and reproduction instructions. New scripts/audit_round2_models.py strictly loads all 48 retained checkpoints and verifies 70 inventoried output hashes; S6 nominal counts include unused C projections (2,080 native / 198 3K parameters). Newly explicit gap: the hosted coded job retained trial JSON but no trained classifier weights. Historical canonical RNG/weights/versions remain unavailable. No historical data or figure inputs changed, and no training was launched/cancelled. Researcher guide: docs/round2_reproduction_protocol.md; final findings/acceptance matrix: docs/final_skeptical_model_triage_2026-10-17.md. Assessment/model checks pass, numerical verifier 892/0, focused tests 64 passed (80.23s), PDF 120 numbered/136 total; revised pages inspected, no critical LaTeX warnings, preview/bundles synchronized. Commit/push to existing draft PR #184 is authorized; use [skip ci] to avoid duplicate full-budget jobs. This is technical closure within disclosed scope, not exact reproduction of every learned result, advisor approval, a merge, or email delivery. Do not infer that different seeds caused all discrepancies.


**RNG audit update (2026-10-17):** historical canonical seed 42 controls training data, not Torch initialization/shuffling. Four full-budget native MLP controls at b4059e6 are archived under results/reproduction/20261017-rng-seed42: legacy repeats differ in 102/110 trial BERs; explicit Python/NumPy/Torch 42 plus deterministic algorithms match initial/final weights and all 110 BERs exactly on this host. Seeded vs published maximum mean-BER difference 0.00150 remains unresolved. Future suite runs declare all seed policies, require deterministic Torch algorithms and save global RNG states; preserve original study-local seeds. Ten focused tests pass, 47 evidence hashes verified. Hosted run 39024026594 stays at frozen c0394f0; no active jobs changed/restarted. Canonical_native completed 10:49:37 UTC with worker exit 0 and RETRAINING_DIFFERENCE_REQUIRES_REVIEW (635 fields, classical controls exact); artifact 12099095123 verified. Sequence/coded still running at the check; automated archival pending. PR #184 remains draft. New commit uses [skip ci] to avoid launching duplicate full-budget jobs during the active run; no CI pass is claimed for that commit. See docs/rng_reproducibility_audit_2026-10-17.md.

**Completed rerun review (2026-10-17; supersedes live-status notes below):** All four hosted workers in run 39024026594 completed at frozen source c0394f0, exit 0, scientific status RETRAINING_DIFFERENCE_REQUIRES_REVIEW. Classical arrays match exactly. Archive job succeeded; commit dc7c02c00cbebdfb566362a810ab4b45cb6abbe0 contains every job (index missing_jobs=[]). No experiment was stopped or restarted. Section 7.1.4 now reports the new VAE gap, sequence penalty/spread changes, fresh canonical/coded paired tests and seed-42 control limits; historical tables and plots are unchanged. New scripts/assess_round2_reruns.py checks hashes and produces results/round2_rerun_assessment.json. Verification: 892 values / 0 inconsistencies, 64 focused tests passed; rebuilt PDF 120 numbered / 136 total pages, bundles and preview synchronized, revised pages visually inspected. Existing font-feature/duplicate table-anchor warnings remain, but no undefined refs/citations, missing characters or overfull boxes. Advisor drafts updated and unsent. Publish to existing draft PR #184 with [skip ci] to prevent duplicate experiments; do not claim a new CI pass or merge while discrepancies remain unresolved. Details: docs/round2_rerun_manuscript_update_2026-10-17.md.


Session note (2026-10-17, interrupted-run recovery): session87408 unavailable; local run stopped updating09:07:25 UTC, exact termination cause unknown. Archived16 completed training seeds/checkpoints, original last-live records and comparison53 changed BER cells. Four completed checks still MATCH. BPSK best-of-three ordering unchanged, Transformer best3.723647162 vs3.519872627 dB; classical curves exact on both reached channels. New GitHub Actions workflow isolates four full-budget jobs, retains failures and automatically commits versioned evidence to the reproduction branch. Original simulation scripts/protocols and published numbers unchanged. Workflow/embedded code syntax validated. Next inspect actual hosted status and compare new evidence; no email sent.

Session note (2026-10-17, live reproduction evidence): draft PR184; active run20261017-round2-v2 at source43ce4a6, session87408, continues in background. Four checks match; sequence training running, coded/native/3K queued. Initial setup failure retained and repaired with integration coverage. ISI MLP new best1.965266649 vs published1.973793596; classical baseline exact. Independent original-driver control matches instrumented replay (zero differences), confirming instrumentation did not cause that gap. Historical cause unresolved; partial curves/four checkpoints retained, originals preserved. Evidence index results/reproduction/README.md. Email draft prepared, not sent. Next monitor/compare/publish completed evidence without interrupting the run.

Session note (2026-10-17, reproduction setup): PR #183 merged as42e3e386 after CI; verified content published through GitHub connector, archive218a1c550. Created experiments/round2-reproduction-20261017 from updated main. Implemented isolated background runner/worker, comparison checks and scope documentation; CPU Torch2.8.0 installed separately. Concise Khina resubmission email drafted, not sent. Next freeze/publish code, launch20261017-round2, preserve initial evidence and investigate differences. User asked to keep this work going while preparing the reply.

Session note (2026-10-17, final skeptical re-audit): completed the additional architecture, canonical provenance, Table 6.7 training, padding/device/ledger and interpretation fixes; two censored-median flags fixed (no numeric changes). Report docs/skeptical_reaudit_2026-10-17.md. Verification 873/0; provenance clean; 348 passed/4 optional skips; PDF 120 countable/136 total. Author explicitly authorized push/merge and then requested a fresh reproduction branch with background runs and retained evidence/reference links. Next: publish PR #183, then implement and launch that separate task. Earlier no-push notes are historical. No original training/BER experiment rerun yet.

Session note (2026-10-17, skeptical re-audit): user authorized merge and push, then requested another professor-style audit for inconsistent sentences and reproduction gaps. Corrected two censored-median metadata flags with regression coverage; guide and canonical provenance wording updated. Manuscript corrections/build are in progress before publication.

Session note (2026-10-17, final wording fix): in response to “Fix”, aligned the Hebrew and English abstracts on real-tap factorization, the 1.23-times count, the evaluated hard/soft MLP outputs and QPSK BCJR relay-output scope. Corrected Hebrew i.i.d. wording. Both abstracts rendered cleanly on one page; body remains 119 countable pages / 135 total. Verification 873/0, provenance clean, tests 347 passed/4 optional skips. All fixes are local; no push or merge authorized.

Session note (2026-10-17, final-submission review): local follow-up to PR #183 e58e94d addresses all six round-2 points; final report docs/final_submission_review_2026-10-17.md. Factorized cost plot and matched columns now principal; three unweighted plots in Appendix 9.7; causality, censored correlation, regeneration and budget conclusions corrected. 873 verified values/0 inconsistencies; provenance clean; 347 tests passed/4 optional skips; 119 countable/135 PDF pages. Await author Hebrew review and explicit confirmation before any push.

Session note (2026-10-17, independent follow-up): reread PR #183 at e58e94d and preserved its newer fixes. Local changes on fix/khina-round2-codex-review: immutable code/data snapshot in place of nonexistent tag, block normalization operator and qualified differentiability, optimizer-default disclosure, brief main diagnostics with detailed Appendix 9.7, corrected lower-bound spread interpretation and Figure 9.4. No experiments rerun. 873 checks/0, provenance clean, 347 tests passed/4 optional skips, clean 120-countable/136-total-page PDF. Explicit author instruction: no push without confirmation. Nothing pushed.

Session note (2026-10-12, independent audit): branch `fix/khina-independent-audit-20261012` corrects caption/trial-budget disclosure, reply oracle and code-length wording, README page count and historical-document authority. Evidence table: `docs/independent_khina_audit_2026-10-12.md`. Verifier 678/0, provenance passes, local tests 343 pass/4 skip; saved headline evidence independently recomputed. Original October nine-comment email absent; canonical training reproducibility limitation remains. Reply not sent; mirror publication waits for PR merge.

# Progress — E6 Porting Checklist

Session note (2026-10-17, Khina round 2): his six remaining points addressed on fix/khina-round2 (code+data 02459a0, thesis 5d19033): canonical runs identified (Sec. 4.3.3, Table 4.2 tbl:canonical-runs, canonical_run_provenance.py; AF/DF reproduce bit for bit; learned columns not reproducible); comparable-target penalties (ber_metrics.penalty_common, penalty_common_targets.py): Table 6.7 Mamba-2 ISI(QPSK) +5.83 (>3.28) was +4.69, Table 9.4 coded strict budgets none, Table 9.5 pooled 148/0.176/>=1.226/>=1.555, Table 9.6 Transformer >=0.892, window gain 'more than 8 dB', coded sweep +1.24-+1.92 (printed +1.86 was an earlier error) and >= +9.75 matched; matched-decoding scope = MLP read-outs (Mamba not rerun); Sec. 4.1.5 restated; matched decoding leads Sec. 5.5; factorized counts primary (crossover 6.70); Remark 4.1, Sec. 4.4, simplified Mamba, Fig. 6.5, archive tag thesis-2026-10, Ch. 5 header. Verifier 858/0, 351 tests, 120 countable pages. Reply thesis/submission/reply_to_khina_round2_2026-10.md. Hebrew complexity/coded sentence re-drafted, needs author approval. PR awaiting approval; tag to be created on the merge commit. Follow-up (#183): script names moved out of the prose into ledger references [C1]-[C40] at the author's request; 120 countable pages, 0 overfull. Then the Khina-checklist pass: 22 findings, 19 fixed (paired tests, Table 6.6 common reference, ordering, contradictions), 3 judgement items; verifier 873.

Session note (2026-10-14, merged): #178 merged as fe8198b after a Khina-fix check (one Sec. 7.1.1 referent slip fixed, two trimmed caveats restored, nothing else touched); mirror abf77f8 = main fe8198b; 120 countable pages, verifier 732/0, 347 tests, CI green. Hebrew margin sentence approved by the author on 2026-10-14.

Session note (2026-10-14, red-team fixes): #176 merged (225d70e); R1–R8, R10, R11 applied on fix/red-team-findings (supersedes #177), R9 not applied; blind lead resolved 0–12 dB only; H1 margin 0.003–0.007 (Hebrew number re-drafted, needs author reading); Fig. 6.7(b) from committed data, 58–82×; verifier 732/0, 120 countable pages, 347 tests; merged-branch deletion skipped at the author's direction; awaiting approval.

Session note (2026-10-13, audit fixes): all committee-audit findings fixed on fix/committee-audit-findings (supersedes #174): ten-pilot wording, Mamba look-ahead disclosure, Sec. 7.1.3, Xavier init, Mamba blocks, Table 6.6 column sourced and verified, S3 reported, H2, FER numbers sourced, bib venues; 120 countable pages, verifier 689/0, 347 tests; awaiting approval.

Session note (2026-10-13, committee audit): #173 merged (main 82875bc), mirror 748962e; committee-level audit written to docs/committee_audit_2026-10-13.md (7 substantive: ten-pilot lead unresolved in both abstracts, Mamba zero look-ahead, Sec. 7.1.3 Rayleigh training, Xavier init, Mamba block descriptions, Table 6.6 learned-relay column unsourced, H5 nonlinear bias unreported; plus 17 B and 23 C items). Report only; fixes await the author.

Session note (2026-10-12, pre-submission audit): #173 audited (clean build, 119 countable pages, fonts embedded, bibliography complete, numbers consistent); five British spellings fixed; not merged pending approval; title-page date open.

Session note (2026-10-12, combined): #171 (independent recheck) and #172 (author's patch) reviewed and combined on fix/combined-171-172; #171's claims verified against data; K-sweep reply wording refined; 119 countable pages; awaiting the author's approval to merge.

Session note (2026-10-12, author corrections): author's patch applied (bit-wise MAP in Sec. 4.1.5, DF/scalar-MAP wording in Sec. 6.4.1, reply comments 1 and 8 with Transformer counts 26,017/25,408/609 verified, README 119 pages). 119 countable pages.

Session note (2026-10-12, figures): all 18 figures checked against their data; Figs. 5.3-5.5 redrawn from the 100-trial runs their tables report, Table 5.8 20 dB soft block-DF cell back to the nominal run Fig. 5.3 plots, Fig. 6.3 single AWGN panel, Fig. 6.7(a) given a script (MLP-170), Fig. 5.5/5.1 labels fixed; reply corrected. 119 countable pages, verifier 678/0. #169 (main 6809399); mirror 4b0e1e1.

Session note (2026-10-12, READMEs): #166 brought every README current (root, three e6_* result READMEs, archive README) and the generated mirror README template; zips and thesis_preview.pdf rebuilt; mirror 5b1da55 = main c8f7710 (README only). #168 corrected thesis/OVERLEAF.md (Hebrew font, hebrewcal.sty stub) and OVERLEAF_SYNC.md (file counts, route, tooling location); mirror 2a3fd16 = main c4b6a0c (OVERLEAF.md only, thesis.pdf byte-identical). Memory-bank build recipe updated to thesis/main.tex. Thesis unchanged, 119 countable pages.

Session note (2026-10-12, final review): page-by-page review (#164): body 122 -> 119 countable pages (limit 120), table overflows fixed, Sec. 4.2 sqrt(2) leftover, Transformer d_ff, cGAN lambda_L1, Eq. 2.9, appendix letters, revision language, bibliography; reply updated.

Session note (2026-10-12, final): #162 fixed the last stale claim (reliability-weighted decoding listed as unmeasured in Ch. 7/Ch. 8); mirror 162fb85 = main 1eaeb08; all tracking documents updated (HANDOVER, changelogs, README, REPRODUCE, context, memory bank). Round closed; remaining items are the author's (send reply, Overleaf pull, f174a3b).

Session note (2026-10-12, recheck): audit #158, reply #159, then a second recheck (fix/khina-recheck-gaps): factorization caveat everywhere M^L is stated, Table 5.11 look-ahead caption, Sec. 4.1.5/7.1.1/7.6/Ch. 1 wording, Hybrid described as coded (fixed 5 dB threshold on a per-axis power estimate; replay shows the switch at 6-8 dB), sequence-model 15% holdout + best-training-loss restore, Appendix B table, test counts. Thesis 138 pages, verifier 678 cells, 347 tests.

Session note (2026-10-12, later): Khina round merged through #154 (C4 matched coded metrics reverse the coded high-SNR conclusion; C4b secondary studies; C5, C7, C8; C6 closed). Thesis 133 pages, verifier 657 cells, 347 tests. Remaining: republish mirror, draft Khina reply (HANDOVER.md P, R).

Session note (2026-10-12): Khina round -- A, C1-C3, B0, B1 merged (#131-#143);
thesis 124 pages. Remaining chain (B2-B5, C4-C6, republish, reply) in HANDOVER.md.

Session note (2026-10-03, later): exact I/Q MLSE factorization paragraph and
scope-wording patch applied (see activeContext); thesis 121 pages.

Session note (2026-10-03): pre-submission theory-review fix patch applied to
the thesis (7 wording/citation items: Proakis Ch.13 pinpoint, Leshno-1993 UAT
attribution, AF readiness qualification in tbl:table41, equal-parameter claim
softened, pilot-crossover heading corrected, Viterbi 64-op convention stated,
Mamba expressiveness reworded). verify_thesis_tables.py clean, provenance audit
clean, main.pdf rebuilt at 120 pages. No data or committed numbers changed.

Branch: `copilot/fix-bug-in-data-processing`. Reference spec: `experiments-standalone/PORTING.md`.

Session note (2026-09-04): replaced the Overleaf sync. `git subtree push
--prefix=thesis` published the working trail (draft chapters, excluded
review-response appendices, build artefacts, changelogs, `\REV` records) into
the authoring surface. Now `scripts/overleaf_project.py` defines the project as
main.tex's transitive closure (54 files) for both the zips and the sync, and
`scripts/overleaf_sync.py` publishes it on a generated `overleaf-dist` branch
whose root is the project root, annotations stripped. One-way by construction:
`--push` refuses when Overleaf has commits the branch lacks (verified against a
stand-in bare repo, including `--force`). Bundles are now byte-reproducible.
Branch content compiles standalone and renders text-identical to
`thesis/main.pdf`; 205 tests pass. `make overleaf-{show,sync,mirror,push,pull}`.
The tooling lives on `claude/overleaf-sync`; `overleaf-dist` is mirrored to
origin so the published state is visible without Overleaf credentials.
Second route added: `--repo` / `make overleaf-repo` publishes the same tree to
a dedicated GitHub repo (`thesis-repo` remote, branch `main`) for Overleaf's
GitHub integration, sharing the clobber guard. `Gilzuk/relaynet2-thesis` was
created by the user (this session's GitHub app cannot create repositories,
403) and now holds the project at `main` = `overleaf-dist` = `8ed4228`; the
first publish needed `--force` to clear GitHub's auto-init README. The
generated tree gained a root `README.md` marking it generated and
not-to-be-edited (55 files).

Session note (2026-09-03, latest): regenerated and committed the two Overleaf
bundles, after fixing the generator. `build_bundles.py` hardcoded
`hebrewcal.sty` while `main.tex` loads `hebcal` — every committed bundle was
missing a style file it needs and could not compile; the extras list is now
discovered from the sources and the build fails if a local `.sty` is absent
from the zip. `strip_rev` also ate the space after inline `\REV{...}`, so the
submission copy ran sentences together in 99 places; it now only collapses
whitespace for an annotation that occupied its whole line
(`tests/test_strip_rev_spacing.py`, 6 tests). Both zips were extracted outside
the repo and compiled: 0 errors, 0 undefined references, 132 pages, text
identical to `thesis/main.pdf`.

Session note (2026-09-03, later): unified the E6 first-error bit budget. The
per-SNR cap dict (1G at 16 dB / 10G at 18–20 dB) in `e6_sim_ported.py` is
replaced by the adaptive rule already implemented in `run_ber_first_error`
— stop at 10 × bits-to-first-error — with one hard ceiling
`FIRST_ERROR_MAX_BITS = 10G` per run at every first-error SNR. Committed data
unaffected (16 dB stopped at 334M ≪ 1G; 18/20 dB already 10G); documented in
`provenance_audit.py` REVIEWED_STALE. Metadata keys renamed to
`first_error_max_bits`.

Session note (2026-09-03): the three 20 dB trellis-control SERs in the QPSK
withdrawal prose (0.184 / 0.000 / 0.022) were from an ad-hoc uncommitted run
built with *unnormalized* taps (`ComplexISIRayleighChannel.__init__` normalizes
its taps argument in place, so construction order silently decides what a
trellis sharing `H_ISI` sees). Committed `qpsk_trellis_controls.py` reruns them
under the exact `tbl:tableE6qpsk` configuration: 0.113 / 0.000 / 0.0003, now
consistent with `qpsk_error_decomposition.json` and the table's genie BER.
Registered in `provenance_audit.py`; verifier check `prose:qpsk-controls`
added (509 cells / 0). Also fixed the stale `fig:figE6qpsk` caption (still
carried the withdrawn claim) and the Appendix F "symbol-MAP" wording (bit-wise
MAP is the BER-optimal comparator). PDF rebuilt: 149 pages.

Session note (2026-09-01): the QPSK "genie-CSI Viterbi" benchmark was never
given the CSI. Hop 1 of the QPSK study is `ComplexISIRayleighChannel`,
`y[n] = g[n](h*x)[n] + v[n]` with an independent Rayleigh magnitude per symbol;
the trellis was built from `h` alone, so it was model-mismatched, not genie.
`FadingAwareViterbiQPSKRelay` scales each branch by `g[n]`, and the channels now
record `last_gains`. With the correct benchmark the published "reversal"
disappears: genie CSI leads MLP-QPSK at every SNR, 0.0001 against 0.0508 at
20 dB. The criterion-mismatch conjecture the thesis offered was ruled out first
by `qpsk_error_decomposition.py` (the MLP led on symbol error rate too). The
BPSK study is unaffected -- `ISIChannel` does not fade. Chapter 7's QPSK
subsection rewritten; `tbl:tableE6qpsk` gains a fifth row.

Session note (2026-09-01): confidence intervals are now hierarchical. Pooling
3 seeds x 10 trials as 30 i.i.d. understates the MLP interval by 7-8x, since the
10 trials inside a seed share a trained network. `ber_metrics.hierarchical_ci`
averages within a seed and puts a Student-t interval on the seed means, and
returns the pooled interval alongside. `e6_sim_ported.py` computes both and
persists raw per-column BERs so intervals can be recomputed without re-running.
`--reuse-rare-event` carries the 16-20 dB cells (single measurements, no
interval either way) over from the committed run rather than spending hours of
10-billion-bit searches reproducing numbers that cannot move. Note that skipping
them changes the RNG stream, so the 8 and 12 dB means moved by ~2e-4; the thesis
carries the new run's values.

Session note (2026-09-01): the 0.25 memoryless-relay floor is now derived rather
than asserted, for AF as well as DF (`isi_slicer_floor.py`). Both closed forms
track their measured columns to 3-4 decimals across the sweep, which is an
independent check on two columns of `tbl:tableE6` and on the rare-event
estimator behind their high-SNR entries.

Session note (2026-08-31): replaced the first-error estimator with an
error-counting one. `run_ber_first_error` in `e6_sim_ported.py` no longer stops at
the first error -- it fixes the budget at `10 x N1` bits (capped per SNR) and
reports accumulated errors / total exposure; a cell with no error inside the cap
reports the rule-of-three `3/N` 95% upper bound, not `1/N` and not zero. The
2026-08-28 note below is superseded: its 16 dB figure of $1.61\times10^{-7}$ came
from mean reciprocal waiting time, one event per trial. All four E6 setups were
regenerated on three seeds (`a3a07ab`, full log in
`results/e6_sim_rerun_progress.txt`) and
`tbl:tableE6` repointed to that run. The old estimator was wrong by more than
rounding at high SNR: DF at 16 dB read `0.500` from a single error in two bits
against `0.2296` from 22,959 errors. MLP S1 is now `0.0064` / `9.55e-5` /
`4.79e-8` / `<3.0e-10` at 8/12/16/20 dB. Results are written to
`e6_unknown_channel_results/`, never `/tmp/`, and checkpointed after each setup.

Session note (2026-08-28): enforced the high-SNR first-error runtime policy in
`e6_sim_ported.py` so 18 dB and 20 dB now run until the first observed failure
or a 10G-bit timeout (`FIRST_ERROR_MAX_BITS_BY_SNR`), with the per-SNR caps
saved in output metadata.

Session note (2026-08-28): reran 16 dB with 100 independent first-error trials
and a 1G-bit cap per trial. All trials found an error; mean reciprocal
waiting-time BER was $1.61\times10^{-7}$ with 95% CI $\pm5.60\times10^{-8}$,
and mean stopping time was 20.5M bits. Updated the Chapter 7 table and
rebuilt `thesis/main.pdf` to 130 pages.

Session note (2026-08-28): synchronized the five regenerated E6 figures from
the adaptive-bit result set into `thesis/results/`. The adaptive-bit metadata
was verified in `e6_multi_training_results/e6_sim_ported_results.npy`. The
thesis PDF was subsequently rebuilt with `latexmk -xelatex`; it is now 130
pages and reflects the synchronized figures.

Session note (2026-08-28): N_TRAIN=3 multi-seed robustness study completed.
All 5 MLP scripts modified to pool 3 training seeds × N_TRIALS trials.
Runs completed; results in `e6_multi_training_results/`. ch07 text, table
values, and 5 conclusion paragraphs updated. All 5 thesis figures regenerated
from new data via `plot_e6_figures.py`. New 1-vs-3-seed comparison figure
saved to `e6_multi_training_results/e6_seed_comparison.png` via
`plot_e6_seed_comparison.py`. `thesis/main.pdf` rebuilt to **129 pages**.
The 120-page target is 9 pages over; author must decide what to cut.

Session note (2026-08-27, later): merged the one kissing-figure pair in
compiled Chapter 6 (old Figures 6.7/6.8 -> a single two-panel Figure 6.7) and
moved it above the closing paragraphs, removing the float-only page. Compiled
Chapter 6 is `ch07_unknown_and_mismatch_channels.tex`;
`ch06_experiments_extension_Higher_Order_Modulation.tex` is not included in
`main.tex`. PDF still 128 pages.

Session note (2026-08-27): rebuilt `thesis/main.pdf` (128 pages, no unresolved
refs/citations) so it matches the sources after PR #39's `[H]` → `[htbp]` figure
placement change, which merged without a PDF rebuild. TeX is not preinstalled in
the agent container but is installable via apt — see `activeContext.md` for the
exact package list and the `latexmk -C` caveat.

Session note (2026-08-26, later): thesis layout cleanup removed near-empty
pages by tightening the live abstract, tightening the Chapter 3 closing
sentence, and reducing TOC chapter-entry spacing in `thesis/main.tex`.
Also removed stale cGAN re-run commentary from the thesis text/captions and
rebuilt `thesis/main.pdf` to 125 pages.

Session note (2026-08-26): thesis theory wording was corrected in the compiled
chapter files and `thesis/main.pdf` was rebuilt to 128 pages. A new general
rule now lives in `.clinerules/00-general.md`: check theory claims against
standard textbooks or primary sources and narrow wording to the conditions they
actually support.

## Done ✅ (verified at 5 trials × 50k bits, SNR 0–20dB/2dB)

| Experiment | Ported file | Status | Key numeric checks |
|---|---|---|---|
| E6_SIM (S1–S4: unknown ISI/nonlinear-bias, AWGN/Rayleigh, control) | `e6_sim_ported.py` | ✅ Verified | ISI floor 0.18–0.24 (theory ~0.25) ✓; MLP <5e-5 @16dB S1 ✓; non-monotonic DF confirmed ✓ |
| E6_VITERBI (genie CSI + LS-estimated MLSE) | `e6_viterbi_ported.py` | ✅ Verified | Viterbi-genie ~1–1.5dB ahead of MLP @1e-2 BER ✓; LS-est ≈ genie ✓ |
| E6_FLAT (unknown phase/gain/I-Q-imbalance, memoryless control) | `e6_flat_ported.py` | ✅ Verified, rescaled to 10×100k | See "E6_FLAT bug fixes" below — the "gaps 0.02–0.99" previously blamed on "spec too strict" were two real bugs, now fixed. Post-fix gaps: F1=0.0075, F2=0.0041, F3=0.0050 (target ≤0.0036) — same order as the standalone script's OWN F1 gap (0.0037, itself barely over target), so this residual is the genuine MC/training-seed floor, not a further bug. |

Infrastructure landed as part of the above: `relaynet/relays/mlp.py`, `relaynet/relays/viterbi.py`, `relaynet/channels/e6_channels.py` (8 channel classes), `test_e6_core.py`.

## Done ✅ (cont'd)

- **`e6_sim_enhanced.py`** — multi-architecture relay comparison (AF, DF-Hard, DF-Soft, MLP-170, MLP-512, Viterbi-Genie), BPSK only, executed at full scale (5×50k). Confirmed DF-Hard is non-monotonic (ISI hard-decision error lock-in, rises 0.201→0.235 dB from 10→20dB SNR) while DF-Soft avoids it (tracks AF, 0.206 @20dB). AI relays ordered Viterbi-Genie ≤ MLP-512 ≲ MLP-170 at low/mid SNR, converge ~0.005 by 20dB. Chart + data in `/tmp/e6_sim_enhanced_comparison.png` / `.npy` (ephemeral — not committed to repo). See `activeContext.md` for full numbers.
- **`e6_sim_enhanced_multimod.py`** — extends the classical AF/DF-Hard/DF-Soft comparison to QPSK and 16-QAM (BPSK included for continuity), same scenario (unknown ISI → AWGN), executed at full scale (5×50k). Required framework additions (now landed): `ComplexISIChannel` + `ComplexAWGNChannel` in `relaynet/channels/e6_channels.py`, modulation-aware `DFHardRelay`/`DFSoftRelay` local to the script. **Result: the DF-Hard non-monotonic lock-in vs DF-Soft robustness pattern generalizes to QPSK and 16-QAM** — same qualitative shape in all three modulations; 16-QAM sits at a higher overall BER floor (denser constellation, more fragile under ISI+noise), as expected. Chart + data in `/tmp/e6_sim_enhanced_multimod_comparison.png` / `.npy` (ephemeral).
  - **Scoped out**: AI relays (MLP/Viterbi) for QPSK/16-QAM — `MLPRelay`'s single tanh output regresses one real value per window, correct for BPSK but not a valid target for 2- or 4-bit/symbol modulations without a multi-output redesign. Flagged to user as a separate, larger task if wanted.
- **`e6_viterbi_qpsk.py`** + `ViterbiMLSEQPSKRelay` (`relaynet/relays/viterbi.py`) — generalized the BPSK Viterbi/MLSE trellis to QPSK's 4-symbol Gray-coded alphabet (16 states, L=3), complex branch metrics; verified noiseless ISI round-trip = 0 BER before the full run. Full-scale (5×50k) comparison of AF/DF-Hard/DF-Soft/Viterbi-Genie on unknown-ISI→AWGN: **Viterbi-Genie breaks away from the classical ISI floor (~0.18–0.23) starting ~6dB, <1e-2 BER by 10dB, ~0 by 14dB**, while AF/DF-Hard/DF-Soft stay pinned at the floor at all SNRs — concrete confirmation that sequence detection (not a smarter memoryless decision rule) is what actually fixes ISI. Chart + data in `/tmp/e6_viterbi_qpsk_comparison.png` / `.npy` (ephemeral).
  - **Explicitly out of scope per user instruction**: QAM16 Viterbi (256-state trellis for L=3) — user said "no, viterbi only for qpsk". Do not build this unless asked.
- **`e6_mlp_vs_viterbi_qpsk.py`** — fair MLP-170 (BPSK) vs Viterbi-Genie (BPSK) vs Viterbi-Genie (QPSK) comparison, all three under an identical scenario (unknown ISI → plain AWGN, no fading). **Caught and corrected a confound**: an earlier naive comparison mixed numbers from two scripts with different hop-2 channels (`RayleighChannel` vs `ComplexAWGNChannel`), making it look like QPSK-Viterbi dramatically beat BPSK-MLP. Under matched conditions, Viterbi-Genie BPSK ≈ Viterbi-Genie QPSK (statistically indistinguishable, as theory predicts and as a correctness cross-check on `ViterbiMLSEQPSKRelay`), and MLP-170 trails Viterbi-Genie by ~1.5–2dB in the transition region, consistent with the original E6_VITERBI BPSK finding. See `activeContext.md` for full writeup and the "always check hop-2 channel matches" lesson.
- **`e6_relay_comparison_symmetric.py`** + `ISIRayleighChannel`/`ComplexISIRayleighChannel` (`relaynet/channels/e6_channels.py`) — relay comparison redesigned with **symmetric hops**: identical channel model (unknown ISI + Rayleigh + AWGN) on both hop 1 and hop 2, so relay quality isn't conflated with one hop being artificially easier. Full-scale (5×50k), BPSK: AF/DF-Hard/DF-Soft/MLP-170/Viterbi-Genie; QPSK: AF/DF-Hard/DF-Soft/Viterbi-Genie. **Key findings**: DF-Hard becomes the *worst* relay at high SNR (lock-in from hop 1 compounds with hop 2's own ISI); AF/DF-Soft plateau ~0.34 BER almost immediately; MLP-170/Viterbi-Genie both bottom out ~0.225–0.230 (best, but capped since neither corrects hop 2); BPSK/QPSK numbers statistically indistinguishable per relay (modulation-invariance holds again). See `activeContext.md` for full writeup.
- **`e6_viterbi_qpsk_tap_sweep.py`** — QPSK BER vs ISI tap count (L=3,4,5), same symmetric-hop methodology, geometric-decay taps `0.7^k`. Reduced to 3 trials (from project default 5) since `ViterbiMLSEQPSKRelay` decode cost scales ~4x per tap (benchmarked L=3:1.8s, L=4:6.1s, L=5:23.6s per 50k-symbol block; L=6 not attempted, ~98s/block). **Non-monotonic finding**: AF/DF-Soft degrade monotonically with L as expected, but DF-Hard and Viterbi-Genie both hit their *best* BER at L=4, not L=3 (Viterbi-Genie @20dB: 0.230→0.172→0.199 for L=3/4/5, confirmed real, CIs ±0.001–0.002). Reported to user as an unconfirmed hypothesis (competing hop-1-exploitable-structure vs hop-2-uncorrected-difficulty effects), not an established mechanism — do not treat the L=4 sweet spot as settled without more investigation (more trials, L=6+, or isolating hop1 vs hop2 contributions).
- **`MLPQPSKClassifierRelay`** (`relaynet/relays/mlp.py`) + **`e6_mlp_qpsk_vs_viterbi.py`** — the actual "MLP for QPSK" implementation (previously scoped out as needing a multi-output redesign): 4-class softmax classifier over the Gray-coded QPSK alphabet (window=11, hidden=7, 193 params), class indices matched to `ViterbiMLSEQPSKRelay.ALPHABET`. Full-scale (5×50k) run at L=3 taps, symmetric ISI+Rayleigh+AWGN hops: **MLP-QPSK tracks Viterbi-Genie closely in BER** (0.2363 vs 0.2289 @20dB, ~3% relative gap) while being **~179x faster** (9.5ms vs 1700ms per 50k-symbol block) — the concrete MLP-wins-on-wall-clock-despite-Viterbi-being-optimal result E6_COMPLEXITY was meant to show. L=4/5 not yet rerun with this classifier (deferred per user: "for now only l=3").
- **`e6_viterbi_qpsk_pilot_overhead.py`** — realistic Viterbi-Est with 1%-pilot-overhead LS channel estimation (250 pilots / 25,000-symbol data block) vs Viterbi-Genie, QPSK, L=3 taps, symmetric hops. Exercises `ViterbiMLSEQPSKRelay`'s pre-existing `pilot_symbols=` constructor path for the first time on QPSK. **RESOLVED finding** (was flagged as an open hypothesis, now confirmed via direct diagnostic + N_TRIALS=20 rerun, not left open): Viterbi-Est-1pct's small, consistent edge over Viterbi-Genie (~0.002–0.005 BER) is caused by `Viterbi-Genie` being fading-blind (unit-gain branch metric) against a channel where `ComplexISIRayleighChannel` applies fading multiplicatively after ISI — the LS pilot fit accidentally self-calibrates toward `true_taps * E[|h|]` (E[|h|]=√π/2, confirmed to <0.2% via a 200-repeat isolated diagnostic). Added `Viterbi-Genie-EhScaled` (genie taps × analytic E[|h|]) as a properly-specified oracle: it matches Viterbi-Est-1pct almost exactly at every SNR point (statistically indistinguishable, N_TRIALS=20), confirming genie CSI, correctly specified, is once again the upper bound — as it must be. See `activeContext.md` for the full mechanism writeup and the "always double-check what a genie baseline's assumed model actually covers" lesson.
- **`e6_viterbi_qpsk_partial_csi.py`** — three-tier CSI comparison for QPSK Viterbi (worst: 5 pilots / medium: 20 pilots / ideal: `Viterbi-Genie-EhScaled`), L=3 taps, symmetric hops. Originally run at N_TRIALS=5, **now rescaled to project-standard 10×100k** (see below). **Finding**: worst-case (5 pilots, just above the L=3 identifiability floor) is not just worse on average, it's dramatically *unstable* — CIs 5–40x wider than medium/ideal, visibly non-monotonic in SNR from occasional catastrophic LS fits. Concrete demonstration of the "Viterbi collapses near the pilot-count identifiability floor" behavior from the still-not-started `E6_PARTIAL` spec. Medium (20 pilots, 0.08% overhead) already close to ideal and stable.

## Rescaled to project-standard scale (10×100k), thesis integration explicitly deferred
Reviewed all QPSK/symmetric-hop findings from this thread (user asked "review the findings... propose if to report in thesis"). Classified into: Tier 1 (solid, mechanism-confirmed, worth rescaling) = symmetric-hop comparison / MLP-QPSK-vs-Viterbi / CSI pilot tiers; Tier 2 (methodological footnote, not headline) = the genie-fading-blind resolution; Tier 3 (not ready) = the L=3/4/5 tap-count non-monotonic anomaly (mechanism still unconfirmed — NOT rescaled, still excluded).

User approved rescaling + thesis integration ("Do so"). Before touching any `chapters/*.tex`, investigated the actual thesis structure and found **no existing "Chapter 7 / E6 unknown-channel" content anywhere in the compiled thesis** — `ch05_experiments.tex`'s "E6" is an unrelated CSI-injection/LayerNorm experiment (naming collision with `PORTING.md`'s terminology), `chapter7_experiments.md` is a separate differently-structured doc, and Appendix C doesn't contain the reproducibility claim `PORTING.md` paraphrases. Surfaced this via `AskUserQuestion` before proceeding. **User's answer: rerun only, do NOT update the thesis yet.**

Rescaled `e6_relay_comparison_symmetric.py`, `e6_mlp_qpsk_vs_viterbi.py`, `e6_viterbi_qpsk_partial_csi.py` in place to `N_TRIALS=10, N_BITS=100_000` (the 5×50k dev-scale values are gone from these files now — this is the only version going forward) and ran all three in parallel. All qualitative findings replicated. **Caught and corrected a CPU-contention artifact**: running all three jobs concurrently inflated the MLP-QPSK latency measurement 4x (39.42ms vs true ~9ms), dropping the reported Viterbi/MLP speed ratio from ~180x to 42.5x in that run's raw log — do not trust that number; corrected via isolated re-measurement to **183.1x** and patched into the saved `.npy`.

**Results committed to the repo** at `e6_qpsk_rescaled_results/` (3 PNGs + 3 .npy + README) — NOT inside `results/` (that's the thesis's canonical figures dir; this content is explicitly not yet thesis-integrated). See `activeContext.md` for full detail.

## Done ✅ (cont'd — continuing the original PORTING.md 7-script scope)

- **`e6_composite_ported.py`** — E6_COMPOSITE ported: DBPSK → 3-tap ISI[1,0.6,0.4] → Rapp PA(sat=1.2) → unknown block phase → noise (reuses existing `CompositeChannel` for hop 1), hop 2 = new `AdaptiveRayleighChannel` (`relaynet/channels/e6_channels.py` — Rayleigh + real/complex-adaptive AWGN, needed because AF forwards a complex undecoded signal while DF-diff/Viterbi-diff/MLP forward real decided ones, and the existing `RayleighChannel`/`ComplexISIRayleighChannel` don't adapt to input type). Ported `ViterbiDiffCompositeRelay` (4-state differential MLSE with complex pilot-LS channel estimate, absorbs the per-block phase into the complex channel estimate, blind to PA) verbatim per `PORTING.md`'s "port verbatim" instruction; DF-diff and MLP training/inference implemented inline matching the standalone script. Full scale (5×40k, matching the standalone's own budget). **All targets confirmed**: AF/DF-diff floored ~0.254 @20dB (target ~0.25); MLP-169 (169 params) = 0.0051 @20dB (target ~5e-3); Viterbi-diff visibly ahead of MLP-169 in the 8-12dB transition region (0.0302 vs 0.0403 @12dB); MLP-large (1153 params) ≈ MLP-169 (both 0.0051 @20dB, confirming H3: more params don't help once the task is this hard). Results in `/tmp/e6_composite_ported_results.npy` (ephemeral, not yet copied to repo).

- **`e6_blind_ported.py`** — E6_BLIND ported: composite channel with a FRESH RANDOM 3-tap ISI drawn every call (new `RandomISICompositeChannel` in `relaynet/channels/e6_channels.py`, generalizes `CompositeChannel` to redraw `[1.0, U(0.3,0.7), U(0.2,0.5)]`-normalized taps per call instead of a fixed response), NO pilots anywhere. Hop 2 = `ComplexISIRayleighChannel(taps=[1.0])` reused as a trivial-ISI/always-complex-AWGN Rayleigh channel (this standalone script's `hop2()` always adds complex noise regardless of forwarded-signal type, unlike E6_COMPOSITE's adaptive hop2 — confirmed by reading both standalone scripts side by side before assuming they share a channel). Ported `cma_dfe` (blind constant-modulus equalizer, 7 taps) and `blind_viterbi` (decision-directed bootstrap MLSE, 3 rounds) verbatim. Full scale (5×40k). **Key finding reproduced**: Viterbi-blind is the least stable relay — 95% CI at 10dB is ±0.088 vs MLP's ±0.010 (~9x wider), consistent with the standalone's reported instability (exact CI magnitude differs from the standalone's 0.164, expected given this experiment's inherently high run-to-run variance by design) — **this instability is itself the finding, not something to "fix."** CMA-blind and MLP-169 both hit 0.0024/0.0026 @20dB matching PORTING.md's targets almost exactly. Noted: CMA's adaptive taps occasionally overflow at very low SNR (0-2dB) — transient (verified no NaN/Inf reaches the final BER stats), consistent with CMA's known real-world divergence behavior under severe distortion, not a porting bug.

- **`e6_partial_ported.py`** — E6_PARTIAL ported: reuses `RandomISICompositeChannel`/`ComplexISIRayleighChannel` from E6_BLIND. Panel (a) pilot-count sweep {800,200,50,20,10,5} @10dB, full scale (6×40k, standalone's own budget): Viterbi wins ≥10 pilots (BER 0.027–0.032, vs MLP-169's flat 0.0447), COLLAPSES at 5 pilots to **0.1192** — matching `PORTING.md`'s stated "COLLAPSES at 5 (0.119)" almost exactly. Panel (b) block-length sweep {40,80,160,320,1000} with a fixed 10-pilot preamble, same operating point: overhead spans 25% (L=40) → 1% (L=1000) exactly as specified, MLP zero overhead throughout. Panel (b)'s original standalone script was never in the repo — only its cached `.npy` output (`experiments-standalone/e6_blocklen.npy`, `Nmin=10`, `op=10.0dB`) survived; reconstructed from `PORTING.md`'s description + that `.npy`'s exact structure, then verified at full scale. Results in `/tmp/e6_partial_ported_results.npy`.

- **`e6_complexity_ported.py`** — E6_COMPLEXITY ported. Panel (a) analytical flop-count grid (`viterbi_ops(M,L)` vs `mlp_ops()` read directly off a real `MLPRelay` instance's weight shapes, not hand-computed): confirms honest caveat that at BPSK/L=3 Viterbi is cheaper per-flop (64 vs 330) while MLP wins purely on M^L-vs-constant scaling. Panel (b) measured wall-clock: unlike the standalone script (which hand-rolled its own Viterbi/MLP for timing), this port uses relaynet's **actual** `ViterbiMLSERelay.process()` and `MLPRelay.fwd()` — directly satisfying PORTING.md's reconciliation note ("re-measure with relaynet's actual Viterbi implementation"). Measured speedup 30.8x–85.1x across signal lengths 1k–50k, within the standalone's reported 30–90x range. Reported honestly as numpy-MLP vs Python-Viterbi in this codebase, not a universal claim. Results in `/tmp/e6_complexity_ported_results.npy`.

This completes porting all 4 remaining PORTING.md experiments (E6_COMPOSITE, E6_BLIND, E6_PARTIAL, E6_COMPLEXITY), alongside the earlier E6_SIM/E6_VITERBI/E6_FLAT — all 7 of PORTING.md's experiments now have a `relaynet`-based port.

## E6_SIM/E6_VITERBI/E6_FLAT rescaled to 10×100k + direct standalone comparison

Closed the three gaps flagged earlier (dev-scale only, no literal standalone-script comparison, no thesis figures — figures still pending). Rescaled `N_TRIALS,N_BITS` from 5×50k to 10×100k in all three, then ran BOTH the rescaled ports AND the actual `experiments-standalone/e6_sim.py`/`e6_viterbi.py`/`e6_flat.py` (at their own native 5×50k budget, unmodified) side by side for a literal, not just PORTING.md-target, comparison.

- **E6_SIM, E6_VITERBI**: matched the standalone closely at every SNR point (e.g. S1 AF @0dB: standalone 0.3411 vs ported 0.3408; Viterbi-genie S1 @6dB: standalone 0.0828 vs ported 0.0826). No bugs found.

- **E6_FLAT bug fixes**: the rescale surfaced two real bugs that the earlier 5×50k "verified (qualitatively)" pass had mischaracterized as "spec may be too strict for finite trials":
  1. **Sign inversion in F1's DBPSK path**: `diff_detect()` returns the recovered symbol (≈ x_bpsk), but AF/DF thresholded it with `>= 0` instead of `< 0` (the convention used everywhere else in the file). This flipped every decoded bit — BER climbed toward 1.0 as SNR increased instead of falling toward 0, the textbook signature of a sign bug. Fixed by thresholding with `< 0`.
  2. **Unpaired hop-1 channel realization across relays**: `FlatPhaseChannel`/`FlatGainChannel`/`BranchAsymmetryChannel` hold persistent internal RNG state that advances every `__call__`. The experiment runner called the channel separately for AF, DF, and MLP within the same trial, so the three relays were silently compared against three *different* random unknown-channel draws — breaking the whole point of the "control" experiment (showing DF ties MLP when there's no memory to exploit) and inflating the MLP-vs-DF gap well past the ≤0.0036 target. Fixed by drawing bits + hop-1 output once per trial and sharing across all three relay branches (hop2 paired too, via a separately-seeded RNG shared across relays).
  3. Also found and fixed a **training-diversity gap**: the standalone trains its MLP on 4 sub-batches per SNR (4 fresh random θ/gain/asymmetry draws), but the port only drew one per SNR — 4x less diversity of the unknown parameter. Matched the standalone's structure.

  Post-fix, full 10×100k results: F1 max MLP-DF gap 0.0075, F2 0.0041, F3 0.0050 (target ≤0.0036) — same order of magnitude as the standalone script's *own* F1 gap (0.0037, itself barely under/over the target), so this residual is the genuine Monte-Carlo/training-seed floor at this scale, not a further bug. Qualitative control conclusion (classical DOES tie MLP when there's no ISI/memory) now holds cleanly and quantitatively, not just "qualitatively." **Correction to the record**: the earlier "spec may be overly strict" characterization was wrong — always investigate a BER that's worse than 50% (approaching 1.0) as a likely sign-inversion bug, not sampling noise; genuine noise cannot push BER above ~0.5 systematically.

Results: `/tmp/e6_sim_full.log`, `/tmp/e6_viterbi_full.log`, `/tmp/e6_flat_full_v2.log` (ported); `/tmp/e6_sim_standalone.log`, `/tmp/e6_viterbi_standalone.log`, `/tmp/e6_flat_standalone.log` (standalone, for direct comparison) — all ephemeral, not yet committed to repo.

## Thesis-integration blocker — RESOLVED (added as supplementary appendix)

**The thesis has no "Chapter 7" and no existing content matching PORTING.md's E6_SIM/VITERBI/FLAT/COMPOSITE/BLIND/PARTIAL/COMPLEXITY experiments at all.** `ch07_equation_ref.tex` is an equation-reference appendix, not an experiments chapter — the thesis's actual experiments chapter is `ch05_experiments.tex`, which has its OWN 8 experiments (E1–E8) plus its own, unrelated "E6" (`E6: Input Normalisation and CSI Injection`, SISO 16-QAM/16-PSK). The "E6" naming collision is coincidental — `experiments-standalone/`'s scripts use "E6" as an internal addendum-numbering scheme, not a thesis chapter/section reference. `.clinerules/30-experiments.md` also explicitly fixes Ch5 at "exactly 8 experiments" — confirmed current (not stale) by checking `clean-thesis` directly, so adding a 9th experiment section would violate a real, current rule.

Asked the user via `AskUserQuestion` (a prior attempt failed with a tool AbortError; retried successfully this time): new E9 in Ch5 vs. a new standalone chapter vs. appendix-only. **User chose appendix-only, no Ch5 change.**

**Implemented on `clean-thesis` (commit `64c4dc9`)**: added `\section{Supplementary: Unknown-Channel Relay Experiments}\label{sec:app-supplementary-unknown-channel}` to `chapters/ch09_appendices.tex` (appears as Appendix A.14), covering all 7 experiments with compact tables (4 representative SNR points, matching the document's existing compact-table convention) + one figure each. Added a single-sentence pointer from Ch6's Future Work "Imperfect CSI" item (`chapters/ch06_discussion.tex`) to the new appendix section — the only main-chapter change, and it was explicitly the option the user picked. No Ch5 changes, no renumbering, no existing conclusions altered, per `.clinerules/30-experiments.md` and `90-safety.md`.

Figures/tables use distinct `unkchan_*`/`supp-*` naming (not `e6_*`/PORTING.md's own numbering) specifically to avoid the naming collision described above — both the filenames AND the in-image plot titles were checked and fixed (an initial pass left "E6_SIM"/"E6_FLAT" etc. baked into the plot titles themselves, defeating the point; regenerated with clean titles like "Unknown ISI -> AWGN").

Compiled the full `xelatex → bibtex → xelatex → xelatex` sequence in a separate git worktree (`/tmp/clean-thesis-work`, cleaned up after); `check_log.py` confirms `Undefined References: None`. Visually verified rendered pages via PyMuPDF (`pdftotext`/`pdftoppm` unavailable in this environment) — cross-references (Section 6.6, Chapter 5, Table/Figure numbers) all resolve correctly, no residual "E6" naming in the rendered figures. Pushed to `clean-thesis` per `90-safety.md`'s "always push to clean-thesis only" rule.

`e6_unknown_channel_results/` on this branch (`claude/porting-md-file-l6xzsr`) was renamed to match (`e6_*` → `unkchan_*`), with composite/blind/partial/complexity figures added there too for parity with what's now in the thesis appendix.

## Not started ⏳

All 7 PORTING.md experiments are ported, rescaled/bug-fixed, and now integrated into the thesis as a supplementary appendix. Remaining PORTING.md "After porting" items not done (and likely superseded by the appendix-only decision above, not literal targets): replacing `results/e6_*.png` (none exist), updating "Chapter 7" tables (no such chapter), updating Appendix C's reproducibility statement to mention this work specifically (optional, appendix already documents itself). Nothing currently blocking.

## Known issues fixed this session
1. `ViterbiMLSERelay`: `self.L` used before assignment ahead of `_ls_estimate()` call — fixed.
2. `diff_detect()` in `e6_flat_ported.py`: returned array one element short — fixed by prepending boundary value `1.0`.
3. QPSK/QAM16 support: `calculate_ber` turned out to already be modulation-agnostic (no fix needed there); the actual gap — no complex-aware ISI/AWGN channel, and `DecodeAndForwardRelay` hard-coded to BPSK — fixed via `ComplexISIChannel`/`ComplexAWGNChannel` + local modulation-aware DF relay classes. See `techContext.md` gotcha #3 and `e6_sim_enhanced_multimod.py`.

## Final deliverables (blocked on remaining ports above)
1. Re-run all 7 experiments at project-standard scale (10 trials × 100k bits).
2. Replace `results/e6_*.png` with relaynet-generated figures.
3. Update Chapter 7 tables in the thesis with new BER numbers.
4. Update Appendix C reproducibility statement to state Chapter 7 results are relaynet-generated.
5. Open a PR only if/when the user explicitly asks for one (per repo working agreement).

## Thesis state — 2026-08-17 (canonical restructure)
The thesis side has moved on past the E6 checklist above; current state:
- **Canonical setup is now SISO / i.i.d. Rayleigh fast fading / complex baseband / BPSK **and** QPSK / uncoded BER.** Ch5 §5.3 carries both constellations on the canonical channel (QPSK table moved in from Ch6, now Table 5.6); Ch5 §5.2 is the AWGN *baseline* (calibration + a lean AF/DF/MLP/Transformer/Mamba-2 comparison, 0–8 dB, no ranking drawn from it); Ch6 is now 16-QAM only.
- **Lean relay set for re-runs** is AF / DF / MLP / Transformer / Mamba-2 (cGAN, VAE, Hybrid, Mamba-S6 dropped from re-runs via `run_experiments.py --skip-relays`). Existing committed results for the dropped relays are untouched, not deleted.
- **All channel SNR is on the Eb/N0 axis** after the 3 dB `sigma^2 = N0/2` correction; `tests/test_snr_convention.py` pins every BPSK channel to its closed form and fails if the old convention returns.
- **Verification status:** cold `latexmk -xelatex` exit 0 / 149 pp / 0 undefined refs; `verify_thesis_tables.py` 352 cells / 0 inconsistencies; `pytest tests/` 119 passed.

### Outstanding (do these next)
1. **Ch6 AWGN tables `tbl:table14` / `table15` / `table24` still use the pre-correction AWGN convention** — must be re-run before submission. The verifier explicitly does not cover them.
2. **Ch7 flat-channel control passes by only 0.0097 against a 0.010 tolerance** — margin too thin; needs a larger trial budget.
3. **Overleaf sync — OUT OF SCOPE, the user does it manually.** Per instruction 2026-08-17. Do not push to the `overleaf` remote, do not rebuild subtree splits, do not treat the divergence between the Overleaf project and this branch as a defect to fix. GitHub (`claude/porting-md-file-l6xzsr`) is the delivery target.

## Reference documents already in repo (don't duplicate, update instead)
- `E6_PORTING_STATUS.md` — running progress tracker (slightly stale vs this file as of last edit; treat this `memory-bank/progress.md` as the live source of truth going forward and update `E6_PORTING_STATUS.md` in sync if it's kept)
- `E6_VERIFICATION_REPORT.md` — full numeric verification writeup for the 3 completed experiments
- `E6_PORTING_COMPLETE.md` — final report snapshot for the 3/7 completed state

## New: Coded block-DF experiment (2026-08-19) — complete
Added as a supplementary study, not a re-run of existing results. Rate-1/2 convolutional code (K∈{3,5,7}) with soft-decision Viterbi decoding, genuine block-DF relay, on both canonical constellations (QPSK, 16-QAM); two coded-aware learned relays (MLP, Mamba-S6) trained on the same task for comparison. Full writeup: `activeContext.md` "Latest (2026-08-19)" entry. Code: `relaynet/coding/`, `relaynet/relays/coded_df*.py`. Results: `results/coded_df_experiment.json`. Thesis: new §5.x in `ch05_experiments.tex` + pointers from Ch1/Ch8/both abstracts. Tests: `tests/test_coding.py` (19, all passing). Verifier: 421/0 (was 349/0).

Note found in passing: this file's "Thesis state — 2026-08-17 (canonical restructure)" section above is now stale (describes a BPSK+QPSK dual-canonical setup that was later corrected to QPSK-only canonical, BPSK confined to AWGN calibration) — not reconciled in this pass since it wasn't the ask; flagging so a future session doesn't trust it at face value.

## 2026-08-24: E6_SIM's S2/S4 (Rayleigh variants) removed from the thesis

The "Done" row above for `e6_sim_ported.py` describes all four S1-S4 setups
(S2 = unknown ISI -> Rayleigh, S4 = canonical-Rayleigh control) as verified —
that remains true of the underlying simulation/data, which is untouched. What
changed is which of them the thesis *reports*: per author instruction, every
BPSK-with-Rayleigh configuration was removed from Ch.6 (`tbl:tableE6`'s S2
and S4 rows, and `tbl:tableE6qpsk`'s QPSK-Rayleigh row), leaving only the S1
(ISI -> AWGN) variant in each table. The `.npy` data files still contain
S2/S4/Rayleigh — nothing was deleted from `e6_unknown_channel_results/` —
only the thesis's *use* of them was cut. Full detail in `activeContext.md`,
"Latest (2026-08-24)".

## 2026-08-25: coded family re-measured at 100 trials; Ch2 coding background added

The coded study above was re-run end to end at 100 trials (author's bar:
update only where the delta exceeds 5%). Nine scripts regenerated; Tables 5.4,
5.7, 5.9, 40, 42, 43 and the K-sweep prose updated to match. **Verifier now
reports 361 cells / 2 inconsistencies** — both are 5.5e-07 rounding artifacts
at `tbl:table44` 20 dB where the published cells are correct as printed (see
`activeContext.md`); they are not defects and should not be "corrected".
pytest 159 passed. The 421/0 figure quoted in the 2026-08-19 entry above
predates several table retirements and is no longer the current count.

Three more scripts needed the unseeded-global-RNG fix (`coded_k_sweep_qpsk`,
`coded_k_sweep_qam16`, `coded_mamba_relay`), bringing that total to nine.

Ch2 gained §2.6 (convolutional codes, Viterbi, BCJR, puncturing/AMC), which
Ch5's coded study had been relying on without ever introducing. Thesis is now
**124 pages**, up from 120; the author accepted this on 2026-08-25 and will
review the document later.

Two things a future session should not have to rediscover:
- `verify_thesis_tables.py` checks table cells only. Figures quoted in
  **prose** are unguarded, and a stale 10-trial `1.35×` survived four commits
  of table updates because of it. After any re-run, grep the prose for the old
  values as well as running the verifier.
- Table 41's compute timings are now reported for **two machines** on purpose,
  as the evidence for their machine-dependence. Add a machine rather than
  overwriting if they are ever measured again.

## 2026-08-25 (later): evidence audit, objectives restructure, page-budget findings

Three PRs merged (#25, #26, #27); `main` at `8a4306a`. Beyond the 100-trial
re-run recorded above:

- **Chapter 3 restructured.** The main objective scoped itself to "a single
  canonical setup" while 30% of the body — the 14-page coded study and the
  20-page unknown-channel chapter, the latter being the principal
  contribution — sat under no objective at all. Reframed to the question the
  thesis answers (under which conditions a learned relay surpasses classical
  processing), canonical setup named as control rather than scope, two
  objectives added. Objective 4 (SSMs vs attention) removed as an NN-vs-NN
  question that serves no part of the main objective.
- **Gap 3 removed, folded into Gap 1; Gap 2 narrowed.** Both had claims that
  a literature check falsifies. Both now concede prior work *with* citations.
- **ViterbiNet cited.** Chapter 6 previously cited one work, Forney 1972, and
  did not engage the data-driven-detection literature it sits inside.
  Shlezinger et al. already establish H5's capability statement; the chapter
  now concedes that and claims the boundary instead, which is what it
  actually measured.

**Page budget: 125 against a 120 target.** See `activeContext.md` for the
measured list of what does and does not recover pages — the short version is
that only figure merging works, and it is nearly exhausted. Do not re-try
prose trimming, table merging, the equation list, or font/margin changes.

**Citations added this session were verified against search results, not
publishers** — `arxiv.org` and `link.springer.com` are blocked by the egress
proxy. Worth one verification pass before submission.

- 2026-08-27: Compressed Chapter 2 background exposition, then reverted the
  compression at the user's request (all seven equations restored, back to
  128 pages). Kept the claim fixes: a thesis-wide audit found TWO live
  instances of the retracted "VAE is a consistent underperformer" reading --
  Sec 2.3.1 and, more seriously, Chapter 5's Table 8 conclusion, which
  contradicted both its own table (VAE-3K sits inside the feedforward group
  at every SNR) and Chapter 8's explicit retraction. Both now state what the
  data shows. PDF rebuilt in the same commit.

---

## Experiment provenance ledger

Generated by `provenance_audit.py` -- do not hand-edit this table; re-run the
script. It links every experiment to its script, its committed data, the commit
that produced that data, and the published table or figure resting on it.

**Read this before touching any published number.** The audit fails when an
output is uncommitted, or when its data is older than the script that produces
it. That second condition was silently true for all five E6 datasets between
27 August (when N_TRAIN=3 was added) and 31 August, during which the thesis
reported three-seed numbers backed by single-seed data.

| Experiment | Script | Data | Produced by | Backs | Status |
|---|---|---|---|---|---|
| Coded minimum size | `coded_min_size.py` | `coded_min_size.json` | `ac26dab` 2026-08-29 | `prose: coded row` | ok |
| E6 QPSK unknown channel | `e6_qpsk_unknown_channel.py` | `e6_qpsk_unknown_channel_results.npy` | `aaddb79` 2026-09-01 | `tbl:tableE6qpsk` | ok |
| E6 blind / posterior-free | `e6_blind_ported.py` | `e6_blind_ported_results.npy` | `455c119` 2026-08-31 | `fig:figE6blind`, `prose:E6blind` | ok |
| E6 composite cascade | `e6_composite_ported.py` | `e6_composite_ported_results.npy` | `455c119` 2026-08-31 | `fig:figE6composite`, `prose:E6composite` | ok |
| E6 flat control | `e6_flat_ported.py` | `e6_flat_ported_results.npy` | `96e8884` 2026-08-31 | `tbl:tableE6flat` | ok |
| E6 pilot-budget sweep | `e6_partial_ported.py` | `e6_partial_ported_results.npy` | `2512cb2` 2026-08-31 | `fig:e6-partial`, `prose:E6partial` | ok |
| E6 unknown ISI (S1-S4) | `e6_sim_ported.py` | `e6_sim_ported_results.npy` | `aaddb79` 2026-09-01 | `tbl:tableE6` | ok |
| ISI slicer floor, closed form | `isi_slicer_floor.py` | `isi_slicer_floor.json` | `0441455` 2026-08-31 | `eq:slicer-floor`, `prose: closed-form slicer BER table` | ok |
| Joint latency/memory | `joint_latency_memory.py` | `joint_latency_memory.json` | `ef0f4b7` 2026-08-31 | `tbl:joint-latency` | ok |
| MAC accounting | `unified_latency_axis.py` | `unified_latency_axis.json` | `ab2cb8b` 2026-08-31 | `eq:mac-crossover` | ok |
| MMSE complexity-matched baseline | `mmse_equalizer.py` | `mmse_equalizer.json` | `b1ea325` 2026-08-31 | `tbl:mmse-baseline`, `prose: MMSE monotonicity by tap count` | **stale, reviewed: additive change (859027f): main() now also persists per-target penalties and attained MMSE to mmse_equalizer_detail.json, which was committed from the same run. The headline JSON is not stale -- it reproduced byte-identically on that re-run, so git recorded no change to it and its last commit predates the script edit.** |
| MMSE complexity-matched baseline | `mmse_equalizer.py` | `mmse_equalizer_detail.json` | `859027f` 2026-09-01 | `tbl:mmse-baseline`, `prose: MMSE monotonicity by tap count` | ok |
| Memory sweep, precision re-run | `joint_memory_precision.py` | `joint_memory_precision.json` | `ef0f4b7` 2026-08-31 | `tbl:joint-memory` | ok |
| Minimum relay size, 9 channels | `mlp_min_size_all_channels.py` | `mlp_min_size_all_channels.json` | `1336b84` 2026-08-29 | `tbl:table-minsize`, `fig:minsize-crossover`, `fig:minsize-budget` | **stale, reviewed: comment correction plus a display-name change to the isi_rayleigh comparator ('MLSE' -> 'MLSE (taps only)'). Same relay object, same numbers; only the JSON's `baseline` label would differ on a re-run.** |
| Minimum size, window x depth | `mlp_min_size_bisect.py` | `mlp_min_size_bisect.json` | `a18b10d` 2026-08-29 | `prose: depth 1-3, window 1-7` | ok |
| QPSK error decomposition | `qpsk_error_decomposition.py` | `qpsk_error_decomposition.json` | `bcb39ef` 2026-09-01 | `prose: QPSK SER/BER and bits-per-symbol-error` | ok |
| Seed spread, equal budget | `seed_spread_architectures.py` | `seed_spread_architectures.json` | `0ca3432` 2026-08-30 | `tbl:seed-spread`, `tbl:seed-spread-3k` | ok |
| Sequence models on memory | `seq_models_on_memory.py` | `seq_models_on_memory.json` | `8880cc0` 2026-08-30 | `tbl:seq-on-memory` | **stale, reviewed: 6048c95 touched only main()'s console reporting -- a NaN guard around min() over architectures that reached no target. Every value written to the JSON is computed before that code runs.** |
| Transformer instability | `transformer_instability.py` | `transformer_instability.json` | `ce59ed1` 2026-08-30 | `fig:transformer-seed-curves`, `fig:transformer-loss-penalty` | ok |

**Current audit status** (regenerate; do not hand-edit): clean -- all declared outputs are committed and no data predates its script.

Regenerate with `python provenance_audit.py --markdown`.

## 2026-09-15: final consistency rerun

On branch `codex_final_consistency`, applied the final theory/reference/code
review.  The current real/complex AWGN implementations and thesis equations
now agree on (N_0/2) per real dimension.  `ComplexAWGNChannel` was fixed to
use total I+Q noise power (E_s/N_0); `tests/test_channels.py` adds real and
complex power checks.  `e6_viterbi_codex_consistency.py` reran the AWGN
Viterbi genie and 200-pilot baselines at 10 trials x 100,000 bits, writing
the prefixed `.npy` and JSON manifest while retaining the historical array.

The E6 table/verifier now read the prefixed rerun (genie 0.00137 and LS 0.00148
at 8 dB; zero-error cells are shown with rule-of-three display bounds).  The
original E6 figure was regenerated to 300-dpi PNG plus vector PDF with open
censored markers/arrows and an “insufficient data” legend entry.  Mamba-2,
QPSK, coded-relay, abstract, and summary claims were scoped to measured
conditions; Divsalar--Simon now has DOI `10.1109/26.48887`.

Validation: 263 tests pass; verifier 527/0; provenance registry includes the
new rerun and is clean.  `latexmk -xelatex` remains unavailable on this host,
so `thesis/main.pdf` was not regenerated; compile on Overleaf/CI before merge.

## 2026-09-15: merge conflict resolution

Merged `origin/main` matched-protocol Viterbi updates into
`codex_final_consistency`.  The manuscript now uses the newer matched-protocol
values (0.001354/0.001460 at 8 dB and 3e-8 high-SNR bounds), while retaining
the prefixed rerun as an auditable cross-check.  Resolved the verifier's
scientific-notation tolerance and strict upper-bound logic without weakening
coverage.  Full tests and the 527-cell verifier are green; local TeX tooling
remains unavailable.

## 2026-09-17: professor handover PR #123

Completed the final scoped theory, citation, internal-consistency, and writing
review on `codex_professor_handover`, then opened
[PR #123](https://github.com/Gilzuk/relaynet2/pull/123). The thesis and clean
Overleaf bundle compile to 120 pages; the report and reusable fix-guidelines
prompt are in `docs/`. Local checks: 299 tests passed, table verifier reported
526 cells and 0 inconsistencies, and provenance audit was clean.

PR CI first found that the composite-AF provenance test depended on mixed
working-tree line endings and bit-exact SciPy confidence intervals. Source
fingerprints now use canonical LF bytes; the interval regression allows a
tight cross-platform floating-point tolerance. Commit `6e1765d` passed GitHub
CI run 490. Only the composite-AF comparison was newly run; historical result
arrays were preserved. The generated table provenance ledger identifies
`6e1765d` as the source/data commit.

## 2026-10-13: Mamba look-ahead rerun (committee-audit A2), not in the thesis

Branch `exp/mamba-full-window-readout`. Mamba-S6 and Mamba-2 were rerun on the three Table 6.7 channels
with the centre readout (control), the full-window ("last") readout and a bidirectional readout. MLP
and Transformer were rerun in the same environment.

- **Look-ahead.** Look-ahead does not rescue Mamba. The MLP leads every architecture and readout on
  every channel, so the Table 6.7 conclusion stands.
- **Reproducibility.** The published per-seed values and spreads on ISI do not reproduce. Seeded
  retraining is bit-exact on one machine and differs across machines.
- **Mamba-2 at QPSK.** The published cell selects a seed that misses 1e-3.

Thesis integration awaits the author: `docs/mamba_lookahead_rerun_2026-10-13.md`. The tests pass at
347, and the provenance audit is clean with the five new experiments registered.

## 2026-10-13 (later): Mamba rerun folded into PR #175

At the author's direction the rerun is now in the thesis, in Sec. 6.4.2 and Table 6.7:
- the dagger on the Mamba-2 QPSK cell, plus the 3.14 dB margin note;
- the reproducibility paragraph, replaced;
- the look-ahead sentence in the limitation paragraph.

The new verifier check `prose:seq-rerun` brings the verifier to 708 cells, 0 inconsistencies. The
build is 120 countable pages. Provenance labels and the ledger are updated, as are the companion
documents. #175 still awaits the author's approval.

## 2026-10-13 (last): author decisions applied to #175

The Hebrew abstract is approved. The H2 verdict reads "Supported as matching … none is detectably
better". The title pages read October 2026. The build is still 120 countable pages, and #175 awaits
the merge approval.

## 2026-10-13 (final): #175 merged; mirror republished

#175 was merged into main as 4bac934 on the author's approval. It carries the committee-audit fixes,
the A2 Mamba rerun and the author's three decisions. The Overleaf mirror was republished as a401ee4
under the author's identity. On main: 120 countable pages, verifier 708/0, provenance clean, 347 tests.

## 2026-10-13: red-team review (report only)

`docs/red_team_review_2026-10-13.md` contains 2 high, 5 medium and 4 low findings, plus 7 cleared
attacks. The new diagnostic `e6_blind_intervals.py` (results/e6_blind_intervals.json) is registered
in the provenance audit. Fixes await the author's choice.

## 2026-10-14: red-team fixes (#178, merged as fe8198b; mirror abf77f8)

All findings except R9 applied on `fix/red-team-findings`; the report's Outcome section lists each.
New: `e6_complexity_ported.py` timing protocol and committed output, `scripts/plot_e6_timing.py`,
`results/e6_timing_measured.png`; `e6_blind_intervals.py` now backs the thesis. Verifier 732/0,
provenance clean, 347 tests, 120 countable pages. #176 merged as 225d70e. Deleting the merged
remote branches was refused by the proxy and then skipped at the author's direction.

