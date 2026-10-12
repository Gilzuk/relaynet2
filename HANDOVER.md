# Handover — Khina revision round (task chain)

_Last updated: 2026-10-12. Khina round complete: every task merged, including a final page-by-page review (FR), a README pass (RM) and a figure check (FG), mirror republished after them, reply drafted and committed but not sent. Thesis 119 countable pages (Chapter 1 through the appendices; the faculty limit is 120; 135 PDF pages); verifier 678 cells; 347 tests. Nothing is left to implement; the open items are the author's decisions listed under "Next"._

**Independent recheck (2026-10-12):** branch `fix/khina-independent-audit-20261012` records additional wording/documentation corrections and a nine-comment evidence mapping in `docs/independent_khina_audit_2026-10-12.md`. The saved coded reversal is confirmed independently, but exact canonical training reproduction and verification against the absent original October email remain limited. Local validation is 343 passes / 4 Mamba-2 skips, not an independently confirmed 347 passes. Republish the generated mirror after this PR merges.

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
| P  | Republish mirror: done after Phase B (46ad144), after C4–C8 (f3eabf6, 133 pages), after the audit (4d3e992, 136 pages), after the second recheck (69a3386 = main e26361b, 138 pages), after #162 (162fb85 = main 1eaeb08), after the final review #164 (3d45423 = main eee7895, 119 countable pages), after the README pass #166 (5b1da55 = main c8f7710; only the generated README.md changed) after the Overleaf-guide fix #168 (2a3fd16 = main c4b6a0c; only OVERLEAF.md changed) and after the figure check #169 (4b0e1e1 = main 6809399; two chapters, six figures and thesis.pdf, 119 countable pages); author Gil Zukerman, `thesis.pdf` byte-identical to main each time | DONE | — |
| R  | Reply to Khina: drafted 2026-10-12, one paragraph per comment with the thesis location of each fix (no PR numbers); C4 reversal stated; ×2 multiplication-count convention; C7/C8 disclosures. Committed at the author's request: `thesis/submission/reply_to_khina_2026-10.md` (not sent); updated after the point-by-point audit (#158) and the second recheck (RC), section/table numbers re-checked against the 138-page build | DRAFTED, COMMITTED (awaiting author) | #156, #157, #159, #160 |
| RC | Second recheck of the rendered PDF against every sub-request: factorization caveat wherever M^L is stated (Fig. 6.7, Sec. 6.5, 2.6, O2, H5, Appendix); Table 5.11 "Look-ahead" caption; Sec. 4.1.5 matched destination metric; Ch. 1 contribution 3; Sec. 7.1.1, 7.6 wording; Hybrid described as coded (`hybrid_branch_selection.py`, verifier `prose:hybrid-branch`); sequence-model holdout; Appendix B Table 9.1; test counts | DONE | #160 |
| RC2 | Found while updating the documents: Ch. 7 (comparator scope) and the Ch. 8 closing paragraph still listed "fully reliability-weighted decoding" as unmeasured, contradicting Sec. 5.5.2; both now say a destination metric that also models relay decision errors is what remains unmeasured | DONE | #162 |
| FR | Final page-by-page review of the rendered PDF: body was 122 countable pages against the 120 limit (#160 added two) - now 119 by cutting redundancy where chapters ended with 2-3 lines; Table 5.13 ran off the page and five tables overflowed - 0 overfull boxes now; Sec. 4.2 still described a sqrt(2) axis rescaling (contradicting Eq. 4.16 and the code); Transformer rationale said d_ff 128 (code 64); cGAN lambda_L1 100 (code 20); Eq. 2.9 notation; DF closed-form percentages; Table 5.12 caption; boundary wording vs Sec. 3.1.1; appendix letters (no Appendix A) dropped; revision language removed; bibliography arXiv IDs and casing | DONE | #164 |
| D  | Tracking documents brought up to date: thesis, rerun and framework changelogs, README findings, REPRODUCE, context, Overleaf-sync guide (publish identity), Hebrew-abstract note, superseded cover letter marked, memory bank (techContext SNR trap, systemPatterns), table provenance regenerated; reply final pass | DONE | #163 |
| RM | Every README checked against the thesis and code: root `README.md` (Ch. 6 table restated with the thesis's own numbers plus the scalar MAP row and the factorized trellis count; Hybrid snippet shows the fixed 5 dB threshold on a per-axis power estimate, switch at 6-8 dB; Transformer d_ff 64; cGAN lambda_L1 = 20; Overleaf mirror row); `e6_unknown_channel_results/README.md` (backs Ch. 6 on main, S2/S4 not reported, October reruns, scripts write into the repo); `e6_multi_training_results/` and `e6_qpsk_rescaled_results/` marked historical/exploratory; `results/_archive_precorrection/` stale appendix reference removed; mirror README template in `scripts/overleaf_project.py` (AF qualifier, matched-metric coded reversal, factorized trellis); `thesis_overleaf*.zip` and `thesis_preview.pdf` rebuilt from the current thesis (both were September builds). Then the mirror's other doc, `thesis/OVERLEAF.md` (shipped in the project and the zips): Hebrew font is David CLM, not Arial; the `hebrewcal.sty` stub main.tex loads since #123 is explained; `thesis/OVERLEAF_SYNC.md`: 50-file project (18 figures, `hebrewcal.sty`) + README + thesis.pdf = 52, route B named, tooling on main, `origin/overleaf-dist` (2026-09-04) marked not current. Memory bank: techContext/projectbrief build recipe now `thesis/main.tex` via latexmk, not the old root `thesis_tau.tex` | DONE | #166, #168, #167 |
| FG | Figures checked against their data by regenerating the 16 reproducible ones from committed files (12 matched). Figs. 5.3-5.5 were drawn from earlier runs than Tables 5.8/5.12/5.13 (5.3 and 5.4 printed 20 and 10 trials; redrawn from the 100-trial runs); Table 5.8's 20 dB soft block-DF cell showed the doubled-variance run while its footnote said Fig. 5.3 plots it (cell back to the nominal 0.002348, doubled-variance value footnoted, reply corrected); Fig. 6.3 still had the removed Rayleigh-hop-2 panel (single AWGN panel, Fig. 6.2's style, title names the per-symbol fading, caption no longer says the MLP 'separates' from 2 dB); Fig. 6.7(a) had no script and read MLP-169 (new `scripts/plot_e6_complexity_cost.py`, MLP-170 reference line; panel (b) timed the 169-parameter I/Q relay, now stated); Fig. 5.5 axis said latency (now buffer-readiness budget); Fig. 5.1 title had an internal '(§7.10)'. 119 countable pages | DONE | #169 |

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

### Next (author decisions; nothing to implement)

1. **Send the reply** in `thesis/submission/reply_to_khina_2026-10.md` (drafted, not sent), with the
   revised PDF (`thesis/main.pdf`, 119 countable pages) or the Overleaf project. Read the Hebrew-abstract
   changes first (`thesis/HEBREW_ABSTRACT_SYNC.md`): that Hebrew is drafted, not the author's own.
2. **Overleaf:** pull once from GitHub (Menu → Sync → GitHub). If Overleaf still points at a pre-rewrite
   commit, unlink and relink first (no Overleaf-side edits exist).
3. **Mirror commit f174a3b** was authored "Claude"; rewriting it to the author's identity needs the
   author's OK (force-push to `relaynet2-thesis`).
4. **Title-page date** reads September 2026 (`thesis/chapters/frontmatter.tex`); change it to the submission
   month if the faculty requires that.
5. If Khina replies with further comments, start a new task chain here, one branch and PR per fix.

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
- Against conventional variable-gain AF the 0 dB margin is 0.007, not 0.073 (C3).
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
4. `python3 verify_thesis_tables.py` (678 cells now, 0 inconsistencies), `python3 provenance_audit.py`,
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
