# Handover — Khina revision round (task chain)

_Last updated: 2026-10-12. C4 done. Thesis PDF: 130 pages; verifier 612 cells; 347 tests._

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
   Phase A). **Remaining: the equal-budget discussion still alternates between acknowledging
   confounds and claiming to isolate architecture / capacity as dominant cause.**
9. Positioning: explain what the thesis adds beyond estimate-and-forward, Farsad–Goldsmith
   (learned detection without CSI) and ViterbiNet; Table 3.1 (`tbl:table17`) should compare
   **named studies** and properties, not "prior work: ad-hoc / none". Scalar-MAP reference (done, C2).

Presentation notes: running headers (#124), ledger labels (#135), verification-coverage wording
(#134), figures vs principal tables using different values (→ C5).

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
| C4b | Secondary coded studies rerun with matched metrics (`CODED_METRIC=matched`, 30 trials, published seeds; outputs `results/*_matched.json`, envelopes `coded_matched_envelopes.py`). Matched block DF stronger everywhere: relay symbol errors ~1e-5/1e-6 (Table 38); larger K now helps (8 dB: 0.0033/0.00069/6.7e-5); reliable regime = oracle (0 errors 24–32 dB); equal-throughput coding wins from 8 dB (12.3× at 16 dB); rate-adaptation envelope far above denoise 8–32 dB; 150-symbol budget: block DF leads ≥20 dB; 100-symbol structural lock-out survives; coded size-sweep penalties +7.6–9.9 dB. Matched values added beside published ones (Tables 38/40/42/43, prose), Ch. 8/9 updated; verifier check `coded-matched-secondary` (43 cells) | DONE | (this PR) |
| C5 | Corrected-figure vs superseded-table consistency (Khina presentation note). The ISI→Rayleigh (S2) and control (S4) rows had already been removed from `tbl:tableE6` and Fig. E6 (#122/#123), so no rerun was needed. Remaining instance fixed: Table 39's 20 dB soft block-DF cell now shows the doubled-variance run Fig. 5.x plots (0.001376), nominal 0.002348 in the footnote; verifier maps the cell to `coded_soft_df_calibration.json` | DONE | (this PR) |
| C6 | Causal/streaming normalization + latency re-evaluation (comment 6; only if readiness claims are kept) | TODO | — |
| C7 | E6_FLAT Hop-2 noise fixed (1/γ → 1/(2γ) for real forwarded signals, three sites in `e6_flat_ported.py`) and rerun on the original seeds. DF/MLP BERs roughly halve; DF-vs-MLP similarity holds from 8 dB; below 8 dB the unknown-phase MLP trails differential detection by up to 0.0106 at 0 dB (exceeds the 0.0036 control target in CLAUDE.md; reported, not tuned). Table `tbl:tableE6flat`, Ch. 6 conclusion and Appendix G updated; new verifier check `prose:flat-conclusion` | DONE | (this PR) |
| C8 | Training protocol now described from the committed code: all relays train with Adam (MLP/Hybrid lr 0.01 batch 32 — the NumPy SGD path is a no-PyTorch fallback the canonical run excludes; VAE/sequence 1e-3 batch 64; cGAN 1e-4, β=(0,0.9)); samples 25k (MLP, Hybrid) / 50k (VAE, cGAN, sequence), 100 epochs (cGAN 200). Exact reproduction is not possible (unseeded torch init, global-RNG channel draws; data predate repo history) — stated once in Ch. 4. Fixed #133's "canonical SGD" and Ch. 4's "trained by the same protocol" | DONE | (this PR) |
| M  | Memory bank update for this round (`activeContext.md`, `progress.md`) | TODO (with each PR) | — |
| P  | Republish mirror after Phase B (DONE: 46ad144, author Gil Zukerman, PDF identical to main) and again after C4–C8 | PARTIAL | — |
| R  | Draft reply to Khina (note the ×2 multiplication-count convention: thesis 16/64/256, crossover 6.70, ratio 1.23 vs his 32/128/512, 5.70, 2.46) | TODO | — |

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
5. **C6: causal normalization** (comment 6) — only if the 5/10-symbol readiness claims are kept;
   otherwise remove those claims and close C6.
6. **M: memory bank** — `memory-bank/activeContext.md` and `progress.md` entry for the whole round.
7. **P: republish the mirror** with the author identity (see Environment gotchas); verify fast-forward
   and that `thesis.pdf` is byte-identical to `thesis/main.pdf`.
8. **R: reply to Khina** — draft only (do not send): one paragraph per comment with the PR that answers
   it, the C4 reversal stated plainly, the ×2 multiplication-count convention, and C7/C8 disclosures.

### Task notes

**B2 (done).** Files: `thesis/chapters/ch03_objectives.tex` (O1/O2 at `sec:two-objectives`, H1–H5 at
`sec:research-hypotheses`, Specific Objectives). Add a short "hierarchy" paragraph: main objective →
O1/O2 → H5 + Ch. 6 studies answer O1; joint latency–memory study (`sec:joint-latency-memory`) and
pilot sweep (`sec:partial-posterior`) answer O2; H1–H4 are the canonical control that calibrates the
framework, not the principal contribution. Add a compact table: RQ / hypothesis → section(s) →
experiment script(s) → verdict location (Ch. 9 table). Use `\ref` only, never raw labels. Do not
claim the studies are cumulative relaxations or measure a common boundary.

**B3.** Mostly `ch07_unknown_and_mismatch_channels.tex` (renders as Ch. 6) and the appendix. Read
the generating scripts (`e6_*_ported.py`, `e6_sim_enhanced_multimod.py`, `relaynet/channels/e6_channels.py`)
and transcribe the actual laws — never invent a distribution. The SNR-penalty definitions must
match the code that computes them (find it via `provenance_audit.py` REGISTRY / `verify_thesis_tables.py`).

**B5.** Needs bib entries checked: Farsad–Goldsmith and ViterbiNet (Shlezinger et al.) are already
cited in the appendix and Ch. 6 — reuse `FarsadGoldsmith2018NNDetection` and `Shlezinger2020ViterbiNet`; estimate-and-forward is not cited yet (Khojastepour/
Sabharwal/Aazhang "estimate-and-forward" / Cover–El Gamal compress-forward lineage). Verify any new
reference's identity before adding it. (Abou-Faycal & Médard, "Optimal uncoded regeneration for binary
antipodal signaling", ICC 2004, could NOT be verified and is not cited.) Watch the page budget.

**C4.** Coded pipelines: `relaynet/relays/coded_df.py`, `soft_coded_df.py`, `relaynet/coding/`.
Matched metric: weight each zero-forced squared residual by |h|² (i.e., use the un-equalized metric
|y − h x|²). Rerun the principal coded comparisons into `results/`, add REGISTRY rows, update the
verifier. **This can reverse coded conclusions — report whatever comes out.**

**C5.** `tbl:tableE6` Rayleigh rows vs the corrected figures; rerun with the current SNR convention
(γ = 10^(SNR/10)), commit data, update table + verifier.

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
- Script comments in `e6_blind_ported.py` call Hop 2 "always-complex AWGN"; it is compensated Rayleigh
  (`ComplexISIRayleighChannel(taps=[1])`).

## Procedure (every task)

1. `git fetch origin && git checkout -B <new-branch> origin/main` — **confirm `git branch --show-current`
   before any reset**; never `git reset --hard` on `claude/porting-md-file-l6xzsr`.
2. Edit. Results go into the repo (`results/`), never `/tmp/`. New experiment → REGISTRY row in
   `provenance_audit.py` + a check in `verify_thesis_tables.py`.
3. Build: `cd thesis && latexmk -xelatex -g -interaction=nonstopmode main.tex` three times; check exit
   code, `grep -c '^!' main.log` = 0, `grep -c undefined main.log` = 0, `pdfinfo main.pdf` pages.
4. `python3 verify_thesis_tables.py` (540 cells now, 0 inconsistencies), `python3 provenance_audit.py`,
   `python3 -m pytest -q tests` (343 pass).
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
