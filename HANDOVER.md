# Handover — Khina revision round (task chain)

_Last updated: 2026-10-12. Thesis PDF: 127 pages after B3; verifier 558 cells._

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
| B2 | RQ hierarchy + RQ→experiment mapping table (comment 2); O1/O2 answered by name in Ch. 9 | DONE | (this PR) |
| B3 | Model specifications + SNR-penalty definitions (comment 7): Appendix G; ISI penalties now derived by `e6_isi_penalty.py` and verified | DONE | (this PR) |
| B4 | Equal-budget wording pass (comment 8, last bullet) | DONE | (this PR) |
| B5 | Positioning vs estimate-and-forward, Farsad–Goldsmith, ViterbiNet; named-study Table 3.1 (comment 9) | NEXT | — |
| C4 | Reliability-weighted Viterbi/BCJR at relay + destination; rerun coded comparisons (comment 4) | TODO | — |
| C5 | Rerun Rayleigh rows of `tbl:tableE6` on the current convention (figures vs tables) | TODO | — |
| C6 | Causal/streaming normalization + latency re-evaluation (comment 6; only if readiness claims are kept) | TODO | — |
| C7 | E6_FLAT Hop-2 noise: `e6_flat_ported.py` adds real noise of variance 1/γ (not 1/(2γ)) to real forwarded signals; AF in flat-phase gets complex 1/(2γ)/dim. DF-vs-MLP fair, absolute BERs 3 dB pessimistic on Hop 2, AF favoured in flat-phase. Fix the noise, rerun, update `tbl` flat rows + Appendix G paragraph | TODO | — |
| C8 | Training-protocol description vs code: Ch. 4 says 25k/20k/10k samples (supervised/generative/sequence) and NumPy SGD for MLP/Hybrid; `run_experiments.py::train_base_relays` uses 25k/50k/50k (5k/5k/3k with `--quick`) and `MinimalGenAIRelay` uses PyTorch Adam lr 0.01 whenever torch is installed. No log records which settings produced the canonical weights. Establish it (retrain from the committed seed and compare with `results/modulation/qpsk_rayleigh.json`, or locate the weights' provenance) before correcting the text; also re-check Phase A #133's "canonical ReLU/SGD" | TODO | — |
| M  | Memory bank update for this round (`activeContext.md`, `progress.md`) | TODO (with each PR) | — |
| P  | Republish mirror after Phase B and after C4–C6 | TODO | — |
| R  | Draft reply to Khina (note the ×2 multiplication-count convention: thesis 16/64/256, crossover 6.70, ratio 1.23 vs his 32/128/512, 5.70, 2.46) | TODO | — |

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
reference's identity before adding it. Watch the page budget.

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
