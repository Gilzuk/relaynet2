# Active Context (update this file first, every session)

_Last updated: 2026-08-24_

## CURRENT STATE: merged to main; thesis at 120 pages

PR #23 (`claude/porting-md-file-l6xzsr` -> `main`) merged 2026-08-23/24 on top
of PR #15 below: block-DF reliable-decoding-regime study (new Table 5.7),
AK #33 fix, abstract reconciliation, and an Appendix E repair (chapter-
numbering drift from the 16-QAM removal below). Full writeup at the bottom of
this file, "Latest (2026-08-24)". A further branch restart after that merge
removed every BPSK-with-Rayleigh configuration from Ch.6 (unknown-channel
chapter), per author instruction -- see the same bottom entry.

PR #15 (`claude/porting-md-file-l6xzsr` -> `main`) merged 2026-08-22. **`main`
is the base and the source of truth from here**; branch off it for new work and
merge back through a pull request. Do not commit to `main` directly -- see the
workflow note below. The old feature branch is fully contained in `main` and can
be deleted.

**Conflict you will hit, and how it resolves.** The "Git Safety" section of
`.clinerules/90-safety.md` still reads "Always push to `clean-thesis` branch
only." Do not follow it for new work. That instruction is superseded by fact,
not by preference. As of this note (2026-08-22) `clean-thesis` had not moved
since 2026-07-18, contained no commit `main` lacked, and `main` was 128
commits ahead of it. Those counts drift; re-derive them with
`git rev-list --count origin/clean-thesis..origin/main` and its reverse rather
than trusting the numbers here. What does not drift is the direction: every
`chapters/**` change through the 120-page reduction is on `main` and none of it
is on `clean-thesis`, so pushing thesis edits there would fork the thesis onto
a dead branch. The rule text was deliberately left in place by the repository
owner, who confirmed `clean-thesis` is a stalled branch; it is recorded here as superseded rather
than edited there. If the owner ever revives `clean-thesis`, this note is what
needs revisiting first.

**Workflow in force (owner instruction, 2026-08-22).** Every fix goes on a
branch and returns through a pull request with a Copilot review requested; when
the review posts, its findings are verified against the source, the real ones
fixed, and the cycle repeats until the review is clean. Nothing is pushed
straight to `main`.

What landed after 2026-08-19, in order:

1. **Theoretical-bounds verification.** DF-theory and single-hop-floor overlays
   added to Fig. 10 and Table 2; ergodic Rayleigh (Shannon) capacity column
   added to Table 42. `verify_thesis_tables.py` extended to check both.
2. **Two rounds of independent review addressed.** Round 1: revision
   annotations suppressed, master experiment ledger appendix, Limitations for
   single-seed training and unequal training budgets, multiple-hypothesis
   caveat, generalization language tightened. Round 2 (five blocking items):
   stale canonical BERs in Ch.8 replaced from Table 2's 0 dB row; a figure
   captioned AWGN in a Rayleigh chapter corrected; the false "sequence- and
   bit-optimal detection coincide for BPSK" claim fixed in both places;
   **H3 downgraded to Partially supported** and the 11,201-parameter
   "overfitting" claim withdrawn (that model exists in no results file);
   **H4 renamed to an equal-parameter-budget comparison**.
3. **Copilot review.** Three findings, all real: a tail double-count in
   `coded_latency_capacity.py` (frame lengths 1-2 symbols long; no MCS
   selection flipped), `NaN` in a results JSON, a sentence fragment in Ch.2.
   A second pass added two more, fixed here.
4. **Scope reduction to 120 pages (from 151).** Removed: the 16-QAM extension
   chapter, the Mamba-S6 study (**the old H5, the SSM training-speed
   hypothesis, was dropped**), the AWGN calibration study, the
   per-architecture runtime table,
   the constraint-length sweep table (finding kept as a paragraph), and
   figures that only re-plotted an adjacent table. 16-QAM survives *only* as
   an MCS rung in the link-adaptation study, with a scope note saying so.
   Mamba-S6 survives as a data point in the canonical comparison, since
   dropping one architecture's measured result while keeping its peers would
   be selective reporting.

Standing verification gates, all green at the merge: `verify_thesis_tables.py`
389 cells / 0 inconsistencies, 159 tests, cold XeLaTeX rebuild 0 errors /
0 undefined refs / 0 bidi errors, 120 pages.

**Hypothesis numbering (renumbered 2026-08-22).** Dropping the old H5 left the
thesis posing H1-H4 plus H6, with a visible gap at 5. The unknown-channel
hypothesis has therefore been renumbered **H6 -> H5**, so the thesis now poses
a contiguous **H1-H5** and the Chapter 3 list renders 1-5. No hypothesis
statement, result or outcome changed -- only the label. Historical records
(`RERUN_CHANGELOG.md`, the `_`-prefixed dead chapter files) still use the old
numbering on purpose and must not be renumbered.

**Supervisor review, 2026-08-22 — "READY WITH MINOR REQUIRED REVISIONS".** Of the eight
items, four were applied (items 4, 5, 6 and 7), item 3 was declined, items 1
and 2 were not defects, and item 8 needs a human. Items 1 and 2
("AComparative", "SystemModel") do not exist in the source or the rendered
PDF and are artifacts of the reviewer's text extraction, most likely from the
stale Overleaf copy -- do not "fix" them.
Item 3 (script and JSON names in the ledger) was declined: those sit in Appendix C,
which is where reproducibility artifacts belong. Item 8 (faculty page limit) needs
a human against the formal regulations: 120 pages total = front matter 1-6, body
7-109, references 110-112, appendices 113-117, Hebrew abstract 118-120.

**Outstanding follow-ups.** The Hebrew abstract is machine-produced and needs a
native speaker. Supervisors last saw a six-hypothesis draft in which H5 was the
SSM training-speed hypothesis; they need telling both that it was withdrawn and
that "H5" now denotes the unknown-channel hypothesis they knew as H6. Note that the "thesis submitted" decision recorded
further down refers to an *earlier* version; whether this 120-page revision has
been submitted has not been stated, so neither assume it has nor that it has
not.

## DECISION IN FORCE: the cGAN stays out of the results

Instruction 2026-08-18: "drop the cgan for now". The cGAN-inclusive re-run was stopped mid-flight and its driver deleted. **Do not re-add the cGAN without a fresh instruction.**

Two findings from the attempt, both worth not rediscovering:

1. **The surviving cGAN numbers cannot be reused.** `relaynet/relays/cgan.py` was created 2026-04-01; the cGAN results still in the repo are dated 2026-03-23, so they come from a superseded implementation. This is the same trap as the VAE: its March numbers (0.3928 at 0 dB) differ from the current implementation's (0.3359) by far more than the SNR-convention change explains, which `results/vae_convention_ab.json` demonstrates by controlled A/B. Splicing March cGAN values into tables measured with today's code would be mixing implementations, not filling a gap.
2. **A cGAN column cannot be run separately and pasted in.** Training it consumes the shared RNG stream, so its presence shifts every other relay's numbers; a table assembled that way would have rows from two different draws. Adding the cGAN means re-running each affected study end to end and re-transcribing.

**Cost, measured not guessed:** a probe at 2,000 samples x 5 epochs took 4.1 s, extrapolating to ~1.1 h per training at the production 50,000 x 200. The real run blew past that — over 1 h 50 m on a single training with no completion — so the linear extrapolation from a small probe understates it badly. Section 7.11 would need three trainings, one per activation, and is the stage to drop first if this is ever revisited.

**State is unaffected.** The run was killed before writing any output; every result file is the one committed earlier, and the verifier is at 349 cells / 0 inconsistencies. Tables 2, 14, 15 and 24 report eight relays and say in their captions that the cGAN is excluded.

## Latest: AWGN companion comparison removed from Ch5 — QPSK/Rayleigh takes its slot
User: "move the qpsk Rayleigh to chapter 5 instead of awgn over Rayleigh". Ch5's canonical section had three tables (Rayleigh BPSK / AWGN companion / QPSK-vs-BPSK on Rayleigh); the AWGN companion is now deleted and QPSK/Rayleigh is the second table (**now Table 5.5**, renumbered from 5.6).
Removed: `tbl:table2awgn` + its 2 paragraphs, `fig:fig10awgn`. `results/awgn_comparison_ci.png` and `scripts/plot_awgn_companion.py` are now orphaned — left in the repo, not referenced.
**[SUPERSEDED 2026-08-22 — §5.2 no longer exists.]** It was removed entirely in the 151→120 page reduction, and the simulator is now calibrated on the canonical channel itself against the closed-form two-hop DF composition. Do not restore it. *At the time of this entry the position was:* §5.2 (AWGN calibration) is untouched and must stay, because the closed forms the simulator validates against are AWGN expressions. What went away is any *relay* measured on AWGN in the canonical chapter. Appendix E's AK#4 response was updated accordingly; it had cited the companion table as part of AWGN's bounded role, so leaving it would have dangled a `\ref` and misdescribed the design.
`check_table2awgn` deleted from `verify_thesis_tables.py` (source map + registry + function). **Verifier now 327 cells / 0 inconsistencies** (was 352; the 25 are the companion's). Cold build: exit 0, **146 pp**, 0 undefined refs.
**Build gotcha reconfirmed:** deleting `main.aux` before `latexmk` makes bibtex exit 12 with "I found no \citation commands". Not a real failure — just run `latexmk` again (twice, to settle refs). Cost 2 extra passes to rediscover.

## SUPERSEDED (2026-08-22): thesis tables stay frozen

**This decision is no longer in force.** It is kept for history because it
explains why `results/` and `chapters/` diverged for a period. It was overtaken
by explicit later instructions that required editing chapter tables directly:
adding the DF-theory and Shannon-capacity columns, correcting the stale Ch.8
BER figures, and the scope reduction that removed several tables outright. Every
such change was verified against its source file by `verify_thesis_tables.py`,
which reports 389 cells and 0 inconsistencies. Do not re-freeze tables on the
strength of the text below without a fresh instruction.

### Original decision (historical)
User submitted the thesis, then instructed "Keep" in answer to a choice between (a) freeze the submitted tables and land the re-run as data only, and (b) carry on re-transcribing. Reading taken: **(a)**. Consistent with `.clinerules/90-safety.md` (never alter numerical results without explicit instruction), and the cheaper error to recover from if the reading is wrong.

**So: commit re-run outputs under `results/`, do NOT re-transcribe any `chapters/*.tex` table.** The submitted PDF and the branch will therefore diverge on data — that is intended, not drift. Do not "fix" the mismatch by quietly updating tables; it needs a fresh instruction.

**When integration is eventually authorised**, the affected surface is:
- `tbl:table2` + `tbl:table2awgn` (ch05) ← `results/bpsk_comparison/{rayleigh,awgn}.json` — loses its cGAN column
- `tbl:table14ray` (ch05 Table 5.6, canonical QPSK) + `tbl:table14` (ch06) ← `results/modulation/*.json`
- `tbl:table24` (ch06) ← `results/all_relays_16class/` — loses its cGAN row; ch06 prose currently cites cGAN's 16-class failure (0.353/0.282) and must stop
- `tbl:table15` (ch06) ← `results/qam16_activation/` — lean 5-relay set
- `tbl:table8` (ch05, hypothesis H4) ← `results/normalized_3k/3k_rayleigh.json`
- §5.2's lean-set paragraph, which was written when only the AWGN companion was lean
Then: `python3 verify_thesis_tables.py`, cold `latexmk -xelatex`, `pytest tests/`.

## Why the re-run covers everything, not just the AWGN/BPSK tables
Two findings from scoping it, both worth not re-deriving:
1. **Every learned relay trains through `awgn_channel`** (`relaynet/utils/activations.py:214`). The 3 dB fix therefore moved the trained *weights*, so Rayleigh and QPSK/16-QAM results are stale too — even though `fading.py` was never touched and complex modulations never hit the corrected real-noise branch. Do not assume "channel unchanged ⇒ results valid" in this repo.
2. **`results/bpsk_comparison/rayleigh.json` (source of the canonical headline table) carried `created=2026-03-23`.** Its Aug-17 mtime was just the restore-copy made after an earlier lean run overwrote it. **Judge result freshness by the `created` field inside the JSON, never by file mtime** — a pre-fix backup copy looks fresh to `ls`.

## Re-run in flight (`scripts/rerun_all_experiments.sh`, started 16:56Z)
7 stages, sequential (4 cores, no CUDA). cGAN dropped everywhere per the lean instruction. Two tiers, deliberately not uniform: **breadth (8 relays)** for 7.2/7.3, 7.10, 7.8, 7.17 — those tables exist to compare architectures, and `tbl:table8` *is* hypothesis H4, so leaning them would delete findings rather than shrink them; **lean (AF/DF/MLP/Transformer/Mamba-2)** for the 7.11 and 7.13 activation ablations, where the variable is the activation and 7.13 is 12 combinations.
Order: 7.1 ✅(7s) → 7.2 → 7.10 → 7.17 → 7.8 → 7.11 → 7.13, chosen so an interruption still leaves the load-bearing tables refreshed. Est. 12–16 h. Logs in `results/rerun_logs/`.
`--skip-relays` had to be fixed first: it matched literal display names, but one relay appears as "MLP (169p)"/"MLP-3K"/"MLP 16-cls", so lean runs of 7.8 and 7.17 silently skipped nothing. Now family-based (`relay_family()`), mamba2 tested before mamba_s6.

## Latest: canonical restructure — Rayleigh carries BPSK **and** QPSK; AWGN is the baseline
Swapped what Ch5 and Ch6 each own. Previously Ch5's headline comparison ran on AWGN/BPSK (the one channel Appendix E comment 4 says the thesis draws no conclusions from) while QPSK-on-the-canonical-channel sat in the extension chapter — backwards. Now:
- **§5.2 "Simulation Baseline: AWGN Calibration"** — AWGN's role stated up front: it is where closed forms exist, so it is what the simulator is validated against. Its relay comparison (Table 5.5, **lean set AF/DF/MLP/Transformer/Mamba-2**, 0–8 dB) is a baseline; no relay is ranked on it. Cut at 8 dB because beyond that every relay reads zero at any feasible bit budget (20 dB AWGN needs ~6.6e24 bits).
- **§5.3 "Canonical Relay Comparison (SISO, Rayleigh, BPSK and QPSK)"** — the QPSK-vs-BPSK block moved in from Ch6, now **Table 5.6**.
- **Ch6 "Extension: Multi-Level Modulation (16-QAM)"** — keeps only the constellation needing a different relay formulation; its QPSK block replaced by a pointer to §5.3.

**Seven places asserted the canonical setup as BPSK-only** and all now read "complex baseband, BPSK and QPSK": ch01 (scope para, scope-table caption, Channel row, Modulation row, E1/E2/E4 summary rows), ch03, ch04, ch08, ch09. Appendix E's AK#4 response argues why this is still *one* setting: topology/channel/metric each stay single-valued and QPSK is the two-axis use of the same baseband through identical relays.

**Verified:** cold `latexmk -xelatex` exit 0, **149 pp, 0 undefined refs**; `verify_thesis_tables.py` **352 cells / 0 inconsistencies**; `pytest tests/` **119 passed**.

**Open follow-ups (NOT done):**
1. **Ch6's AWGN tables `tbl:table14`/`table15`/`table24` are still on the pre-correction AWGN convention** — they predate the 3 dB `sigma^2 = N0/2` fix and need a re-run (~1 h each even on the lean set). The verifier does not cover them (see its own "informational" footer).
2. **Ch7 flat-channel control passes by 0.0097 against a 0.010 tolerance** — margin too thin to trust; needs a larger trial budget or a defended tighter tolerance.
3. ~~Overleaf push~~ — **CLOSED, do not attempt. The user syncs Overleaf manually** (instruction, 2026-08-17: "Stop overleaf sync I will do it manually"). The `overleaf` remote (`git.overleaf.com/69cd8f24043dbf2a2982370`) is deliberately left configured because the user needs it; its presence is **not** an invitation to push. Do not run `git push overleaf`, do not rebuild `git subtree` splits, and do not reopen the problem — the agent proxy 403s that host anyway, Overleaf bans the force-push the divergent history would need, and the project carries 163 objects of unrelated history. Deliver work by pushing to `claude/porting-md-file-l6xzsr` on GitHub and leave the Overleaf leg to the user.

## Latest: MIMO equalization section REMOVED (completing AK comments 2/4/6)
Audited all 17 supervisor (AK) comments in `thesis/ak_comments.json` against Appendix E — all 17 documented, and the verifiable claims hold EXCEPT two gaps found:
1. **AK ch02 #7 (MMSE→LMMSE) was only partially applied** — 3 equalizer refs still said MMSE (SIC ordering/detection steps + the "outperforms linear MMSE" sentence). Fixed first, then made moot by (2).
2. **The whole "Theoretical Foundations: MIMO Equalization" section had survived a deletion Appendix E already claimed** — Appendix E's "Summary of removed material" explicitly lists "the $2\times2$ MIMO second hop and its ZF/LMMSE/SIC equalization" as removed (AK comments 2, 4, 6), but §2.6 was still in ch02. User caught this ("The mmse was removed that's why not appearing" / "The MIMO"). **Section now deleted** (36 lines: ZF/LMMSE/LMMSE-SIC equations, V-BLAST steps, 2 eq labels + section label). Nothing depended on it: 0 citations inside, 0 refs to its equation labels; the only 2 refs to its section label were in Appendix C (rewritten to point at future-work instead) and Appendix F item 11 (rewritten to record the removal). Post-removal label/ref audit: 220 labels, 181 refs, **0 broken**.
**Consequence:** "MMSE" now appears in the thesis ONLY in its correct non-linear sense (MMSE-optimal estimator E[x|y]) — so AK's LMMSE terminology comment is fully satisfied by construction. Pages: annotated 131, clean 128.

## Latest: all 24 REV findings resolved inline + Appendix F added (on main)
User reported still seeing `\REV` blocks flagging *unfixed* issues. Applied the actual fix for every outstanding recommendation, removed all REV/revblock artifacts from every compiled chapter (now **zero** repo-wide), and added **Appendix F "Independent Review Findings and Resolutions"** documenting all 24 findings in 3 groups (11 substantive / 8 consistency / 5 editorial) + a "recorded but no change needed" paragraph for the praise items.

Substantive fixes applied this round: AF end-to-end SNR now cited + variable-gain assumption stated; universal-approximation overclaim split into exact-fixed-σ² vs approximate-family; Mamba **A**=−exp(A_log) Hurwitz convention stated (the exp(ΔA)→0 step was unjustified); VAE result caveated (stochastic z at inference, deterministic-decoding variant never tested — paradigm claim softened in BOTH ch02 §2.3.3 and ch05); "identifiability" → **estimation-variance cliff** at 5 pilots (LS is overdetermined, 5 eqs/3 taps — the real mechanism is variance; CI 0.119±0.084 now quoted, pulled from the npy); "only option that remains reliable" → "only *one-shot* option" (CMA is also pilot-free, untested at L=40); wall-clock caveat; ch03 Perfect-CSI scope qualified to canonical core; ch04 eq:relay-received-siso rewritten to post-compensation |h₁|x+ñ form + h(·)→𝓗(·) to kill symbol collision; DF composition formula exact form; MIMO §2.6 reframed as background.
Editorial: misdirected (Equation~) pointers repointed; noise-convention **quote→numbered Remark** (was a bare \ref resolving to the section number) + both citing sites updated; Mamba block markup rebuilt (addcontentsline was inside the equation env); enumeration (i)(ii)(iv)(v)→(iii)(iv); ch06 stale `h1-partially-confirmed` label renamed; ch07 item 8 split into 3 items.

**Then, per user request, the annotations were RESTORED as inline `Fix applied:` records** (27 of them, one per fix site) and kept in BOTH bundles including the clean one — the user wants the record of what changed visible inline, not only in Appendix F. Then, on the follow-up instruction "Unclean", the two bundles were **re-separated**: `thesis_overleaf.zip` = annotated (27 notes, 132 pp), `thesis_overleaf_clean.zip` = stripped submission copy (0 notes, 129 pp). Appendix F is in BOTH — it is the permanent record, so the clean copy still documents every fix. Repo `thesis/` source stays annotated (master); the clean bundle is generated from it by the stripper. **Gotcha found while doing this:** one annotation (ch04 activation) had been spliced mid-paragraph, so the own-line strip regex missed it (26/27 strippable) — it was relocated to the end of its paragraph. If you add annotations, always put them on their OWN line or the clean build silently keeps them. The praise-only and already-recorded notes were NOT restored; only change-records. Both verified via cold-start `latexmk -xelatex`: exit 0, 132 pp, 0 undefined refs/cites, 0 bidi errors, no `??`, Hebrew intact, Appendix F present, 27 fix annotations rendered. Verifier still 200 cells / 0 inconsistencies.
**Gotcha:** a mid-edit `latexmk` run can exit 12 with "bibtex: I found no \citation commands" — that's a stale/partial `chapters/*.aux` race, not a real failure. Always re-check from a cold copy (`rm -f main.aux main.bbl chapters/*.aux`) before believing it.

## Latest: E6 blind/partial/composite now fully in the verification loop (on main)
Closed the last reproducibility gap. The three scripts already saved npys to /tmp — the July-14 originals (the run the thesis prose was transcribed from) were still sitting in this container's /tmp and were rescued into `e6_unknown_channel_results/` moments before a fresh rerun would have overwritten them. Verifier extended with `prose:E6blind/partial/composite` checks (12 new cells parsing the ch05 prose claims via anchored regexes) + these 3 scripts added to `--rerun`. **Now 200 cells, 0 inconsistencies.** One prose fix surfaced: the blind-MLSE "mid-SNR CI 0.164" is the 8 dB value (npy), and the MLP companion was 0.008 not 0.014 → prose now reads "mid-SNR (8 dB) ... 0.164 (versus 0.008 ... 0.011)"; verifier parses the declared dB. **Key finding: the E6 ported scripts are fully seeded — a from-scratch rerun reproduces the committed npys bit-for-bit** (verified on all three: every checked quantity identical to 4 decimals). PR #8 (clean-thesis→main) closed per user instruction; repo has zero open PRs.

## Latest: chapter-by-chapter fact/hallucination review + fixes (on main)
User asked for a full review of the thesis for wrong facts / LLM hallucinations, then approved fixing everything found. Verdict: no fabricated results — all analytics recomputed by hand check out (ISI slicer amplitudes, 0.25 floor, all parameter counts 169/170/1777/2946, trellis op counts, "126 pytest tests" exact). Fixes applied (all chapters + bib + zips + PDF, verifier still 0 flags):
- **references.bib**: `SamuelDiskinWunder2019MIMODetect` was a hallucinated entry (wrong 3rd author "Wunder", wrong title, wrong pages) → corrected to Samuel/Diskin/**Wiesel**, "Learning to detect", TSP 67(10):2554–2564. Removed duplicate entries `Nosratinia2004Cooperative` and `dao2024transformersssmsgeneralizedmodels` (same papers under 2 keys → appeared twice in printed refs); cites repointed. Bibliography now 27 unique entries (was 29 w/ dupes).
- **ch03**: "The Mamba architecture \cite{GuGoelRe2022SSM}" misattributed the S4 paper → `gu2023mamba`; "entirely unexplored" (PHY-wide) softened to relay-scoped.
- **ch02**: eq:df-ber-hops was P1+(1−P1)P2 (union, wrong by P1P2) → exact odd-flip form P1(1−P2)+(1−P1)P2 with reconciliation note; §2.6 MIMO reframed as background-only (no MIMO results exist), MMSE→LMMSE per Appendix E's own claim, implementation note moved to Appendix C.
- **table24 + fig51 (KEY DATA-PROVENANCE FIND)**: `grouped_bar_16class.png` was from an OLDER run contradicting the committed JSON (old fig: VAE 4-cls 0.4992 failed, CGAN 4-cls 0.0081 fine; JSON: VAE 4-cls 0.0087 fine, CGAN fails both 0.353/0.282). `ber_all_relays_16class.png` and `top3_16class.png` had already been regenerated (have `_orig` backups); grouped_bar was missed. Regenerated grouped_bar from committed JSON (root + thesis copies); rewrote fig51 caption; set ALL table24 cells from JSON (single provenance): 4-cls 0.00867 across relays, 16-cls MLP 0.00017/S6 0.00033/Mamba2 0.00167, VAE/Hybrid/Transformer 0 (caption notes resolution floor ~2e-4, no ratio quoted for zero cells).
- **Timing drifts**: "<3 s training" → "<5 s" (abstract, ch06 ×2; table13 says 4.9 s); "S6 1.83 s" → 1.88 s (ch06 vs table13).
- **ch07**: "a few dB behind Viterbi" → precise 1–1.5 dB (ISI) / ~2 dB (composite); 30–90× wall-clock claims now carry reference-implementation caveat (ch05 §complexity, fig caption, ch07).
- **Appendices**: stale `ch00_frontmatter.tex` filename → `frontmatter.tex`; VAE "PyTorch (CUDA)" → "(CPU/CUDA)"; Appendix C now documents the shipped-but-unused MIMO equalizers.
- Both Overleaf zips patched (full keeps REV, clean REV-stripped via validated stripper — regex: remove revblock env + lines that are solely `\REV{...}`); clean zip fresh-compile verified: 124 pp, 0 undefined refs/citations, no REV leakage. Repo PDF rebuilt (129 pp annotated).
- **Still unverifiable from committed data**: E6 blind/partial/composite prose numbers (CMA 2.4e-3 etc.) — only PNGs + scripts committed, no npy; suggested running `e6_{blind,partial,composite}_ported.py` once to close the loop. Bergel/Akdemir bib entries unverified (user to confirm).

## Latest: 6 verifier-flagged thesis table cells fixed (branch claude/cleanup-temp-scripts)
Completed the "6 Flagg cells fix" from the `verify_thesis_tables.py` verification work. `verify_thesis_tables.py` (the single publish-ready verification script; reads thesis `.tex` longtable rows and compares cell-by-cell against committed JSON/npy/closed-form sources) flagged 6 transcription errors. All fixed in `chapters/ch05_experiments.tex` (commit `ab2f250`), each corrected to its authoritative source — no numbers tuned:
- **ber_validation Rayleigh row** (4 dB, 16 dB): `9.10E-2`/`7.58E-3` → `7.71E-2`/`6.17E-3`. Matches closed-form `0.5*(1-sqrt(g/(1+g)))`, confirmed by 4M-bit MC. 10 dB was already correct.
- **table24 CGAN row**: copy-paste artifact (`0.00811 / : / :`) → committed 16-class results `0.3530 / 0.2817 / ---`.
- **tableE6flat "Unknown gain" DF + MLP-170 rows**: stale pre-bugfix values → committed post-bugfix `e6_unknown_channel_results/e6_flat_ported_results.npy` (DF `0.0681/0.0330/0.0131/0.0050`, MLP `0.0682/0.0333/0.0132/0.0050`). **Gotcha**: the verifier references `res[rkey][0]` — the *first trial row* of the (2,11) array, NOT the mean over trials. The phase and iqimb rows already matched row0, so gain was simply transcribed from stale data. Don't compute `.mean(axis=0)` when reproducing these — use row0 to match the verifier's convention.

After fix: verifier passes 185/185 cells, 0 inconsistencies, exit 0. (STOCHASTIC_TABLES tolerances: tableE6/tableE6flat 0.010, table24 0.002 — this is why only the 8/12 dB gain cells were flagged, not the within-tolerance 16/20 dB.) **Branch note**: this session's designated branch is `claude/porting-md-file-l6xzsr`, but the fix was committed to `claude/cleanup-temp-scripts` because its base (the verifier + cleaned repo, commit `75d7f9e`) lives only there — flagged to user.

## Latest: full LaTeX environment set up + × mojibake fixed on clean-thesis
Two follow-ups from the thesis-review/PDF work: (1) got a fully faithful compile working (real Hebrew RTL via `bidi.sty` from `texlive-lang-arabic`, real Times New Roman/Arial/Courier New via `ttf-mscorefonts-installer`, worked around a python3.11/3.12 `apt_pkg` mismatch blocking its dependency's postinst) — see `techContext.md` "Compiling the thesis" section for the exact recipe, this environment starts with zero LaTeX installed every session. (2) Found and fixed a real, pre-existing bug: 12 occurrences of a double-UTF-8-encoded `×` (rendering as `Ã—`) in `chapters/ch09_appendices.tex`, introduced during an old table/figure relocation pass. **This time did it correctly**: checked `clean-thesis`'s actual history first (confirmed present on its real tip, confirmed unintentional via `git log -S`), applied the fix by checking out `clean-thesis` directly (not this branch), committed and pushed there (`d626b67`), then switched back to `claude/porting-md-file-l6xzsr`. Verified via full recompile before AND after committing (zero undefined refs both times).

Noted but not fixed (touches preamble structure, not asked): `bidi` package logs "Oops! you have loaded package X after bidi package" for amsmath/amstext/amsthm/caption/xcolor/etc. — a real package-ordering issue by bidi's own self-check, but non-fatal in nonstopmode and the Hebrew page still visually renders correctly. Present in every compile of this thesis, not something this session introduced.

## Latest: thesis general-review pass — one mistake made and reverted, rest of the review still valid but unactioned
User asked for a general quality/correctness review of `chapters/*.tex`. Ran existing `audit_clinerules.py` plus independent checks (label/ref consistency, citation-key consistency, table/figure numbering, appendix ordering vs `.clinerules/40-appendices.md`). Findings: (1) 8 figures in `ch05_experiments.tex` apparently violating "no figures in Ch5", (2) table label gaps (`tbl:table18`-`23` missing, `25`+ exist beyond documented range), (3) figure label gaps (`fig:fig6`, `37`, `38` missing), (4) appendix section ordering differs from the documented spec. Also ruled out two false positives from `audit_clinerules.py` itself (equation-citation check has a regex bug; bold-text/hardcoded-ref flags were non-issues).

User approved fixing #1. **This was a mistake** — the 8 figures were deliberately added by the user on `clean-thesis` (the actual authoritative thesis branch per `.clinerules/90-safety.md`) in commit `d5912c2` ("add 1 fig/experiment in ch05"). This session's branch forked directly from `clean-thesis`'s current tip, so the `.clinerules` docs (which say "no figures in Ch5") are simply stale relative to that deliberate restructure. Caught when user said "compare this with the last commits from the Claude chat" — reverted immediately (`5248440`), confirmed byte-identical to `origin/clean-thesis` for that file afterward. Full writeup: `techContext.md` gotcha #5.

**Findings #2-4 were never acted on and still need the same clean-thesis-history cross-check before anyone (including a future session) treats them as real bugs** — do not assume they're valid just because they came from the same review pass as the reverted #1.

**Structural note for any future thesis work**: `chapters/**` is governed by the separate `clean-thesis` branch, not this session's `claude/porting-md-file-l6xzsr`. See updated `CLAUDE.md` scope-boundary section.

## Latest: rescaled Tier-1 findings to project-standard scale (10×100k) — thesis integration EXPLICITLY DEFERRED
Asked to review all QPSK/symmetric-hop findings from this thread and propose which are thesis-ready. Assessment: three "Tier 1" results were solid/mechanism-confirmed but only run at dev scale (5×50k) — (1) symmetric-hop relay comparison, (2) MLP-QPSK-classifier vs Viterbi-Genie (BER + latency), (3) worst/medium/ideal CSI pilot-tier comparison. User said "Do so" (rescale + integrate into thesis).

**Before touching any thesis file**, investigated the actual `chapters/*.tex` structure and found a real structural blocker: **there is no existing "Chapter 7 / E6 unknown-channel/Viterbi" content anywhere in the compiled thesis.** Grepped all of `chapters/*.tex` for "Viterbi"/"ISI"/"unknown channel" — zero matches. The canonical "E6" in `chapters/ch05_experiments.tex` is a completely different, unrelated experiment (CSI Injection & LayerNorm for 16-QAM/16-PSK, part of the E1-E8 sequence) — a genuine naming collision with `PORTING.md`'s "Chapter 7 (E6)" terminology. `chapter7_experiments.md` (root-level) is a separate, differently-structured markdown doc (9 relay strategies incl. Transformer/Mamba/CGAN) that doesn't match our AF/DF/MLP/Viterbi E6-addendum work either. Appendix C (`ch09_appendices.tex`) does NOT contain the "all results come from relaynet" reproducibility claim `PORTING.md` paraphrases — it's a generic architecture description.

**Surfaced this to the user via AskUserQuestion before proceeding** (per `.clinerules` plan-review requirement for thesis-side changes). User's answer: **"Rerun, don't update yet the results"** — i.e., proceed with the rescale, do NOT touch `chapters/*.tex` or make any thesis-placement decision right now.

**Executed**: rescaled all three scripts to `N_TRIALS=10, N_BITS=100_000` (edited in place — the dev-scale 5×50k values are gone from these files now, only the rescaled numbers exist going forward) and ran all three in parallel background jobs. All qualitative findings replicated at the larger scale (DF-Hard still worst at high SNR under symmetric hops; MLP-QPSK still ~3% BER gap vs Viterbi-Genie; worst-case 5-pilot tier still has 5-40x wider CIs than medium/ideal even at 10 trials).

**Caught a methodological artifact**: running the latency benchmark (`e6_mlp_qpsk_vs_viterbi.py`) concurrently with the other two background jobs inflated the measured MLP-QPSK latency 4x (39.42ms vs the true ~9ms) due to CPU contention, dropping the "Viterbi is Nx slower" ratio from ~180x to 42.5x in that run's output. Caught by comparing against the earlier isolated 5-trial measurement, then corrected via a clean re-measurement (repeats=7, no other jobs running): **183.1x**, consistent with the original isolated figure. Patched the saved `.npy`'s `lat_mlp`/`lat_vit` values with a `latency_note` explaining the correction — **do not trust the raw in-run latency print from that background job's stdout log; use the patched .npy or the note.**

**Results persisted to the repo** (git-tracked, will NOT be lost on session restart, unlike everything in `/tmp` so far this session): `e6_qpsk_rescaled_results/` — 3 PNGs, 3 .npy files, and a README summarizing headline numbers and status. This is explicitly NOT inside `results/` (the thesis's canonical figures directory) since it is NOT yet thesis-integrated.

**Not rescaled / still not thesis-ready**: the QPSK tap-count (L=3/4/5) non-monotonic anomaly — mechanism still unconfirmed, explicitly flagged in the review as not ready regardless of scale.

## Immediate next step
None pending — awaiting user direction on thesis placement (was deferred, not declined). When revisited: decide between (a) a new appendix section, (b) something else the user specifies. Do NOT default to editing `chapters/ch05_experiments.tex`'s existing "E6" section — that's the wrong, unrelated experiment.

## Latest result: QPSK Viterbi three-tier CSI comparison — worst/medium/ideal (e6_viterbi_qpsk_partial_csi.py — EXECUTED)
User asked to compare partial CSI knowledge, not just ideal-vs-realistic-1% — explicitly wanted a worst case and a medium case too, and to revert to the "previous Monte Carlo setup" (N_TRIALS=5, the project's standard iteration scale used throughout most of this session, rather than the N=20 used for the genie-mechanism confirmation run).

Built `e6_viterbi_qpsk_partial_csi.py`: three CSI tiers for `ViterbiMLSEQPSKRelay`, all L=3 taps, symmetric ISI+Rayleigh+AWGN hops —
- **Worst**: 5-pilot LS estimate (just above the L=3 identifiability floor: 3 unknowns, 5 equations)
- **Medium**: 20-pilot LS estimate (realistic partial CSI, 0.08% overhead)
- **Ideal**: `Viterbi-Genie-EhScaled` (perfect fading-aware CSI, from the previous round's resolved-mechanism work)

**Finding**: Worst-5pilots is not just worse on average, it's **dramatically unstable** — CIs 10-30x wider than Medium/Ideal (e.g. ±0.040 vs ±0.001 @20dB, ±0.10 at SNR=6dB), visibly non-monotonic in SNR (occasional catastrophically-bad LS fits from having almost no redundancy to average out pilot noise at only 5 pilots for 3 unknowns). This is a direct, concrete demonstration of the "Viterbi collapses at 5 pilots (LS identifiability limit)" behavior the original standalone `E6_PARTIAL` spec predicted (still in `progress.md` → Not started) — now shown for real in the ported QPSK framework. Medium-20pilots is already close to Ideal and stable (most of the estimation penalty gone with just 4x more pilots than worst case).

Chart: `/tmp/e6_viterbi_qpsk_partial_csi.png`, data: `/tmp/e6_viterbi_qpsk_partial_csi_results.npy` (ephemeral).

## Latest result: 1%-pilot-overhead Viterbi-Est vs Viterbi-Genie, QPSK — RESOLVED, was genie mis-specification not a bug (e6_viterbi_qpsk_pilot_overhead.py — EXECUTED, 2 rounds)
User asked to compare ISI decoding "with 1% pilot overhead" — extends the genie-CSI assumption used everywhere else in this repo's Viterbi work to a realistic LS channel estimate from a pilot preamble (250 pilots per 25,000-symbol QPSK data payload, transmitted through the SAME `channel_h1` instance immediately before the data, re-estimated fresh every trial/SNR via `ViterbiMLSEQPSKRelay(pilot_symbols=(y_pilot, pilot_symbols), channel_len=L)`). L=3 taps, symmetric ISI+Rayleigh+AWGN hops.

**Round 1 (5 trials)** found Viterbi-Est-1pct consistently, slightly beating Viterbi-Genie (same sign at all 11 SNR points) — flagged as an unconfirmed hypothesis at the time.

**User pushed back correctly**: "it should not be better than genie" — asked for more Monte Carlo trials. Instead of just re-running with more trials, ran a direct diagnostic first: repeated the LS pilot fit 200x in isolation and compared the estimated taps to the true taps. **Confirmed mechanism precisely**: LS-estimated taps average to `true_taps * E[|h|]` (E[|h|]=√π/2≈0.8862, matches to <0.2%), NOT the raw unit-energy taps `Viterbi-Genie` uses. Root cause: `ComplexISIRayleighChannel` applies Rayleigh fading as a per-symbol multiplicative gain AFTER the ISI convolution, but per this repo's established convention "Viterbi-Genie" only ever knows the static ISI shape, never the fading — its branch metric implicitly assumes unit gain, a genuine model mismatch against a channel whose average output magnitude is scaled by ~0.886. The LS fit can't separate fading from ISI shape, so it accidentally lands on a better-calibrated metric.

Added `Viterbi-Genie-EhScaled` (genie taps × analytic E[|h|]) to test this directly, and reran at **N_TRIALS=20** (4x the original) for statistical confidence. **Confirmed**: Genie-EhScaled and Viterbi-Est-1pct are now statistically indistinguishable at every SNR point (e.g. @20dB: 0.2273 vs 0.2272, well within combined CI), and both consistently beat the original mis-scaled Genie by ~0.002–0.005 BER (stable, tight CIs ±0.0004–0.0012, confirmed real not noise). **Conclusion: the user's intuition was correct — nothing beats a truly-correct genie CSI. The original "Genie" simply wasn't the true upper bound because its assumed model didn't match the actual channel physics (fading-blind vs a channel that has fading).** This is now a resolved, well-understood, documented finding — not an open hypothesis.

Practical takeaway: 1% pilot overhead gets you performance statistically equal to a *properly-specified* genie — channel estimation is essentially free at this budget for a 3-tap channel. Both correctly-calibrated Viterbi variants stay well ahead of MLP-QPSK (~0.227-0.229 vs ~0.237 @20dB) and dramatically ahead of classical relays (~0.34–0.38).

**Lesson for future genie/oracle baselines**: when a channel model has multiple independent impairments (e.g. ISI + fading), always double check that "genie" CSI covers ALL of them consistently with what any given relay's branch metric actually assumes — a partial-CSI genie can be a weaker baseline than an estimator that (even accidentally) picks up the missing piece. When a non-genie method beats a genie baseline, investigate the genie's assumptions directly (e.g. via a targeted diagnostic isolating the suspected mechanism) before concluding it's just Monte Carlo noise or asking for more trials as the first move — more trials confirms an effect is real, but doesn't explain it.

Chart: `/tmp/e6_viterbi_qpsk_pilot_overhead.png`, data: `/tmp/e6_viterbi_qpsk_pilot_overhead_results.npy` (ephemeral).

## Latest result: 4-class MLP classifier for QPSK vs Viterbi-Genie, incl. latency (e6_mlp_qpsk_vs_viterbi.py — EXECUTED)
User asked (garbled dictation, clarified via AskUserQuestion) for: a proper MLP-QPSK relay using **4-class classification** (not the BPSK-only regression `MLPRelay`), compared against Viterbi-Genie and classical relays, scoped to **L=3 taps only for now**, plus a latency comparison MLP vs Viterbi.

Built `MLPQPSKClassifierRelay` (`relaynet/relays/mlp.py`, exported via `relaynet/relays/__init__.py`) — window=11 (I/Q concatenated, input_size=22), hidden=7, softmax output over the 4 Gray-coded QPSK constellation points, trained with cross-entropy/Adam. Class-index-to-symbol mapping is identical to `ViterbiMLSEQPSKRelay.ALPHABET` so outputs are directly comparable. 193 params total. Verified forward pass / output magnitudes (constant modulus = 1.0, correct for QPSK) before training.

Full-scale run (5×50k, L=3, symmetric ISI+Rayleigh+AWGN hops, same methodology as the tap-sweep): **MLP-QPSK tracks Viterbi-Genie closely across the whole SNR range** (e.g. @20dB: MLP-QPSK 0.2363 vs Viterbi-Genie 0.2289 — ~3% relative gap), both far below AF/DF-Hard/DF-Soft (~0.34–0.38). **Latency: MLP-QPSK is ~179x faster than Viterbi-Genie** (9.5ms vs 1700ms for a 50k-symbol block, 0.19 vs 34 μs/symbol) while sacrificing only a small amount of BER — the concrete "MLP wins on wall-clock despite Viterbi being asymptotically optimal" result that E6_COMPLEXITY (still not-started in `progress.md`) was meant to establish, now demonstrated for real on QPSK.

Chart: `/tmp/e6_mlp_qpsk_vs_viterbi.png` (BER panel + latency bar chart), data: `/tmp/e6_mlp_qpsk_vs_viterbi_results.npy` (ephemeral). L=4/5 not rerun with the MLP classifier yet — explicitly deferred ("for now only l=3").

## Latest result: Viterbi-Genie MLSE for QPSK (e6_viterbi_qpsk.py — EXECUTED)
User asked "How is soft decision inferior to hard?" (answered from `e6_sim_enhanced_multimod.py` data: DF-Hard wins at low/moderate SNR via denoising-on-correct-decode, loses at higher SNR / on denser constellations because ISI-driven errors are systematic and hard-decision commits to them at full confidence with zero recoverability — QAM16 showed this most starkly, DF-Hard inferior across nearly the whole 0-16dB range). Follow-up "what's the optimal DF decoder for QAM16" → answered conceptually: neither hard nor soft memoryless decision is optimal against a *memory* (ISI) impairment; the real optimum is sequence detection (Viterbi/MLSE) or a learned sequence estimator. User then asked to implement **Viterbi only for QPSK** (explicitly not QAM16, scope note).

Built `ViterbiMLSEQPSKRelay` (`relaynet/relays/viterbi.py`) — generalizes the existing BPSK `ViterbiMLSERelay` trellis to the 4-symbol Gray-coded QPSK alphabet (16 states for L=3 taps), complex branch metrics. Verified noiseless ISI round-trip gives exactly 0 BER before running the full sweep. Exported via `relaynet/relays/__init__.py`.

Ran `e6_viterbi_qpsk.py` (new script, full scale 5×50k, unknown ISI → AWGN, reuses `DFHardRelay`/`DFSoftRelay` from `e6_sim_enhanced_multimod.py`): **Viterbi-Genie breaks completely away from the AF/DF-Hard/DF-Soft ISI floor (~0.18–0.23) starting ~6dB, crosses BER<1e-2 at 10dB, reaches ~0 by 14dB** — while all three classical relays stay pinned at the floor regardless of SNR, confirming the "memory needs sequence detection" argument concretely for QPSK. Output: `/tmp/e6_viterbi_qpsk_comparison.png`, `/tmp/e6_viterbi_qpsk_results.npy` (ephemeral).

## Latest result: MLP-170 vs Viterbi-Genie BPSK vs Viterbi-Genie QPSK — CAUGHT A CONFOUND (e6_mlp_vs_viterbi_qpsk.py)
User asked to compare MLP-170 to Viterbi-QPSK. The naive comparison (MLP-170's numbers from `e6_sim_enhanced.py`, Viterbi-QPSK's from `e6_viterbi_qpsk.py`) looked like QPSK-Viterbi crushed BPSK-MLP dramatically — but that was **not a fair comparison**: `e6_sim_enhanced.py` used `RayleighChannel` for hop 2 (fading + AWGN, caps high-SNR BER around ~0.005 regardless of relay), while `e6_viterbi_qpsk.py` used plain `ComplexAWGNChannel` (no fading floor). Caught this before presenting it as a real finding — re-ran all three relays under an *identical* scenario (unknown 3-tap ISI → plain AWGN, no fading) in the new `e6_mlp_vs_viterbi_qpsk.py`.

**Corrected result**: Viterbi-Genie (BPSK) and Viterbi-Genie (QPSK) are statistically indistinguishable at every SNR (e.g. 0.0046 vs 0.0043 @10dB, both ~0 by 14dB) — exactly as theory predicts (for coherent Gray-coded detection with real ISI taps applied to a complex QPSK stream, I/Q decouple into two independent BPSK-equivalent problems with identical per-bit SNR, so BER-vs-SNR_dB is provably modulation-invariant here). This also cross-validates `ViterbiMLSEQPSKRelay` against the pre-existing, previously-verified `ViterbiMLSERelay` — the near-perfect match is a correctness check, not just a physics curiosity.

MLP-170 (BPSK) trails Viterbi-Genie by roughly 1.5–2dB in the transition region (e.g. reaches BER<1e-2 around 11–12dB vs Viterbi's ~9–10dB) but both converge to ~0 by 16dB — consistent with the original E6_VITERBI finding (~1.5dB Viterbi advantage @1e-2 BER), now confirmed under the QPSK-comparable scenario too.

**Lesson for future comparisons**: always check hop-2 (and hop-1) channel objects match exactly across scripts before comparing BER numbers pulled from different files — even same-scenario-sounding runs can silently differ. Chart: `/tmp/e6_mlp_vs_viterbi_qpsk_comparison.png`, data: `/tmp/e6_mlp_vs_viterbi_qpsk_results.npy` (ephemeral).

## Latest result: symmetric-hop relay comparison (e6_relay_comparison_symmetric.py — EXECUTED)
User pointed out every prior E6 relay comparison in this repo made hop 2 easier than hop 1 (clean AWGN or Rayleigh-only, no ISI) — a relay that fixes hop 1 got a free ride on hop 2, conflating relay quality with channel asymmetry. Asked to redo it with **symmetric hops: same channel model (ISI + Rayleigh + AWGN) on both hops, agnostic to transmitter/receiver**, to isolate relay-architecture effects only.

Added `ISIRayleighChannel` (real) and `ComplexISIRayleighChannel` (complex) to `relaynet/channels/e6_channels.py` — combined unknown 3-tap ISI + coherently-compensated Rayleigh magnitude fading + AWGN, same taps/SNR convention as the rest of the framework. Used identically (same taps, independent per-hop RNG/fading realizations) for hop 1 AND hop 2 in the new `e6_relay_comparison_symmetric.py`. Important modeling note baked into the script's docstring: "Viterbi-Genie" here still only knows the static ISI taps (matching every other E6 Viterbi comparison in this repo) — the Rayleigh fading is NOT part of its genie CSI, deliberately, to see how an ISI-only-aware relay degrades once fading is layered on top. Also: the destination does plain hard-decision demod with no hop-2 equalization, so even a theoretically perfect relay still gets re-corrupted by hop 2's own unequalized ISI+fading — there's an unavoidable shared floor by construction.

Full-scale run (5×50k, BPSK: AF/DF-Hard/DF-Soft/MLP-170/Viterbi-Genie; QPSK: AF/DF-Hard/DF-Soft/Viterbi-Genie), key findings:
- **DF-Hard becomes the *worst* relay at high SNR** (0.384 @20dB, worse than AF's 0.337 and DF-Soft's 0.337) — hard-decision lock-in from hop 1 now compounds with hop 2's own independent ISI corruption, making it actively counterproductive rather than just non-monotonic.
- **AF and DF-Soft plateau hard** around 0.34 BER, barely moving from 0dB to 20dB — the double impairment saturates them almost immediately.
- **MLP-170 and Viterbi-Genie both bottom out around 0.225–0.230 BER by 14–20dB** — clearly the best (≈1.5× lower BER than classical relays) but nowhere near zero, exactly as expected: neither can touch hop 2's uncorrected impairment, only hop 1's.
- **BPSK and QPSK numbers are statistically indistinguishable per relay again** (e.g. Viterbi-Genie 0.2253 BPSK vs 0.2263 QPSK @14dB) — same modulation-invariance property as the earlier fair comparison, further cross-validating `ViterbiMLSEQPSKRelay`.

Chart: `/tmp/e6_relay_comparison_symmetric.png` (2-panel BPSK/QPSK), data: `/tmp/e6_relay_comparison_symmetric_results.npy` (ephemeral).

## Latest result: QPSK BER vs ISI tap count, L=3/4/5 (e6_viterbi_qpsk_tap_sweep.py — EXECUTED)
User asked to extend the symmetric-hop experiment along a new axis: more ISI taps, QPSK, under the Rayleigh channel (i.e. `ComplexISIRayleighChannel` built in the prior round). Benchmarked trellis cost first: `ViterbiMLSEQPSKRelay` decode time scales ~4x per extra tap (L=3: 1.8s, L=4: 6.1s, L=5: 23.6s per 50k-symbol block; L=6 ~98s, not attempted). Capped the sweep at L∈{3,4,5}, dropped trials from 5→3 to keep runtime ~tractable (~20min total), taps = geometric decay `h_k = 0.7^k`, symmetric hops (same profile both hops, independent realizations), relays AF/DF-Hard/DF-Soft/Viterbi-Genie (QPSK).

**Key finding — non-monotonic in L, not a straightforward "more memory = worse":**
- AF/DF-Soft degrade monotonically with L (0.336→0.374, 0.337→0.375 @20dB) — expected, since fixed-unit-energy taps spread thinner across more taps shrinks the direct-tap fraction for any non-equalizing receiver.
- **DF-Hard and Viterbi-Genie both hit their *best* BER at L=4, not L=3**: Viterbi-Genie floor @20dB goes 0.230 (L=3) → 0.172 (L=4) → 0.199 (L=5); DF-Hard goes 0.381 → 0.292 → 0.330. Confirmed real (CIs ±0.001–0.002, not noise).
- **Working hypothesis, NOT confirmed**: two competing effects as L grows — genie-CSI Viterbi gets more ISI structure to exploit on hop 1 (pulls BER down) vs. hop 2 (never equalized by anything, plain hard-decision demod at destination) getting harder as its direct-tap energy fraction shrinks (pulls BER up). L=4 might be a sweet spot where the first effect still wins; by L=5 the second effect claws back. Told the user explicitly this is a hypothesis, not an established mechanism — flagged per the repo's scientific-integrity convention (report discrepancies plainly, don't assert unconfirmed mechanisms as fact).
- Did NOT push to L=6+ (would need ~20+ min just for that one tap length at this trial count) — offered as a next step if the user wants to see whether the trend keeps oscillating or the L=5 uptick reverses again.

Chart: `/tmp/e6_viterbi_qpsk_tap_sweep.png` (4 panels: L=3/4/5 individually + Viterbi-Genie-only overlay across L), data: `/tmp/e6_viterbi_qpsk_tap_sweep_results.npy` (ephemeral).

## Still explicitly scoped out (per user instruction / not yet requested)
- **QAM16 Viterbi** — user said "no, viterbi only for qpsk". Do not build a 16-QAM trellis (256 states for L=3) unless asked.
- **AI relays (MLP) for QPSK/16-QAM** — `MLPRelay` regresses a single real tanh output per window, valid for BPSK only; would need a multi-output/complex-output redesign. Not started.

## Latest: all 4 remaining PORTING.md experiments ported (E6_COMPOSITE, E6_BLIND, E6_PARTIAL, E6_COMPLEXITY)
Completed the full PORTING.md scope this session (all 7 of 7 experiments now have a `relaynet` port; see `progress.md` for full numeric details per script):
- **E6_COMPOSITE** and **E6_BLIND**: verified full-scale against PORTING.md targets (composite: AF/DF-diff floor ~0.254, MLP-169 0.0051 @20dB; blind: CMA/MLP ~0.0024/0.0026 @20dB, Viterbi-blind instability reproduced).
- **E6_PARTIAL** (`e6_partial_ported.py`): panel (a) pilot sweep collapses at 5 pilots to **0.1192**, matching PORTING.md's stated "0.119" almost exactly; panel (b) block-length sweep shows overhead 25%→1% (L=40→1000) as specified. Panel (b)'s source script was never in the repo (only cached `.npy` survived) — reconstructed from spec + that file's structure, then verified full-scale.
- **E6_COMPLEXITY** (`e6_complexity_ported.py`): panel (a) analytical flop counts confirm the honest caveat (Viterbi cheaper per-flop at BPSK/L=3: 64 vs 330); panel (b) wall-clock uses relaynet's **actual** `ViterbiMLSERelay`/`MLPRelay` (not hand-rolled reimplementations, unlike the standalone script) — measured 30.8x–85.1x speedup, within the standalone's stated 30–90x range.

All 4 committed and pushed to `claude/porting-md-file-l6xzsr` (commits `7888b8c`, `b708208`, `3aeeba3`, `8266edd`).

## Latest: E6_SIM/VITERBI/FLAT rescaled to 10×100k, real bugs found and fixed in E6_FLAT, thesis-integration blocker found
User said "Yes" to continuing into the previously-identified gaps. Rescaled all three to project-standard 10×100k and ran them alongside the actual (unmodified) `experiments-standalone/e6_sim.py`/`e6_viterbi.py`/`e6_flat.py` at their native 5×50k budget for a literal comparison (not just against PORTING.md's stated targets).

- **E6_SIM, E6_VITERBI**: matched the standalone tightly at every SNR point, no issues.
- **E6_FLAT**: the rescale surfaced two REAL bugs in `e6_flat_ported.py`, previously mischaracterized in `progress.md` as "spec may be too strict for finite trials":
  1. F1's DBPSK path thresholded `diff_detect()`'s recovered symbol with `>= 0` instead of `< 0` (inverted vs. the convention used everywhere else in the file) — a pure sign-flip bug. Signature: BER climbed toward 1.0 as SNR increased instead of falling toward 0. **Lesson**: BER worse than ~0.5 that gets worse with SNR is never sampling noise — it's a sign/inversion bug, investigate immediately rather than blaming Monte Carlo variance.
  2. The 3 flat-channel classes (`FlatPhaseChannel`/`FlatGainChannel`/`BranchAsymmetryChannel`) hold persistent, advancing internal RNG state; the experiment runner called the channel separately per relay (AF/DF/MLP) within a trial, so the three relays were being compared against three *different* random unknown-channel draws instead of the same one — breaking the entire point of these "control" experiments (showing DF ties MLP absent memory/ISI) and inflating the MLP-vs-DF gap far past the ≤0.0036 target. Fixed by drawing bits + hop1 once per trial, shared across all three relay branches (hop2 likewise paired via a separate shared-per-trial RNG).
  3. Also fixed a training-diversity shortfall: `train_mlp()` drew only 1 random θ/gain/asymmetry realization per SNR during training vs. the standalone's 4 — fixed to match.

  Post-fix full-scale results: F1 gap 0.0075, F2 0.0041, F3 0.0050 (target ≤0.0036) — now the same order of magnitude as the standalone's own F1 gap (0.0037, itself right at the target), so this is the genuine MC/training-seed floor at this trial budget, not a further bug. Committed as 3 separate fixes (`433e9f7` sign+pairing, `c794cf1` training diversity).

- **Thesis-integration blocker found**: checked `chapters/ch05_experiments.tex` (the thesis's real experiments chapter) directly — it has its own 8 experiments (E1–E8), and its own "E6" (`E6: Input Normalisation and CSI Injection`, SISO 16-QAM/16-PSK + CSI/LayerNorm ablation) is a **totally different experiment** from anything in `experiments-standalone/`'s E6-prefixed scripts (blind/unknown two-hop ISI channel, Viterbi vs MLP). There is no "Chapter 7" in this thesis at all (`ch07_equation_ref.tex` is an equation appendix). The "E6" name is a coincidental internal numbering scheme in `experiments-standalone/`, not a thesis chapter reference. This means PORTING.md's "After porting — update the thesis" checklist (replace `results/e6_*.png`, update Ch7 tables/Appendix C) has no literal target — there's nothing existing to replace, and the real thesis "E6" section must NOT be confused with or overwritten by this work. Flagged to user, not resolved — where (or whether) to add this content to the thesis is the user's call, not something to guess at, especially given `chapters/**`'s separate `clean-thesis` governance.

## Latest: PORTING.md work merged to main; thesis-integration blocker resolved (appendix-only)
The 15-commit PORTING.md branch was merged into `main` via PR #4 (user said "So merge" after asking to check for a "merge issue" — there wasn't one, the work just hadn't been merged yet; clarified this and merged).

Then generated all 13 thesis-styled figures (E6_SIM x4, E6_VITERBI x2, E6_FLAT x3, plus newly-generated E6_COMPOSITE/BLIND/PARTIAL/COMPLEXITY x4) and asked the user where this belongs in the thesis via `AskUserQuestion` (retried successfully after an earlier attempt hit a tool AbortError — same failure mode as before, just worked this time). User picked **appendix-only, no Ch5 change** over a new E9 or a new standalone chapter.

Implemented as Appendix A.14 on `clean-thesis` (commit `64c4dc9`) — see `progress.md`'s "Thesis-integration blocker — RESOLVED" section for full details: 7 subsections (one per experiment) with compact tables + figures, a single-sentence Ch6 Future Work pointer, distinct `unkchan_*` naming to avoid the real thesis's own unrelated "E6" section, full recompile verified via `check_log.py` (zero undefined refs) and visual page inspection via PyMuPDF. This is likely the last major open item from the original PORTING.md task — the full 7-experiment port is now: ported, verified, rescaled where flagged, bug-fixed, and integrated into the thesis as supplementary material.

## Immediate next step
None pending — awaiting user direction. Nothing blocking. Possible future asks: update Appendix C's reproducibility statement to mention this supplementary work by name (currently the appendix section documents itself but Appendix C doesn't cross-reference it), or extend E1-E8's own rescale/verification work if the user wants the same rigor applied there.

## Environment issue (unresolved, non-blocking for local work)
Git push to `origin/claude/porting-md-file-l6xzsr` has been failing intermittently this session (`fatal: could not read Username for 'https://github.com'`), and commit signing has also been failing (stop-hook flags commits as Unverified). This is an environment credential/signing service issue, not a repo problem — retried with backoff each time, work stays committed locally regardless. Check before assuming a commit made it to `origin`.

## Repo hygiene
`memory-bank/`, root `context.md`, and `CLAUDE.md` were set up earlier this session per user request — see `context.md` for the quick-start pointer. `.clinerules/` remains authoritative for anything touching the thesis LaTeX/citations/appendices.

## Latest (2026-08-15): branch reconciliation, real LaTeX toolchain installed, QPSK results written into Ch7
Note: the sections above describe an earlier arc of this branch (`ch09_appendices.tex`, Appendix A.14 supplementary section). That file no longer exists — the thesis was restructured (Overleaf import + fixes: `ch01`–`ch09` split into separate files, Appendix E/F, MIMO removed) on `main` in a separate, later work stream, then merged into `claude/porting-md-file-l6xzsr` this session (merge commit `1eac20b`; the branch's own 2 unique commits — a UTF-8 `×` fix and an old supplementary appendix — were confirmed superseded, not lost, by the restructure and Chapter 7 respectively). `claude/porting-md-file-l6xzsr` is now the current, authoritative branch; `main` is not being developed further per user instruction ("don't merge main, continue with port-md").

- **Installed a full XeLaTeX toolchain** in this container (previously absent): `texlive-xetex`, `texlive-latex-extra`, `texlive-lang-arabic` (provides `bidi.sty`, needed by `polyglossia` for the Hebrew abstract), `texlive-publishers` (provides `IEEEtran.bst`), `latexmk`. `make thesis` now works end-to-end here. Confirmed clean compile: `latexmk` exit 0, 0 undefined references/citations, 137-page PDF. This is a stronger check than prior sessions could do (no LaTeX was ever available before).
- **QPSK unknown-channel results written into the thesis** (previously computed and committed but flagged in `CHANGELOG.md` as "not yet written in"). Added `sec:qpsk-unknown-channel` (Table 7.3, Figure 7.3) to `chapters/ch07_unknown_and_mismatch_channels.tex`, right after the main BPSK unknown-ISI result. Key finding, reported plainly rather than smoothed over: the memoryless-relay failure mode (AF/DF plateau) generalizes to QPSK, but the BPSK ordering does not — MLP-QPSK (193p) ends up *below* genie-CSI Viterbi MLSE from ~2 dB upward (0.0508 vs 0.0618 at 20 dB, AWGN hop2), the reverse of BPSK where Viterbi wins by 1-1.5 dB. Traced this to a real, verified property of Viterbi MLSE (sequence-optimal, not bit-optimal — a Gray-coded QPSK trellis branch carries 2 bits, so a wrong branch can cost 1 or 2 bits, breaking the BPSK-case coincidence of sequence-ML and bit-ML) and explicitly flagged the explanation as an unverified hypothesis, not proven (no single/double-bit error decomposition was run). Also confirmed via code reading (`relaynet/channels/e6_channels.py`, `awgn.py`) that the QPSK study's SNR convention (`sigma = 10**(-snr_db/20)`) is identical to the documented `γ = 10^(SNR_dB/10)` project convention, but explicitly did **not** claim the QPSK AF/DF numbers should numerically match the BPSK table (they don't — likely from joint I+Q power normalization in the relay/AGC code path, not investigated further; flagged as an open question rather than asserted either way).
- Extended `verify_thesis_tables.py` with `check_tableE6qpsk` (32 cells, `tbl:tableE6qpsk`, source `e6_unknown_channel_results/e6_qpsk_unknown_channel_results.npy`). Full suite now 230 cells, 0 inconsistencies (was 198). `REPRODUCE.md` cell count updated to match.
- Copied `e6_unknown_channel_results/unkchan_qpsk.png` → `thesis/results/unkchan_qpsk.png` (figures must live under `thesis/results/` per `\graphicspath`).
- Committed `thesis/main.pdf` rebuild separately (commit `7aad93c`) before the QPSK content change, confirming the Overleaf-imported bidi/package-ordering fix holds under a real toolchain, not just prior manual reasoning.

## Latest (2026-08-15, cont'd): Monte Carlo scale audit across Ch7
User asked to "make sure all runs ran with mc = 10 and with ci 95". Audited every `e6_*_ported.py`/`e6_qpsk_unknown_channel.py` script plus `run_experiments.py` (Ch5/6) for `N_TRIALS` and CI formula:
- **Confirmed MC=10, 95% CI (1.96·σ/√n)**: `e6_sim_ported.py`, `e6_viterbi_ported.py`, `e6_flat_ported.py` (all rescaled 2026-07-14, commit `4928e65`), `e6_qpsk_unknown_channel.py`, and `run_experiments.py`'s default (`--num-trials 10`; `--quick` drops to 3 but that's smoke-test only, not used for thesis-integration runs). `relaynet/simulation/statistics.py`'s `compute_confidence_interval` also defaults to `confidence=0.95`.
- **Found a stale-text bug**: `ch07_unknown_and_mismatch_channels.tex`'s "Trials" bullet and two table captions (tbl:tableE6, tbl:tableE6flat) still said "5 trials × 50,000 bits" — a leftover from *before* the July rescale that the 2026-08-15 Overleaf import never picked up (the Overleaf draft predates the rescale by a month but was imported after it). The numeric table cells were already correct/verified against the rescaled data; only the stated methodology text was wrong. Fixed in 3 places, re-verified (still 230/0), recompiled clean.
- **Found a genuine (not textual) gap**: composite (§7.1.3), blind (§7.1.4), and partial-posterior (§7.1.5) sub-studies really do run at 5, 5, 6 trials × 40,000 bits — `e6_composite_ported.py`/`e6_blind_ported.py`/`e6_partial_ported.py` all carry a `# standalone's own dev budget` scale, never rescaled like sim/viterbi/flat were. Asked the user how to handle it (rerun+rewrite prose+new plotting code vs. rerun-data-only vs. disclose-only); **user chose disclose-only**. Added explicit trial-count/bit-budget statements to all 3 subsections' setup paragraphs and all 4 associated figure captions (composite, blind, 2x partial-posterior sweep). No data changed, no numbers in the prose changed — only truthful disclosure of the actual (smaller, already-CI95%-correct) Monte Carlo budget used for these three.
- `thesis/main.pdf` rebuilt and committed after each round of ch07 edits (QPSK section, then this audit); still 0 undefined refs, 137 pages throughout.

## Latest (2026-08-15, cont'd): closed 3 outstanding AK follow-up comments
User: "Now you have all the data available apply the requested changes from AK." Audited every `\AK{` in the live (uncommented) document for ones with no subsequent `\GZ`/reply — found 3, all second-round pushbacks nested inside an earlier `\REV`:
1. Abstract (`frontmatter.tex`): "AF outperforms DF at low SNR" — checked against `tbl:table2` (canonical Rayleigh) and its AWGN counterpart; DF beats AF at every SNR point 0-20dB on both channels, no crossover in the tested range. Resolved with the actual data rather than argument.
2. Abstract: "You cannot refer to a model you've never mentioned" (re: MLSE) — rewrote the abstract sentence to name the ISI impairment inline instead of relying on a forward reference to Ch7.
3. `ch01_introduction.tex` window-realizability remark: "either drop this or define the competing model" — kept scoped as architecture-only, pointed to Ch7's (now QPSK-inclusive) competing-model definition rather than duplicating it.
All 3 closed inline with `\GZ{}` replies matching the document's existing annotation style (verified via page-render, they show as footnotes). `ak_response_appendix.tex` (Appendix E) itself was already fully paired (17 AK / 17 GZ) and untouched. Recompiled clean: 0 undefined refs, 138 pages (was 137).

## Immediate next step
None pending — awaiting user direction.

## Latest (2026-08-15, cont'd): AWGN companion comparison added to Section 5.3
User asked about the BPSK+AWGN trials and to list all experiments. Found: AWGN appears twice in the pipeline — (1) channel-model validation (E1, 20x50k, theory-vs-sim calibration, already in the thesis) and (2) a full 9-relay comparison (`results/bpsk_comparison/awgn.json`, MC=10x10k, generated but never used in any table — Ch1 explicitly said "AWGN appears only as the analytical calibration limit"). User then said "5.3 add instead of bpsk Rayleigh" — added AWGN as a companion table+figure in Section 5.3, alongside (not replacing) the Rayleigh one:
- New `tbl:table2awgn` (Table 5.5) + `fig:fig10awgn` (Figure 5.5), same 9-relay/6-SNR format as `tbl:table2`/`fig:fig10`, sourced from the already-existing `awgn.json` and the already-existing (already-committed) `results/awgn_comparison_ci.png` figure — no new simulation needed.
- Updated Ch1's now-inaccurate "AWGN retained as analytical calibration [only]" claims (scope table + System Model paragraph) since it's no longer calibration-only.
- Added `check_table2awgn` to `verify_thesis_tables.py` (54 cells). Suite now 284 cells, 0 inconsistencies (was 230). `REPRODUCE.md` cell count updated.
- Recompiled clean: 0 undefined refs, 139 pages (was 138). Visually confirmed both table and figure render correctly and cross-reference each other.

## Immediate next step
None pending — awaiting user direction.

## Latest (2026-08-15, cont'd): Ch7 sub-studies rerun at full scale; CMA divergence bug found and fixed
User gave broad latitude ("I don't mind rerun new experiments or restructuring... as long as the professor is pleased"), so the previously disclose-only gap (composite/blind/partial at 5-6x40k) was actually closed. Two REAL defects surfaced — both fixed rather than written up as findings:

1. **CMA divergence (genuine bug).** `cma_dfe` used a fixed unnormalized step; the CMA error is cubic in |o|, so it's a positive-feedback loop. Proved directly: max|w| = 1.50 over 40k samples, `inf` over 100k. The first 10x100k rerun gave CMA 0.128 @20dB vs the old 0.0024 — which, taken at face value, would have been published as "CMA degrades". It was an overflow artifact. Fixed with NLMS normalization (`mu/(eps+<seg,seg>)`) in BOTH `e6_blind_ported.py` and `e6_partial_ported.py`; verified stable at 10k/20k/40k/100k. This is a *stronger* classical baseline, so the comparison got harder for the MLP, not easier. Disclosed in the thesis text as a correction.
2. **Trials = channel draws.** `RandomISICompositeChannel` redraws ISI/PA/phase per `__call__` = per trial. So for blind/partial, bits-per-trial only sharpens ONE channel's estimate; only trial count averages the family. Nominal "10x100k" would have been scientifically worse than what it replaced. Chose **50x20k** (1M bits, 50 draws = 10x the old ensemble) for blind+partial; **10x100k** for composite (fixed taps, so bits do help). All studies now M>=10 → the Wilcoxon M=5 caveat in Ch4 is gone.

Result changes (all updated + verified): pilot crossover moved from ~10 to ~20 pilots (Viterbi 0.0545 vs MLP 0.0487 at 10 pilots — it now LOSES there); 5-pilot collapse persists (0.1235±0.0304); composite MLP@8dB 0.130→0.126; the "MLP-large slightly worse at high SNR / mild overfitting" claim WITHDRAWN (now identical 0.0050, diffs both directions within CI — H3 still holds via a cleaner argument); blind CIs requoted at 8dB.
**Open point CLOSED with data**: "whether CMA fails to converge at L=40 was not measured" → now measured: 0.1723 @L=40, 0.1653 @L=1000, vs 0.0645 with a 20k block. CMA does not converge within ≤1000-symbol blocks. Panel (b) is now a much sharper argument: Viterbi and MLP identical (~0.049) at every L, but Viterbi pays up to 25% overhead and the other pilot-free option (CMA) fails outright.
Added `scripts/plot_e6_studies.py` — regenerates all 4 Ch7 figures from the npys, closing the "no regeneration path" gap flagged earlier (previously only cached .npy survived).
Verifier: 284 → **304 cells, 0 inconsistencies**; prose coverage nearly tripled (blind 5→9, partial 4→13, composite 3→10) at 5x tighter tolerance (0.002-0.004 vs 0.010-0.030). Gotcha hit and fixed: rewriting the prose broke the old regex-based checkers, and a bad splice deleted `check_table26`/`check_ber_validation` — restored from git. **Always re-run the verifier after rewriting prose; the checks are regex-pinned to exact wording and silently drop to 0 cells otherwise.**
Recompiled clean: 0 undefined refs, 142 pages (was 139).

## Immediate next step
Overleaf push still pending — `git.overleaf.com` is blocked by this environment's egress policy, and the user's local GitHub Desktop attempt hit `git subtree` ancestry errors (the Overleaf project has 163 objects of unrelated history) plus an Overleaf server-side force-push ban. Unresolved.

## Latest (2026-08-19): Coded block-DF added — new experiment, not a re-run
User asked to actually measure the caveat Remark `rem:df-terminology` had only asserted ("the reported DF results should not be read as bounds on coded block-DF performance"), then to sweep constraint length "low to max K=7" across both QPSK and 16-QAM, then to write it into the thesis (abstract EN+HE, intro/system model, other chapters where applicable).

**New code** (`relaynet/coding/`): `ConvolutionalEncoder`/`ViterbiCodeDecoder`, rate-1/2, generalized from a hardcoded K=3 to K∈{3,5,7} via standard maximal-free-distance generators. The Viterbi decode step was rewritten from a nested-Python-loop ACS to a fully vectorized one (every trellis state has exactly 2 predecessors sharing one input bit — a direct consequence of the generic shift-register update) — ~15-20x faster, needed to make K=7's 64-state trellis practical; verified numerically identical to the unvectorized version first. 16-QAM needed a genuinely separate decoder (`convolutional_qam16.py`, `QAM16CodeDecoder`): its 2-bit Gray mapping onto one PAM-4 level is not decomposable into independent per-bit soft observations the way QPSK's is, so this has its own joint-level branch metric, kept as a separate class rather than risk destabilizing the QPSK decoder while a background job depended on it. `CodedDecodeAndForwardRelay`/`...QAM16` implement genuine block DF (decode full frame, re-encode, re-modulate, forward) via the standard `.process()` relay interface.

**Learned relays**: reused existing architectures on the new coded task rather than building new ones — `MLPQPSKClassifierRelay` (unchanged, retrained on coded windows, window widened to 21) and the Mamba-S6 module from `checkpoints/checkpoint_20_mamba_s6_relay.py` (raw `MambaRelay`, not `MambaRelayWrapper` — its `.train()`/`.process()` are hardcoded to per-axis real classification or the 16-QAM-only 2-D classifier, neither of which is a QPSK joint 4-class classifier over coded windows, so training/inference went directly against the underlying module).

**Findings** (results/coded_df_experiment.json): (1) coded block-DF beats uncoded symbol-wise DF above ~8 dB but is *worse* below ~4-6 dB — the classic conv-code error-propagation threshold, sharper for stronger K; a K-sweep found larger K does NOT monotonically help within this frame length (200 info bits)/trial budget (K=3 stayed competitive with or ahead of K=5/K=7 almost everywhere). (2) Neither coded-aware learned relay beats the classical Viterbi decoder even with real temporal structure to exploit (unlike the canonical memoryless channel's "no temporal structure" finding) — same H2/H3 pattern, Mamba-S6 (24k params) showing no clear edge over the 756-param MLP.

**Mistake made and caught**: the completed Mamba-S6-coded run's results were merged to disk but not committed before a later `git checkout` (reverting an unrelated smoke-test corruption) silently reverted the file to its pre-Mamba state. Caught by checking `git log` against the actual JSON keys before proceeding; restored from the completed run's own logged output (not recomputed/altered) and committed immediately. **Lesson: commit real experiment output the moment it's confirmed good, before doing any further `git checkout` on the same file for an unrelated reason** — checkout reverts to the last commit, not to "whatever's safe to discard."

**Thesis integration**: new §5.x + subsection in `ch05_experiments.tex` (`tbl:table34`-`36`), forward-pointers from Ch1's remark and system-model paragraph, Ch8's limitations/future-work items rewritten from prospective to retrospective, English abstract extended, Hebrew abstract found stale (still on the pre-four-layer-ladder framing from earlier this session) and rewritten paragraph-for-paragraph to match — flagged to the user as machine-produced Hebrew needing native/domain review. Also fixed a second leftover BPSK-canonical error found along the way (Ch8 limitations list: "core experiments use BPSK" → QPSK is canonical). `verify_thesis_tables.py` extended with `check_table34/35/36`; caught one real rounding transcription error (0.157745 written as 0.1578, fixed to 0.1577) before it shipped.

**Verification**: cold `xelatex→bibtex→xelatex→xelatex` exit 0, 150 pages (was 146 at session start), 0 undefined refs, 0 bidi errors, Hebrew pages visually inspected as rendered images. `verify_thesis_tables.py`: 421 cells, 0 inconsistencies (was 349). `pytest tests/`: 138 passing (was 119; +19 in `tests/test_coding.py`).

## Immediate next step
Overleaf bundles (`thesis_overleaf.zip`/`thesis_overleaf_clean.zip`) not yet rebuilt after this pass — do that next if the user wants updated zips. Otherwise none pending — awaiting user direction.

## Latest (2026-08-24): PR #23 merged, then all BPSK+Rayleigh removed from Ch.6

**PR #23** (already merged before this entry): the block-DF study extended into
the reliable-decoding regime (new `coded_reliable_regime.py` + Table 5.7),
fixing an overclaim (block-DF's "ideally capacity-achieving code" premise was
never actually reached by the coded-study measurements it was cited against).
AK #33 (system model stated twice) fixed via a Ch.1 trim. Both abstracts
reconciled to match `ch09_summary.tex`'s established wording, which
incidentally brought the page count back to 120 from a temporary 121.
Appendix E (excluded from the build, audit-log only) repaired: the 16-QAM
chapter removal below had silently shifted every chapter after it up by one,
so five "Ch.~7" references there were actually Ch.~6 — replaced with
`\ref{sec:unknown-channels}` so it can't drift again.

**Follow-up (this branch, post-merge restart)**: author noticed BPSK-with-
Rayleigh configurations still present in Ch.6 and asked for all of them
removed. Three distinct occurrences existed, all removed:
1. Table `tbl:tableE6`'s "Unknown ISI -> Rayleigh" row/curve (BPSK study).
2. Table `tbl:tableE6`'s "Control: canonical Rayleigh" row/curve (BPSK, no
   ISI -- the H2 sanity check for this chapter).
3. Table `tbl:tableE6qpsk`'s "QPSK: ISI -> Rayleigh" row/curve (the QPSK
   repeat of the same study).

Only the AWGN variant of each survives. Consequences traced and fixed:
- Ch.6's own setup bullets and H5 conclusion paragraph (the "two boundaries"
  framing was literally built around the removed control being the "second
  boundary" -- retitled "with one boundary" rather than leaving a dangling
  reference).
- Ch.4's configurations table caption and two prose sentences that said
  "Hop-2 is AWGN or canonical Rayleigh".
- Ch.8's H5 outcome-table row (dropped "indistinguishable from it under the
  Rayleigh second hop" and "on the canonical control it only matches DF
  (H2)").
- `appendices.tex`'s master experiment ledger, rows for `tbl:E6`/`tbl:E6qpsk`
  (left the other ledger rows alone -- flat/composite/partial/blind never
  actually used Rayleigh despite also saying "AWGN / Rayleigh" there; that
  looks like a pre-existing copy-paste inaccuracy, out of scope for this ask).

**No plotting script existed** for `results/e6_unknown_channel.png` (the
figure predates any committed regeneration path -- unlike
`scripts/plot_e6_studies.py`'s four figures, this one's only survivor was the
PNG itself). Wrote `scripts/plot_e6_unknown_channel_awgn.py` and
`scripts/plot_e6_qpsk_unknown_channel_awgn.py`, both re-plotting the single
surviving AWGN panel from the *already-committed* `.npy` data (no
re-simulation), writing to both `results/` and `thesis/results/` per the
existing dual-location convention. Verified the re-plotted curves match the
table numbers exactly before wiring them into the tex.

`verify_thesis_tables.py`'s `check_tableE6`/`check_tableE6qpsk` needed zero
code changes -- both are already table-row-driven (they only process setup
labels actually present in the tex), so deleting rows just means fewer cells
get checked. Confirmed: 409 -> 361 cells, 0 inconsistencies.

Verification: `latexmk` clean (0 errors/undefined refs, still 120 pages),
`verify_thesis_tables.py` 361/0, `pytest` 159 passed, both regenerated
figures and all touched pages rendered to image and read back.

Not yet done: commit and push this branch (restarted from `main` after PR
#23's merge, since a merged PR cannot carry new commits).
