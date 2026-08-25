---
name: code-review
description: Repository context for reviewing relaynet2 — an M.Sc. thesis and the simulation framework that produces its numbers. Explains which defects matter here, which conventions are load-bearing, and which settled decisions not to re-litigate.
---

# Reviewing relaynet2

This repository is an M.Sc. thesis ("Deep Learning Architectures for Two-Hop
Relay Communication") plus `relaynet`, the simulation framework that produces
its numbers. The deliverable is `thesis/main.pdf`. The code exists to generate
the tables and figures in it, so a defect that makes the thesis assert
something untrue is more serious here than an ordinary code smell.

## Hard rules

**1. Numerical results are scientific claims.** Never propose changing a
reported BER, goodput, latency or capacity value, a table cell, or a figure.
If a number looks wrong, say so and explain why — but do not suggest a
replacement value. Corrected numbers may only come from re-running the
measurement that produced them.

**2. `verify_thesis_tables.py` is the source-of-truth gate.** It re-derives
every numerical table in the thesis from its committed result file and flags
mismatches. A change that edits a table without updating the verifier, or
retires a table without retiring its check, is a defect worth flagging.

**3. Treat `results/` files by what kind of defect they have.** The thesis
quotes these files, so how they may be changed depends on what is wrong with
them. Three cases, and conflating them is itself worth flagging:

- *Encoding or representation* (a `NaN` where JSON requires `null`, a
  formatting problem): patch the committed file in place, and verify that no
  numeric value moved.
- *A deterministic re-derivation* — a script that recomputes from already-
  measured data with no randomness, such as the latency-budget envelope: fix
  the script and re-run it. Confirm determinism first by reproducing the
  committed output from the unfixed script; then any diff is attributable to
  the fix.
- *A Monte Carlo measurement*: never hand-edit it, and do not regenerate it
  casually — a re-run re-rolls the randomness and changes what the thesis
  reports. Correcting such a value is a decision for the author, not a
  review suggestion. This is rule 1 restated at the file level.

## What is worth flagging

- Correctness bugs in `relaynet/**`, the `coded_*.py` and `e6_*.py` drivers,
  `verify_thesis_tables.py`, and `scripts/**`.
- **Claims in the thesis that point at content which no longer exists.** This
  repository has had repeated defects of exactly this kind after material was
  cut: cross-references to deleted sections, hypothesis ranges that include a
  withdrawn hypothesis, captions describing removed figures, conclusions
  citing removed studies. Checking that every claim has something behind it is
  high-value here.
- Internal contradictions between two statements in the same document.
- LaTeX tables or figures overflowing the text block. An unbounded `l` column
  holding sentence-length text is the usual cause; prefer a wrapped `p{}`
  column with `\raggedright`. Long unbreakable `\texttt{}` paths overflow too —
  `\url{}` breaks at `/` and `_`.
- Anything that would change the thesis page count. **Report the change; do not
  treat exceeding any particular number as a defect on its own.** The document
  built at 120 pages after the 151→120 reduction, and that figure was once
  described here as a cap, but the author accepted growth to 125 on 2026-08-25
  when Chapter 2 gained its channel-coding background and Chapter 4 its
  per-axis paragraph. Each increase is the author's call, so the useful review
  action is to say what the count became and why. Check it against an actual
  build rather than against any figure quoted in the repository:
  `thesis/CHANGELOG.md`, `memory-bank/progress.md` and
  `memory-bank/activeContext.md` all record earlier builds at 146 and 149
  pages, from before the reduction.

## Conventions that are load-bearing

- **SNR convention.** One invariant holds everywhere: `γ = 10^(SNR_dB/10)` and
  the noise is set so that **`γ = E_s/N_0`, where `E_s` is the average energy
  of the signal actually handed to the channel**. Per-bit follows from the
  constellation: `E_b/N_0 = E_s/N_0 − 10·log₁₀(k)` — the same number for
  1 bit/symbol, 3.01 dB lower for QPSK.

  Many comments and test docstrings across `relaynet/channels/**`,
  `relaynet/modulation/**` and `tests/**` label `snr_db` as `E_b/N_0`. Do not
  read those as a competing convention, and do not "fix" them: each sits on a
  1-bit-per-symbol path where `E_s = E_b` makes the two numerically identical.
  The complex paths do not carry that label — `ComplexISIChannel` states
  `gamma = 1/sigma^2`, and `tests/test_channels.py` calls the real branch
  `Eb/N0` and the complex branch `Es/N0` in the same test, which is the
  clearest statement of the rule in the repository.

  Note that the label tracks the *code path*, not the file:
  `relaynet/channels/awgn.py` carries an `E_b/N_0` comment and also has a
  complex branch, where the same argument means `E_s/N_0`. So a file
  containing that label is not thereby a BPSK-only file.

  What is a real error: comparing a measured QPSK BER against a closed form
  without applying the 3.01 dB conversion. The thesis applies it explicitly
  wherever such a comparison is made. To check any individual label, read what
  the signal on that path carries rather than trusting the wording.
- **Relay interface:** relays expose `.process(received_signal)`; channels are
  callables `channel(signal, snr_db)`. New components should follow these
  rather than inventing conventions.
- **The committed PDF must not lag its sources.** `thesis/main.pdf` is the
  deliverable. A diff that changes `thesis/chapters/**`, `thesis/main.tex`,
  `references.bib` or an included figure **without** a rebuilt
  `thesis/main.pdf` in the same change is a real finding: the merged document
  will not contain the merged edit, and no test catches it. PR #25 shipped
  exactly this way — five chapters of edits against a PDF still at its
  pre-session 120-page build. Conversely, do not ask for a rebuild on
  Python-, `results/*.json`- or notes-only diffs; nothing re-renders.
- **Build invariants:** validate with `latexmk -xelatex`, **not** a hand-rolled
  pass sequence, and check the *final* pass — earlier passes always show
  unresolved references while the `.aux` settles. This is a standing rule in
  `memory-bank/techContext.md`: a forced manual four-pass run converges and
  looks clean even when `bidi` package-order errors are present, which is how
  a class of Overleaf-breaking failure once went unnoticed. A clean build is
  exit 0 with 0 errors, 0 bidi errors and no undefined references or citations
  in the final pass. The thesis contains a Hebrew abstract, so RTL rendering
  must also be checked visually after any change near it.

## Settled decisions — please do not re-flag

These are deliberate and were each decided explicitly. Re-raising them costs
review rounds without improving the thesis.

- **Scope.** The canonical *benchmark* is QPSK over i.i.d. Rayleigh fast
  fading, SISO on both hops, uncoded, half-duplex two-hop with no direct
  source–destination path, and it is held fixed wherever it is used, so that
  the relay function is the only variable. This does **not** mean non-Rayleigh
  content is out of scope: Chapter~7 deliberately departs from the canonical
  model — unknown 3-tap ISI, a composite impairment cascade, blind and
  partial-CSI regimes — and that departure is the thesis's principal
  contribution, not a scope violation.
- **The thesis poses H1–H5, and there is no H6.** The original H5, the
  Mamba-S6 training-time hypothesis, was withdrawn with that study in the
  151→120 page reduction. That left a gap at 5, so the unknown-channel
  hypothesis was renumbered H6→H5 into the empty slot. `H5` therefore now
  denotes the thesis's principal contribution, not the withdrawn
  training-time claim. Two consequences for review: a reference to `H6` in a
  built chapter is a real finding, and the old numbering surviving in
  `thesis/RERUN_CHANGELOG.md` and the `_`-prefixed files in
  `thesis/chapters/` is deliberate — those are the historical record of the
  withdrawal and are not built, so do not flag them as inconsistent.
- **There is no 16-QAM relay study.** 16-QAM survives only as one rung of the
  modulation-and-coding ladder in the link-adaptation study, where the object
  of study is the rate-and-modulation choice, not the relay.
- **The AWGN calibration study was removed** in the 151→120 page reduction.
  The simulator is now calibrated on the canonical channel itself, by checking
  symbol-wise DF against the closed-form two-hop composition (Table 2's
  "DF th." column). AWGN remains legitimately as the noise term of the channel
  model and in the closed-form background of Chapter~2 — those are not
  residue. An older entry in `memory-bank/activeContext.md` predates the
  removal and says the section "must stay"; it is marked SUPERSEDED in place,
  and the current-state header at the top of that file is authoritative over
  all such entries. If the compiled PDF ever refers to an AWGN calibration
  *study* again, that is a real finding worth raising.
- **The cGAN is implemented but excluded from every reported comparison**, on
  cost grounds. It is described in the methods chapter; it has no results
  column anywhere.
- **`main` is the authoritative branch.** `clean-thesis` has been stalled
  since 2026-07-18 and `main` contains everything it has plus a month of
  further work. `CLAUDE.md`, `.clinerules/90-safety.md`, and
  `.clinerules/00-general.md` were all corrected on 2026-08-25 to say so —
  none of them name `clean-thesis` as authoritative any more. Older entries in
  `memory-bank/activeContext.md` still call `clean-thesis` "the actual
  authoritative thesis branch"; those are historical journal entries about a
  past incident, explicitly marked superseded in place, not current guidance
  — do not flag them as a defect.

## Low value here

`memory-bank/**` and `thesis/RERUN_CHANGELOG.md` are working notes and an
append-only audit log, not deliverables. Factual errors in them are worth
flagging; wording, formatting and line-length preferences are not.
