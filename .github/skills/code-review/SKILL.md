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
- Anything that would change the thesis page count. The cap is 120, set by the
  author. Check the count against an actual build rather than against any
  figure quoted in the repository: `thesis/CHANGELOG.md`,
  `memory-bank/progress.md` and `memory-bank/activeContext.md` all record
  earlier builds at 146 and 149 pages, from before the 151→120 reduction.

## Conventions that are load-bearing

- **SNR convention:** `γ = 10^(SNR_dB/10)`. The `snr_db` passed to the channel
  functions is always `E_s/N_0` — total symbol energy over noise, computed from
  the constellation's unit average power. This is documented in
  `relaynet/modulation/qpsk.py`. The per-bit axis then depends on the
  constellation: `E_b/N_0 = E_s/N_0 − 10·log₁₀(k)`, so it is the same number
  for BPSK (`k=1`) and 3.01 dB lower for QPSK (`k=2`).
  Two places in the repository label `snr_db` as `E_b/N_0`:
  `tests/test_snr_convention.py` (docstring) and the comment at
  `relaynet/channels/awgn.py:31`. Both are correct for their own scope — each
  is BPSK-only, and at `k=1` the two quantities coincide, so neither is a
  contradiction of the `E_s/N_0` definition above. Do not generalise that
  label to QPSK. Comparing a measured QPSK BER against a closed form without
  applying the 3.01 dB conversion is a real error, and the thesis applies it
  explicitly wherever such a comparison is made.
- **Relay interface:** relays expose `.process(received_signal)`; channels are
  callables `channel(signal, snr_db)`. New components should follow these
  rather than inventing conventions.
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
  `RERUN_CHANGELOG.md` and the `_`-prefixed chapter files is deliberate —
  those are the historical record of the withdrawal and are not built, so do
  not flag them as inconsistent.
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
- **`main` is the authoritative branch.** Two places still say otherwise and
  neither is current: `.clinerules/90-safety.md` names `clean-thesis` in its
  "Git Safety" section, and older entries in `memory-bank/activeContext.md`
  call `clean-thesis` "the actual authoritative thesis branch". That branch
  has been stalled since 2026-07-18 and `main` contains everything it has. The
  current-state header at the top of `activeContext.md` is authoritative over
  both.

## Low value here

`memory-bank/**` and `thesis/RERUN_CHANGELOG.md` are working notes and an
append-only audit log, not deliverables. Factual errors in them are worth
flagging; wording, formatting and line-length preferences are not.
