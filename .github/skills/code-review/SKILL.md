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

**3. Committed results are Monte Carlo outputs, not regenerable artifacts.**
Files under `results/` were produced by seeded but expensive runs, and the
thesis quotes them. Regenerating one re-rolls its randomness and silently
changes published numbers. When a *generator* has a bug, the right fix is
usually to correct the generator and patch the committed artifact in place,
verifying that nothing except the intended value moved — not to re-run.

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
- Anything that would change the thesis page count. It is capped at 120.

## Conventions that are load-bearing

- **SNR convention:** `γ = 10^(SNR_dB/10)`, and the SNR axis denotes `E_s/N_0`
  throughout. QPSK carries 2 bits/symbol, so `E_b/N_0` is 3.01 dB below the
  stated value. Mixing the two silently invalidates every comparison.
- **Relay interface:** relays expose `.process(received_signal)`; channels are
  callables `channel(signal, snr_db)`. New components should follow these
  rather than inventing conventions.
- **Build invariants:** a cold `xelatex → bibtex → xelatex → xelatex` must
  produce 0 errors, 0 undefined references, 0 undefined citations and 0 bidi
  errors. The thesis contains a Hebrew abstract, so RTL rendering must be
  checked visually after any change near it.

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
- **Hypotheses are H1–H4 and H6. There is no H5** — it was withdrawn together
  with the Mamba-S6 training-time study.
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
- **`main` is the authoritative branch.** `.clinerules/90-safety.md` still
  names `clean-thesis` in its "Git Safety" section; that branch is stalled and
  `main` contains everything it has. See `memory-bank/activeContext.md`.

## Low value here

`memory-bank/**` and `thesis/RERUN_CHANGELOG.md` are working notes and an
append-only audit log, not deliverables. Factual errors in them are worth
flagging; wording, formatting and line-length preferences are not.
