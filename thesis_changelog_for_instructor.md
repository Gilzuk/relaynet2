# Changelog — Revisions in Response to Instructor Comments

**Author:** Gil Zukerman
**Branch:** `clean-thesis`
**Tag of the version reviewed by the instructor:** `thesis-draft-release-1.0`
**Scope of this revision:** Chapter 1 (Introduction) and Chapter 4 §4.1 (System Model). No changes to results, figures, or experiments.

---

## Summary

The four comments addressed:

1. *"You jump between different relay settings."*
2. *"Information-theoretic results assume a line of sight; without LOS the info-theoretic problem is trivial because capacity is not sensitive to delay."*
3. *"You introduce SISO with fixed gains, then complex-valued MIMO fast-fading Rayleigh, then the DMT (only relevant for slow-fading). Concentrate on the model you actually studied."*
4. *"The time index is suppressed yet you apply a relaying function. Explain half-duplex and that, w.l.o.g., the relay operation can be split into a listening phase and a transmission phase."*

The revisions are clarification/framing edits only — no experiments, code, or results changed. The Hebrew abstract and bibliography are untouched. Citation audit remains clean (31 cited keys, all defined; 0 undefined references).

---

## Per-comment changelog

### Comment 1 — A single, explicit "model studied" anchor
**Where:** Chapter 1, immediately after the chapter opening paragraph.
**What changed:** Added a boxed paragraph titled **"System model studied in this thesis"** that fixes the model up-front, before any supporting theory:
- half-duplex, two-hop, single relay, **no direct source–destination path**;
- **Hop 1 (S→R) is SISO** (AWGN / Rayleigh / Rician);
- **Hop 2 (R→D) is 2×2 MIMO with i.i.d. Rayleigh fast fading** (a fresh realization per channel use), equalized at the destination (ZF / MMSE / SIC);
- primary modulation: BPSK; reported metric: **uncoded BER vs. SNR**;
- everything else in the chapter (relay-channel capacity, MIMO ergodic capacity, DMT) is now explicitly labelled *theoretical context and bounds*, not the operating regime.
- The full system model is restated in §4.1 with a back-reference, so the same model statement appears in two places.

### Comment 2 — Info-theoretic results re-framed as Gaussian/LOS context
**Where:** Chapter 1, §1.1.1 (now subtitled **"Information-Theoretic Foundations (Background)"**).
**What changed:** Added an opening italic paragraph that states explicitly:
- the Cover–El Gamal cut-set bound, the DF achievability bound, and the Gaussian DF capacity result are presented as **classical theoretical context for the Gaussian (LOS, delay-sensitive) relay channel**;
- they motivate decode-and-forward as a baseline;
- the channel actually simulated in this thesis is the NLOS, fast-fading regime defined in the system-model statement, where the relay-channel capacity problem is not delay-sensitive;
- the metric reported throughout the thesis is **uncoded BER, not ergodic capacity**.

### Comment 3 — DMT demoted to slow-fading background; SISO vs. MIMO disambiguated
**Where:** Chapter 1, §1.4 (MIMO Systems and Equalization), in the paragraph that introduces the diversity–multiplexing tradeoff.
**What changed:**
- The DMT statement is now explicitly framed as a **slow-fading / outage** concept and is mentioned only as background that motivates why the choice of equalizer matters.
- A sentence was added immediately after, stating that the Hop-2 channel evaluated in this thesis is, by contrast, **2×2 i.i.d. Rayleigh *fast* fading** (a fresh realization per channel use), and that the metric reported is **uncoded BER as a function of SNR averaged over the fading ensemble — not outage and not the DMT**.
- Across Chapter 1 and §4.1, the SISO description is now consistently labelled as **Hop 1** and the complex 2×2 fast-fading description as **Hop 2**, removing the previous ambiguity between the two.

### Comment 4 — Time index restored; half-duplex listening and transmission phases
**Where:** Chapter 1, §1.2 ("Two-Hop Relay Model", retitled **"Two-Hop Relay Model (Half-Duplex; Listening and Transmission Phases)"**) and Chapter 4 §4.1.
**What changed:**
- The two channel equations are rewritten with the **explicit discrete-time index** $i$:
  $y_R[i] = x[i] + n_1[i]$ (listening phase, Hop 1),
  $y_D[i] = x_R[i] + n_2[i]$ (transmission phase, Hop 2);
  noise terms are now stated as i.i.d. across time, $n_k[i] \sim \mathcal{N}(0,\sigma^2)$.
- The relay function is written as a **time-indexed window operation**, $x_R[i] = f_\theta\!\left(y_R[i-w:i+w]\right)$, with $w$ defined as the half-window size ($w=0$ for the memoryless AF/DF baselines, $w>0$ for the AI relays).
- A short paragraph defines **half-duplex** ("the relay does not receive and transmit on the same time/frequency resource") and states that, *without loss of generality*, the relay operation splits in time into a **listening phase** (collect $y_R[\cdot]$) and a **transmission phase** (emit $x_R[\cdot]$). This is consistent with the Slot 1 / Slot 2 description that was already in the draft; the phrasing is now explicit.
- The same time-indexed equations and listening/transmission-phase wording are mirrored in the System-Model paragraph of Chapter 4 §4.1 (System Model and Hop Model), and in the paragraph describing the relay's neural network.

---

## Files touched

| File | Sections affected |
|------|-------------------|
| `chapters/ch01_introduction.tex` | Chapter opening (added "System model studied in this thesis" box); §1.1.1 (renamed → "Information-Theoretic Foundations (Background)", added LOS-context disclaimer); §1.2 (retitled, time index restored, listening/transmission phases); §1.4 (DMT framed as slow-fading background, fast-fading uncoded-BER metric clarified). |
| `chapters/ch04_methods.tex` | §4.1 System Model intro (back-reference to §1.2, half-duplex + listening/transmission phases, time-indexed equations); "Hop Model" paragraph (time-indexed equations); relay neural-network paragraph (time-indexed Hop 1 equation). |

No other files were edited. `references.bib` is unchanged.

---

## What did *not* change

- All experimental results, figures, tables, and the entire Chapter 5 (Experiments) and Chapter 6 (Discussion) are unchanged.
- All citations: 31 cited keys, all defined in `references.bib`; the compilation reports **0 undefined references**.
- Page count and structure of the thesis are essentially unchanged (the revised paragraphs added approximately 1.9 KB to ch04 and a comparable amount to ch01).

---

## Verification

- Compilation: `latexmk -xelatex thesis_tau.tex` succeeds; `\nUndefined References: None` (verified by `check_log.py`).
- Citation audit (`audit_citations.py`): 31/31 cited keys defined in `references.bib`; 0 missing; 0 uncited bib entries.
- Git tag of the *previous* draft (the version the instructor reviewed): `thesis-draft-release-1.0`.
- A new tag for this revision is created on commit; the diff between the two tags shows changes confined to `ch01_introduction.tex` and `ch04_methods.tex` plus the helper script.