# Instructor Feedback — Analysis, Plan, and Draft Reply

> Status: **PLANNING ONLY — no thesis files changed at this stage.**
> Scope of analysis: `chapters/ch01_introduction.tex`, `chapters/ch02_literature_review.tex`, `chapters/ch04_methods.tex`.

---

## 1. Review of the last thesis updates (what changed recently)

From `git log`, the most recent commits did **not** touch the conceptual system-model narrative the instructor is commenting on. They were housekeeping:

- `c7bc04f` / `6cbcb0d` — replaced inline `Source:[X]` annotations in **ch04** with proper `\cite{}` commands; removed the unused `sutton2018reinforcement` entry from `references.bib`. Citation audit is clean (31/31 cited, 0 undefined).
- `d5912c2` — appendix restructure (B = tables, C = TikZ), +1 figure/experiment in ch05.
- `5a6ba25` / `0ace044` / earlier — cover-page redesign, frontmatter, main.tex compilation fixes.

**Implication:** The instructor's comments target the *introduction / literature framing* (ch01 §1.1–1.3, §1.4 MIMO/DMT) and the *methods system model* (ch04 §4.1). None of these were modified recently, so the comments apply to the current text as-is.

---

## 2. What the actual studied model is (ground truth from the code/thesis)

Confirmed from ch04 §4.1 ("System Model") and ch01 §1.2 ("Two-Hop Relay Model"):

- **Topology:** Two-hop, single relay, **no direct source–destination link**, **half-duplex** (Slot 1 = S→R, Slot 2 = R→D). This is already stated explicitly in ch01 §1.2 and ch04 §4.1.
- **Hop 1:** SISO channel (AWGN / Rayleigh / Rician), real-valued BPSK as the primary case.
- **Hop 2:** 2×2 MIMO with **i.i.d. Rayleigh fast-fading** (\(H_{ij}\sim\mathcal{CN}(0,1)\) drawn per use), equalized at the destination (ZF / MMSE / SIC).
- **Relay function:** \(\hat{x}_R = f_\theta(\cdot)\) over a sliding window of received samples (denoising), then power-normalized and retransmitted.

So the **actually-simulated** model is a fast-fading, NLOS-dominant setup. The information-theoretic material (Cover–El Gamal, DF capacity) and the diversity–multiplexing tradeoff (Zheng–Tse) are presented as *background/framing*, not as the operating regime that was simulated. This mismatch is exactly what the instructor flagged.

---

## 3. Comment-by-comment analysis and plan

### Comment 1 — "You jump between different relay settings" (overarching)
**Where:** Spans ch01 (info-theoretic §1.1.1, two-hop §1.2, MIMO/DMT §1.4) and ch04 §4.1.
**Diagnosis:** Valid. The intro presents (a) the general info-theoretic relay channel, (b) a real SISO fixed-gain model, (c) complex 2×2 fast-fading MIMO, and (d) the slow-fading DMT — without a single, explicit "this is the model we study" anchor that the reader carries throughout.
**Plan (deferred — not done now):**
- Add a short, explicit **"System Model Studied in This Thesis"** statement early in ch01 (one paragraph) that pins down: half-duplex two-hop, no direct link, SISO Hop 1, 2×2 i.i.d. Rayleigh fast-fading Hop 2, BPSK primary. Everything else in the intro is then explicitly framed as *context* or *bounds*, not the operating regime.
- Add a one-line forward/back reference between ch01 §1.2 and ch04 §4.1 so the two model statements are visibly the same model.

### Comment 2 — "Info-theoretic results assume a line of sight; without LOS the info-theoretic problem is trivial because capacity is not sensitive to delay"
**Where:** ch01 §1.1.1 (Cover–El Gamal bounds, DF capacity, Gaussian DF capacity Eq. for equal per-hop SNR).
**Diagnosis:** Partly a framing issue. The instructor's point: the capacity/DF bounds we cite are for the *Gaussian (AWGN, LOS-like, delay-insensitive)* relay channel; in a pure NLOS fast-fading regime the relevant info-theoretic story changes (the interesting structure of the relay-channel capacity problem is tied to the delay/phase/LOS structure). Since our *simulated* channel is NLOS fast-fading, citing the LOS/Gaussian capacity bounds as if they govern our setting overstates their relevance.
**Plan (deferred):**
- Reframe §1.1.1 to state clearly that the Cover–El Gamal / Gaussian-DF results are presented as **classical theoretical context** that motivates DF as a baseline, and explicitly note they assume the Gaussian/LOS, delay-sensitive setting — distinct from the NLOS fast-fading regime actually simulated.
- Either (a) move the heavy info-theoretic capacity material to a clearly-labelled "Background" framing, or (b) keep it but add one sentence acknowledging its assumptions and that our performance metric is uncoded BER under NLOS fading, not ergodic capacity.

### Comment 3 — "First real SISO fixed-gain, then complex MIMO fast-fading Rayleigh, then DMT (only relevant for slow fading). Concentrate on the model you actually studied."
**Where:** ch01 §1.2 (real SISO, fixed gain), ch01 §1.3 MIMO system model + §1.4 DMT (Zheng–Tse), ch04 §4.1.
**Diagnosis:** Valid and the most important structural point.
- The DMT (\(d(r)=(N_t-r)(N_r-r)\)) is a **slow-fading / outage** concept; our 2×2 Hop 2 uses **fast (per-use) Rayleigh fading** with uncoded BER. Presenting DMT as if it characterizes our system is inconsistent.
- Mixing "real SISO fixed-gain" with "complex fast-fading MIMO" without signposting confuses which is Hop 1 vs Hop 2.
**Plan (deferred):**
- Keep DMT only as a **brief contextual mention** (one or two sentences) explicitly labelled as slow-fading background that motivates *why* equalizer choice matters — and state plainly that our evaluation uses fast-fading uncoded BER, so DMT is not the operating metric. Alternatively, demote DMT to a footnote / Background subsection.
- Make the SISO-Hop1 vs MIMO-Hop2 split unmistakable: relabel so the reader always knows the real-valued fixed/SISO description is **Hop 1** and the complex fast-fading 2×2 is **Hop 2**.
- Ensure §1.4 MIMO capacity (Telatar/Foschini) is framed as background motivating equalization, consistent with the fast-fading uncoded-BER metric we actually report.

### Comment 4 — "Time index is suppressed, yet you apply a relaying function that transforms received → transmitted; explain half-duplex (time-varying operation), and that w.l.o.g. the relay splits into a listening phase and a transmission phase."
**Where:** ch01 §1.2 (Slot 1 / Slot 2 already named) and ch04 §4.1 (relay window \(f_\theta(y_{R,i-w:i+w})\)).
**Diagnosis:** Mostly a notational/exposition gap. We *do* mention half-duplex and two slots, but:
- The channel equations suppress the time index \(i\) (e.g., \(y_R = x + n_1\)), while the relay function is explicitly windowed over time (\(y_{R,i-w:i+w}\)). The instructor wants the time index made explicit and the listening/transmission phase split spelled out.
**Plan (deferred):**
- Reintroduce the discrete-time index in the model equations: \(y_R[i] = x[i] + n_1[i]\), \(y_D[i] = x_R[i] + n_2[i]\), \(x_R[i] = f_\theta(y_R[i-w:i+w])\).
- Add 2–3 sentences defining **half-duplex** as the relay not transmitting and receiving in the same time/frequency resource, and state that **w.l.o.g.** the relay operation splits into a **listening phase** (collect \(y_R[\cdot]\)) and a **transmission phase** (emit \(x_R[\cdot]\)). This is consistent with the existing Slot 1 / Slot 2 description; just make the phrasing and time-index explicit.

---

## 4. Risk / effort assessment

- All four are **clarification / framing edits** in ch01 §1.1–1.4 and ch04 §4.1. **No new experiments, no code, no re-runs** are required.
- Comment 3 (DMT) is the only one that may need a small structural move (demote DMT to background/footnote). Low risk.
- Estimated effort: ~1 focused editing pass on ch01 + a couple of sentences in ch04, then recompile and re-sync. Citation count and figures are unaffected.

---

## 5. Recommended sequence (when approved to edit — NOT done now)

1. ch01: add explicit "Model studied in this thesis" anchor (Comment 1 + 3).
2. ch01 §1.1.1: reframe info-theoretic bounds as Gaussian/LOS background with assumptions stated (Comment 2).
3. ch01 §1.4: demote DMT to brief slow-fading background; clarify fast-fading uncoded-BER metric (Comment 3).
4. ch01 §1.2 + ch04 §4.1: restore time index; add half-duplex / listening-phase / transmission-phase paragraph (Comment 4).
5. Recompile `thesis_tau.tex`, verify 0 undefined refs, re-sync overleaf, commit.

---

## 6. Draft email reply to instructor

> **Subject:** Re: Thesis draft comments — system model, info-theoretic framing, and half-duplex notation
>
> Dear Professor [Name],
>
> Thank you very much for the detailed comments on the draft — they are very helpful, and I agree with all four points. They mostly concern how I *frame and signpost* the model, and I have a concrete plan to tighten the manuscript accordingly. A short summary of how I intend to address each:
>
> 1. **Jumping between relay settings.** You are right that the introduction moves between several models without a single anchor. I will add an explicit "System Model Studied in This Thesis" statement near the start of Chapter 1 and reference it from the Methods chapter, so the reader carries one model throughout: a half-duplex, two-hop link with no direct source–destination path, a SISO first hop, and a 2×2 i.i.d. Rayleigh fast-fading second hop with BPSK as the primary modulation. All other models will be clearly labelled as background or bounds rather than the operating regime.
>
> 2. **Information-theoretic results and line of sight.** I take your point that the Cover–El Gamal and Gaussian decode-and-forward capacity results I cite assume the Gaussian (LOS-like, delay-sensitive) relay channel, whereas the channel I actually simulate is NLOS fast-fading, for which the information-theoretic problem is, as you note, not delay-sensitive. I will reframe that material explicitly as classical theoretical context that motivates DF as a baseline, state its assumptions, and make clear that my reported metric is uncoded BER under NLOS fading rather than ergodic capacity.
>
> 3. **Concentrating on the model actually studied (SISO vs. MIMO, and the DMT).** I agree the diversity–multiplexing tradeoff is a slow-fading/outage concept and is not the right characterization for my fast-fading, uncoded-BER evaluation. I will demote the DMT to a brief, clearly-labelled background mention that only motivates why the equalizer choice matters, and I will sharpen the text so it is unambiguous that the real-valued fixed/SISO description refers to the first hop while the complex 2×2 fast-fading description refers to the second hop.
>
> 4. **Suppressed time index and half-duplex operation.** I will reintroduce the discrete-time index in the model equations (e.g., \(y_R[i] = x[i] + n_1[i]\), \(x_R[i] = f_\theta(y_R[i-w:i+w])\)) so the relaying function is clearly a time-domain operation, and I will add a short paragraph defining half-duplex (the relay does not receive and transmit on the same resource) and stating that, without loss of generality, the relay operation splits into a listening phase and a transmission phase — consistent with the two-slot description already in the draft.
>
> These are all clarification and framing revisions; they do not require any new experiments or changes to the results. I expect to have the revised Chapter 1 (and the corresponding Methods system-model paragraph) ready for your review by [date]. Please let me know if you would prefer me to address the inline comments in the document directly and return a tracked-changes version.
>
> Thank you again for the careful reading.
>
> Best regards,
> Gil Zukerman