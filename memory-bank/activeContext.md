# Active Context (update this file first, every session)

_Last updated: 2026-07-14_

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

## Immediate next step
None pending — awaiting user direction. All 7 PORTING.md experiments are now ported; remaining work is the "After porting" thesis-integration checklist plus the previously-identified gaps in the first 3 experiments:
1. Rescale E6_SIM/E6_VITERBI/E6_FLAT (and now COMPOSITE/BLIND/PARTIAL/COMPLEXITY where applicable) to project-standard 10×100k.
2. Run direct standalone-script comparisons (not just comparison against PORTING.md's stated expected numbers).
3. Produce thesis-styled figures via relaynet's plotting, replace `results/e6_*.png`.
4. Update Chapter 7 tables and Appendix C reproducibility statement; remove the "clean-room/standalone" caveat.
5. Commit the various `/tmp` charts+data into the repo if these numbers should be kept long-term — nothing under `/tmp` survives a session restart.

## Environment issue (unresolved, non-blocking for local work)
Git push to `origin/claude/porting-md-file-l6xzsr` has been failing intermittently this session (`fatal: could not read Username for 'https://github.com'`), and commit signing has also been failing (stop-hook flags commits as Unverified). This is an environment credential/signing service issue, not a repo problem — retried with backoff each time, work stays committed locally regardless. Check before assuming a commit made it to `origin`.

## Repo hygiene
`memory-bank/`, root `context.md`, and `CLAUDE.md` were set up earlier this session per user request — see `context.md` for the quick-start pointer. `.clinerules/` remains authoritative for anything touching the thesis LaTeX/citations/appendices.
