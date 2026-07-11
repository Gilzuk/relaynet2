# Active Context (update this file first, every session)

_Last updated: 2026-07-11_

## Latest result: Viterbi-Genie MLSE for QPSK (e6_viterbi_qpsk.py — EXECUTED)
User asked "How is soft decision inferior to hard?" (answered from `e6_sim_enhanced_multimod.py` data: DF-Hard wins at low/moderate SNR via denoising-on-correct-decode, loses at higher SNR / on denser constellations because ISI-driven errors are systematic and hard-decision commits to them at full confidence with zero recoverability — QAM16 showed this most starkly, DF-Hard inferior across nearly the whole 0-16dB range). Follow-up "what's the optimal DF decoder for QAM16" → answered conceptually: neither hard nor soft memoryless decision is optimal against a *memory* (ISI) impairment; the real optimum is sequence detection (Viterbi/MLSE) or a learned sequence estimator. User then asked to implement **Viterbi only for QPSK** (explicitly not QAM16, scope note).

Built `ViterbiMLSEQPSKRelay` (`relaynet/relays/viterbi.py`) — generalizes the existing BPSK `ViterbiMLSERelay` trellis to the 4-symbol Gray-coded QPSK alphabet (16 states for L=3 taps), complex branch metrics. Verified noiseless ISI round-trip gives exactly 0 BER before running the full sweep. Exported via `relaynet/relays/__init__.py`.

Ran `e6_viterbi_qpsk.py` (new script, full scale 5×50k, unknown ISI → AWGN, reuses `DFHardRelay`/`DFSoftRelay` from `e6_sim_enhanced_multimod.py`): **Viterbi-Genie breaks completely away from the AF/DF-Hard/DF-Soft ISI floor (~0.18–0.23) starting ~6dB, crosses BER<1e-2 at 10dB, reaches ~0 by 14dB** — while all three classical relays stay pinned at the floor regardless of SNR, confirming the "memory needs sequence detection" argument concretely for QPSK. Output: `/tmp/e6_viterbi_qpsk_comparison.png`, `/tmp/e6_viterbi_qpsk_results.npy` (ephemeral).

## Still explicitly scoped out (per user instruction / not yet requested)
- **QAM16 Viterbi** — user said "no, viterbi only for qpsk". Do not build a 16-QAM trellis (256 states for L=3) unless asked.
- **AI relays (MLP) for QPSK/16-QAM** — `MLPRelay` regresses a single real tanh output per window, valid for BPSK only; would need a multi-output/complex-output redesign. Not started.

## Immediate next step
None pending — awaiting user direction. Natural candidates if asked:
1. Commit the `/tmp` charts+data into the repo if these numbers should be kept long-term — nothing under `/tmp` survives a session restart.
2. Multi-output MLP for QPSK (to compare against Viterbi-Genie the way BPSK's MLP-170/512 were compared against BPSK Viterbi-Genie).
3. Resume the "not started" E6 ports (COMPOSITE, BLIND, PARTIAL, COMPLEXITY) — see `progress.md`.

## Environment issue (unresolved, non-blocking for local work)
Git push to `origin/claude/porting-md-file-l6xzsr` has been failing intermittently this session (`fatal: could not read Username for 'https://github.com'`), and commit signing has also been failing (stop-hook flags commits as Unverified). This is an environment credential/signing service issue, not a repo problem — retried with backoff each time, work stays committed locally regardless. Check before assuming a commit made it to `origin`.

## Repo hygiene
`memory-bank/`, root `context.md`, and `CLAUDE.md` were set up earlier this session per user request — see `context.md` for the quick-start pointer. `.clinerules/` remains authoritative for anything touching the thesis LaTeX/citations/appendices.
