# Active Context (update this file first, every session)

_Last updated: 2026-07-11_

## Latest result: QPSK/16-QAM hard-vs-soft DF (e6_sim_enhanced_multimod.py — EXECUTED)
User answered the open question from last update ("Qpsk and qam16") — extended the DF hard/soft decision-boundary comparison beyond BPSK. Full run completed (5 trials × 50k bits, SNR 0–20dB/2dB, unknown ISI → AWGN, all 3 modulations). Output: `/tmp/e6_sim_enhanced_multimod_comparison.png` (3-panel: BPSK/QPSK/QAM16), `/tmp/e6_sim_enhanced_multimod_results.npy`. **Ephemeral — not committed to repo.**

Framework additions landed (see `systemPatterns.md`/`techContext.md` for detail): `ComplexISIChannel`, `ComplexAWGNChannel` in `relaynet/channels/e6_channels.py` (+ `__init__.py` export); modulation-aware `DFHardRelay`/`DFSoftRelay` classes local to `e6_sim_enhanced_multimod.py`.

Key finding: **the DF-Hard non-monotonic ISI lock-in effect (vs DF-Soft's robustness) generalizes to QPSK and 16-QAM** — same qualitative shape across all three modulations. 16-QAM sits at a visibly higher overall BER floor (~0.27–0.42 vs BPSK/QPSK's ~0.18–0.34), consistent with denser constellations being more fragile under ISI + noise.

Also resolved a misdiagnosis from last session: `calculate_ber` was never actually modulation-specific (pure bit comparison) — the real blocker was the real-only ISI/AWGN channels and BPSK-hardcoded `DecodeAndForwardRelay`. `techContext.md` gotcha #3 updated to reflect this.

## Still scoped out (flagged to user, not yet requested)
AI relays (MLP-170/512, Viterbi-Genie) for QPSK/16-QAM. `MLPRelay` currently regresses a single real tanh output per window — valid for BPSK, not for 2- or 4-bit/symbol modulations without a multi-output (or complex-output) redesign. This is a materially larger task than the classical DF hard/soft extension; do not start it without the user asking.

## Immediate next step
None pending — awaiting user direction. Natural candidates if asked:
1. Multi-output MLP (and/or Viterbi generalization) for QPSK/16-QAM AI relay comparison.
2. Commit the `/tmp` chart+data into the repo (e.g. under `results/` or a new `e6_results/`) if these numbers should be kept long-term — currently they will NOT survive a session restart.
3. Resume the "not started" E6 ports (COMPOSITE, BLIND, PARTIAL, COMPLEXITY) — see `progress.md`.

## Repo hygiene
`memory-bank/`, root `context.md`, and `CLAUDE.md` were set up this session per user request — see `context.md` for the quick-start pointer. `.clinerules/` remains authoritative for anything touching the thesis LaTeX/citations/appendices.
