# Active Context (update this file first, every session)

_Last updated: 2026-07-11_

## Latest result (e6_sim_enhanced.py — EXECUTED)
Full run completed (5 trials × 50k bits, SNR 0–20dB/2dB, BPSK only). Output: `/tmp/e6_sim_enhanced_comparison.png`, `/tmp/e6_sim_enhanced_results.npy`. **Note: `/tmp` does not persist — re-run if needed in a future session, or copy into the repo if these numbers should be kept.**

Key findings:
- AF/DF-Hard/DF-Soft all ISI-floor-limited (~0.19–0.24 BER), as expected.
- **DF-Hard is non-monotonic**: improves to 0.201 @10dB then rises to 0.235 @20dB — hard decisions lock in ISI errors that then propagate. **DF-Soft does not show this** — tracks AF, keeps improving to 0.206 @20dB. This is the hard-vs-soft decision effect the user asked to demonstrate.
- AI relays: Viterbi-Genie ≤ MLP-512 ≲ MLP-170 at low/mid SNR (e.g. 8dB: 0.0825 / 0.0956 / 0.0941), all converge to ~0.005 by 20dB. MLP-512 vs MLP-170 gap is small/within-noise at this trial count — not a strong capacity signal, would need more trials to sharpen.
- All AI relays crush classical at high SNR (0.005 vs 0.20+).

Caveat still open: **BPSK only** — QPSK/QAM16 hard/soft DF comparison not actually run (blocked on missing `calculate_ber` for those modulations, see techContext.md gotcha #3). Not yet resolved with the user whether that's needed.

## What's happening right now
Working on `e6_sim_enhanced.py` — a multi-architecture BER comparison requested by the user on top of the already-verified E6 port. Request was:

> "Can you regenerate the charts only with 3 leading ai relay architecture additionally to af/df. Regarding the df for qpsk and qam16 the decision boundary is not sign it should be hard or soft. Use both"

Interpreted as: compare AF, DF-hard, DF-soft against 3 AI relay tiers (MLP-170, MLP-512, Viterbi-Genie), and make the DF classical baseline decision-boundary-aware (hard quantization vs soft/no-quantization) rather than a bare `sign()`, since that distinction matters once you generalize past BPSK to QPSK/QAM16.

Current state: the script exists and is committed (`e6_sim_enhanced.py`, commit `7e2b7c3` on `claude/porting-md-file-l6xzsr`) but **has not been executed yet**. It's scoped to BPSK only because `relaynet.modulation.qpsk`/`.qam` don't have `calculate_ber()` (see `techContext.md`).

## Immediate next step
Run `python3 e6_sim_enhanced.py` (full scale: 5 trials × 50k bits × 11 SNR points × 6 relays — expect noticeable runtime, MLP training included). Then:
1. Check `/tmp/e6_sim_enhanced_comparison.png` renders both panels correctly (all-relays + AI-zoom).
2. Sanity-check ordering: Viterbi-Genie ≤ MLP-512 ≤ MLP-170 in BER (more capacity/CSI → better).
3. Sanity-check DF-Soft vs DF-Hard differ meaningfully (soft should do somewhat better since it avoids committing to a wrong hard decision before hop 2 noise is added — but on an ISI channel both are still floor-limited).
4. Report results back to user; do NOT claim QPSK/QAM16 results were produced — flag explicitly that this run is BPSK-only.

## Open question for the user (not yet asked)
Whether they actually want `calculate_ber` implemented for QPSK/QAM16 modulation modules so the hard/soft DF comparison can be run on those modulations for real, or whether the BPSK-scoped demonstration satisfies the request. Surface this rather than silently declaring the task fully done.

## Repo hygiene note for this session
The user just asked to set up a `memory-bank/`, root `context.md`, and Claude Code rules (`CLAUDE.md`) — this is that setup work, done in parallel with/before returning to the `e6_sim_enhanced.py` execution above. See `context.md` at repo root for the quick-start pointer, and `.clinerules/` (pre-existing, thesis-writing-focused, still authoritative for LaTeX/citation/appendix rules) alongside the new `CLAUDE.md` (Claude-Code-specific, points back into this memory bank).
