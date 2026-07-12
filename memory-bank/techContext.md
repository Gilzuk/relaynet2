# Tech Context

## Stack
- Python 3, NumPy (hand-rolled MLP forward/backward + Adam — no PyTorch/TF dependency in the E6 port path)
- Matplotlib for BER curve plots
- LaTeX (XeLaTeX) + BibTeX for the thesis document, `sync_overleaf.py` to push to Overleaf
- No pytest — validation scripts use plain asserts and print-based pass/fail

## Repo root is cluttered
The repo root has ~150+ loose files: LaTeX build artifacts, many one-off `_*.py` / `*.py` thesis-processing scripts (citation auditing, appendix building, chapter splitting, etc.), multiple `thesis*.{md,tex,docx,pdf}` variants, and `results/` (the canonical figures/tables directory referenced by the thesis). **Don't assume every `.py` at root is relevant to the current task** — check filenames against what's described in `activeContext.md`/`progress.md` before touching.

## Key gotchas encountered this session
1. **ViterbiMLSERelay init order bug**: `self.L` (channel length) must be set before calling `self._ls_estimate()` — was previously used-before-assigned. Fixed in `relaynet/relays/viterbi.py`.
2. **`diff_detect()` off-by-one**: differential detection for DBPSK returned one fewer sample than the input (boundary condition at the first symbol). Fixed by prepending `1.0` for the first element in `e6_flat_ported.py`.
3. **QPSK/QAM16 `calculate_ber` — RESOLVED, was a misdiagnosis**: `bpsk.calculate_ber(tx_bits, rx_bits)` is pure bit-array comparison, not modulation-specific — it already works for any scheme via `from relaynet.modulation import calculate_ber` (re-exported at package level). The real gap was that `ISIChannel` only adds **real** AWGN (breaks complex QPSK/16-QAM symbols) and `DecodeAndForwardRelay` hard-codes BPSK modulate/demodulate. Fixed by adding `ComplexISIChannel` + `ComplexAWGNChannel` to `relaynet/channels/e6_channels.py` (complex-aware ISI/AWGN, same sigma convention) and modulation-aware `DFHardRelay`/`DFSoftRelay` classes in `e6_sim_enhanced_multimod.py` (not promoted into `relaynet/relays/` yet — kept local since `DecodeAndForwardRelay`'s BPSK-only behavior is relied on elsewhere and must not change).
4. **"Genie" CSI baseline can be a weaker baseline than expected — RESOLVED, not a bug**: on `ISIRayleighChannel`/`ComplexISIRayleighChannel` (ISI + Rayleigh fading + AWGN), `Viterbi-Genie` only ever knows the static ISI taps (this repo's established convention), never the fading — its branch metric implicitly assumes unit gain. A pilot-based LS channel estimate on the same channel accidentally self-calibrates toward `true_taps * E[|h|]` (E[|h|]=√π/2≈0.886 for this fading model, confirmed via a 200-repeat isolated diagnostic to <0.2% error), which is a *better-matched* metric scale than genie's fading-blind taps — so the "genie" baseline lost to a realistic pilot estimate by ~0.002–0.005 BER on `e6_viterbi_qpsk_pilot_overhead.py`. Confirmed genuine (not noise) via N_TRIALS=20 and resolved by adding a properly fading-aware oracle (`Viterbi-Genie-EhScaled` = genie taps × E[|h|]), which matches the pilot estimate almost exactly and restores the expected "genie ≥ estimate" ordering. **Takeaway for any future genie/oracle baseline on a multi-impairment channel**: verify the genie's assumed CSI actually covers every impairment the branch metric needs to be correctly scaled against — a partial-CSI genie is not automatically an upper bound.

## SNR convention (load-bearing — do not change without updating everywhere)
- γ = 1/σ² = 10^(SNR_dB/10)
- Standalone E6 scripts: `sigma = 10**(-snr_db/20.0)`
- relaynet: `noise_power = signal_power / 10**(snr_db/10)`
- These are mathematically identical. Verified numerically in `test_e6_core.py`.

## Experiment scale conventions
- **Iteration/dev scale**: 5 trials × 50,000 bits per SNR point, SNR 0–20 dB in 2 dB steps. Used for all E6 porting verification so far.
- **Project standard / final thesis-integration scale**: 10 trials × 100,000 bits per SNR point — NOT yet run for the E6 port. This is required before replacing any `results/e6_*.png` figures or updating Chapter 7 tables (see `progress.md` → Pending).

## Where results currently live
All E6 port outputs so far are in `/tmp/` (ephemeral — this is a remote/cloud session, container is reclaimed after inactivity). Nothing has been copied into the tracked `results/` directory yet. **Anything under `/tmp/` will NOT survive a session restart** — if a run needs to be preserved, it must be committed into the repo (e.g., under `results/` or a new `e6_results/` directory) or re-run.
