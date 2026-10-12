# Unknown-Channel Relay Experiments — relaynet-generated results

Thesis-styled figures (via `relaynet/visualization/plots.py`'s `plot_ber_with_ci`,
same style as Ch5-6 figures) and raw `.npy` results for all 7 `experiments-standalone/`
E6-prefixed scripts, ported to `relaynet`: `e6_sim_ported.py`, `e6_viterbi_ported.py`,
`e6_flat_ported.py` (rescaled to the project-standard 10 trials x 100k bits), plus
`e6_composite_ported.py`, `e6_blind_ported.py`, `e6_partial_ported.py`,
`e6_complexity_ported.py` (verified at their own standalone trial budgets).

Figures are named `unkchan_*` for a historical reason: when they were generated, Chapter 5
had an unrelated "E6" experiment (CSI injection for 16-QAM/16-PSK), since removed. In the
current thesis, Table 5.1 labels the unknown-channel study itself E6.

**Role in the thesis.** These files back Chapter 6 of the thesis on `main`
(learned relaying under unknown and mismatched channels, the principal
contribution). The tables and figures that cite each file are listed in the
experiment ledger (thesis Appendix 9.3) and checked by `verify_thesis_tables.py`;
the channel models are specified in thesis Appendix 9.6. Setups S2 and S4 of
`e6_sim_ported.py` (ISI to Rayleigh, and the Rayleigh control) remain in the
arrays and figures here but are not reported in the thesis, whose fixed-ISI
results use S1 (ISI on Hop 1, AWGN on Hop 2); the nonlinear-bias setup S3 is not
tabulated in Chapter 6.

**October 2026 changes.** `e6_flat_ported_results.npy` was rerun with the
corrected Hop-2 noise (variance 1/(2γ) per real dimension for real forwarded
signals; the console log is `e6_flat_ported_console.txt`). The composite AF
curve is replaced by `codex_composite_af_validation.json` (AF had added half the
intended complex Hop-2 noise power), and the 12, 14 and 16 dB MLP entries of the
fixed-ISI table come from the predeclared fixed-budget run in
`codex_isi_fixed_budget_validation.json` (16 dB: 36 errors in 3×10^8 bits).

## Files

- `unkchan_sim_S1.png` .. `unkchan_sim_S4_control.png` — unknown ISI/nonlinear-bias
  two-hop relay comparison (AF/DF/MLP-170), 4 scenarios (ISI->AWGN, ISI->Rayleigh,
  nonlinear+bias->AWGN, Rayleigh->Rayleigh control).
- `unkchan_viterbi_awgn.png`, `unkchan_viterbi_rayleigh.png` — Viterbi-genie vs
  LS-pilot-estimated Viterbi MLSE on the same unknown-ISI channel.
- `unkchan_flat_phase.png`, `unkchan_flat_gain.png`, `unkchan_flat_iqimb.png` — the
  three memoryless control experiments (unknown phase, gain and branch asymmetry;
  `iqimb` is the code name of the asymmetry case). These figures predate the
  October 2026 Hop-2 noise rerun; the thesis table uses the rerun.
- `unkchan_composite.png` — ISI x PA-nonlinearity x unknown-phase cascade (DBPSK).
- `unkchan_blind.png` — random ISI per trial, no pilots (blind regime).
- `unkchan_partial.png` — pilot-count and block-length sensitivity sweeps.
- `unkchan_complexity.png` — Viterbi MLSE vs. MLP analytical cost + measured wall-clock.
- `*_results.npy` — raw mean/CI arrays behind each figure, including the
  composite, blind, pilot-budget and QPSK studies; every `e6_*_ported.py` script
  writes here, not to `/tmp/`.
- `codex_isi_fixed_budget_validation.json`, `codex_composite_af_validation.json`,
  `codex_viterbi_consistency_awgn.{json,npy}` — the fixed-budget MLP confirmation,
  the corrected composite AF curve and a Viterbi consistency check.

## Verification

E6_SIM, E6_VITERBI, E6_FLAT ran at 10x100k full scale and were cross-checked directly
against the actual (unmodified) `experiments-standalone/` scripts run at their own
native 5x50k budget — not just against `PORTING.md`'s stated target numbers. E6_SIM
and E6_VITERBI matched closely with no issues found. E6_FLAT's rescale surfaced two
real bugs (a sign inversion in the DBPSK path, and an unpaired per-relay channel
realization that broke the "same channel, compare relays" experimental design), both
now fixed — see `memory-bank/progress.md`'s "E6_FLAT bug fixes" section for the full
writeup. E6_COMPOSITE/BLIND/PARTIAL/COMPLEXITY were verified against `PORTING.md`'s
stated targets at their own standalone trial budgets (see `memory-bank/progress.md`
for full numeric details).

## Rare-event cells (16–20 dB) and where their error counts live

The 16, 18 and 20 dB entries of `tbl:tableE6` are not fixed-budget averages. They
use an error-counting rule: transmit until the first error at `N1` bits, then
continue to `10 x N1` bits (capped per SNR by `FIRST_ERROR_MAX_BITS_BY_SNR`), and
report accumulated errors divided by total exposure. A cell that sees no error
inside the cap reports the rule-of-three 95% upper bound `3/N`, not `1/N` and not
zero. This replaced an estimator that stopped at the first event and reported its
reciprocal waiting time — one sample per cell, which returned values as
implausible as `0.5` for DF at 16 dB where error counting gives `0.2296`.

`e6_sim_ported_results.npy` as committed (run `a3a07ab`, 2026-08-31) stores the
BER means and CIs but **not** the underlying bit and error counts: `_save` only
began persisting `rare_event_meta` after that run finished. Until the next full
regeneration, the per-cell counts for that run are in
**`results/e6_sim_rerun_progress.txt`**, which is committed alongside the `.npy`.
For S1 at 16 dB, for example, it records
`MLP=4.79e-08 (334,002,040b, 16 err)`.

Any BER cell below roughly 1e-4 should have its error *count* checked in that file
before a ratio or a claim is built on it.

In the current thesis the 16 dB MLP entry no longer uses this adaptive estimate:
it is the fixed-budget value 1.20e-7 from `codex_isi_fixed_budget_validation.json`,
and the adaptive 18–20 dB entries are kept as exploratory records only.
