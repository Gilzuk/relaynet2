# Unknown-Channel Relay Experiments — relaynet-generated results

Thesis-styled figures (via `relaynet/visualization/plots.py`'s `plot_ber_with_ci`,
same style as Ch5-6 figures) and raw `.npy` results for all 7 `experiments-standalone/`
E6-prefixed scripts, ported to `relaynet`: `e6_sim_ported.py`, `e6_viterbi_ported.py`,
`e6_flat_ported.py` (rescaled to the project-standard 10 trials x 100k bits), plus
`e6_composite_ported.py`, `e6_blind_ported.py`, `e6_partial_ported.py`,
`e6_complexity_ported.py` (verified at their own standalone trial budgets).

Files are named `unkchan_*` rather than `e6_*` deliberately: the thesis's own Chapter 5
already has an unrelated "E6" experiment (CSI Injection & LayerNorm for 16-QAM/16-PSK).
The "E6" prefix in `experiments-standalone/` is that suite's own internal addendum
numbering, not a thesis chapter reference — the filename/title change avoids any
confusion between the two.

**Now integrated into the thesis as supplementary material**: Appendix A.14
("Supplementary: Unknown-Channel Relay Experiments") on the `clean-thesis` branch,
covering all 7 experiments below, with a one-sentence pointer from Chapter 6's
Future Work section (the "Imperfect CSI" item). This is explicitly supplementary —
not one of Chapter 5's eight numbered experiments (E1-E8), no renumbering, no
existing conclusions altered.

## Files

- `unkchan_sim_S1.png` .. `unkchan_sim_S4_control.png` — unknown ISI/nonlinear-bias
  two-hop relay comparison (AF/DF/MLP-170), 4 scenarios (ISI->AWGN, ISI->Rayleigh,
  nonlinear+bias->AWGN, Rayleigh->Rayleigh control).
- `unkchan_viterbi_awgn.png`, `unkchan_viterbi_rayleigh.png` — Viterbi-genie vs
  LS-pilot-estimated Viterbi MLSE on the same unknown-ISI channel.
- `unkchan_flat_phase.png`, `unkchan_flat_gain.png`, `unkchan_flat_iqimb.png` — the
  three memoryless "control" experiments (unknown phase/gain/I-Q-imbalance), showing
  classical DF ties the MLP absent any channel memory.
- `unkchan_composite.png` — ISI x PA-nonlinearity x unknown-phase cascade (DBPSK).
- `unkchan_blind.png` — random ISI per trial, no pilots (blind regime).
- `unkchan_partial.png` — pilot-count and block-length sensitivity sweeps.
- `unkchan_complexity.png` — Viterbi MLSE vs. MLP analytical cost + measured wall-clock.
- `*_results.npy` — raw mean/CI arrays behind each figure (the `/tmp/e6_*.npy` sources
  for composite/blind/partial/complexity are ephemeral; only sim/viterbi/flat's are
  copied into this directory).

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
