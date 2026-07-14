# E6_SIM / E6_VITERBI / E6_FLAT — relaynet-generated results (10x100k)

Thesis-styled figures (via `relaynet/visualization/plots.py`'s `plot_ber_with_ci`,
same style as Ch5-6 figures) and raw `.npy` results for the three
`experiments-standalone/` scripts ported to `relaynet` and rescaled to the
project-standard 10 trials x 100k bits: `e6_sim_ported.py`, `e6_viterbi_ported.py`,
`e6_flat_ported.py`.

**Not yet integrated into the thesis.** See `memory-bank/activeContext.md`'s
"thesis-integration blocker" note: the actual thesis (`chapters/ch05_experiments.tex`)
has no chapter/section matching this work, and its own existing "E6" section is a
different, unrelated experiment (CSI Injection & LayerNorm for 16-QAM/16-PSK). The
"E6" prefix on these files is `experiments-standalone/`'s own internal addendum
numbering, not a thesis chapter reference. Where (or whether) this content belongs
in the thesis is a pending decision — these figures are generated ahead of that
decision so they're ready whenever it's made.

## Files

- `e6_sim_S1.png` .. `e6_sim_S4_control.png` — unknown ISI/nonlinear-bias two-hop
  relay comparison (AF/DF/MLP-170), 4 scenarios (ISI->AWGN, ISI->Rayleigh,
  nonlinear+bias->AWGN, Rayleigh->Rayleigh control).
- `e6_viterbi_awgn.png`, `e6_viterbi_rayleigh.png` — Viterbi-genie vs
  LS-pilot-estimated Viterbi MLSE on the same unknown-ISI channel.
- `e6_flat_phase.png`, `e6_flat_gain.png`, `e6_flat_iqimb.png` — the three
  memoryless "control" experiments (unknown phase/gain/I-Q-imbalance), showing
  classical DF ties the MLP absent any channel memory.
- `*_results.npy` — raw mean/CI arrays behind each figure.

## Verification

All three ran at 10x100k full scale and were cross-checked directly against the
actual (unmodified) `experiments-standalone/` scripts run at their own native
5x50k budget — not just against `PORTING.md`'s stated target numbers. E6_SIM and
E6_VITERBI matched closely with no issues found. E6_FLAT's rescale surfaced two
real bugs (a sign inversion in the DBPSK path, and an unpaired per-relay channel
realization that broke the "same channel, compare relays" experimental design),
both now fixed — see `memory-bank/progress.md`'s "E6_FLAT bug fixes" section for
the full writeup.
