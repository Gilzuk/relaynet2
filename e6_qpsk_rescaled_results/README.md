# QPSK relay-comparison results — project-standard scale (10 trials × 100k bits)

Rescaled reruns of three findings from the QPSK/symmetric-hop exploration thread, at the
project-standard Monte Carlo budget (per `PORTING.md` / `memory-bank/techContext.md`:
10 trials × 100,000 bits per SNR point, vs the 5×50k dev/iteration scale used while these
were being developed). All qualitative conclusions from the dev-scale runs replicate here.

**Status: computed and saved, NOT yet integrated into the thesis LaTeX.** Per explicit user
instruction (2026-07-12), do not edit `chapters/*.tex` with this content until directed —
there is no existing home for it in the current chapter structure (see
`memory-bank/activeContext.md` for the naming-collision finding: the canonical "E6" in
`chapters/ch05_experiments.tex` is an unrelated CSI-injection experiment).

## Files

- `e6_relay_comparison_symmetric.png` / `.npy` — BPSK (AF/DF-Hard/DF-Soft/MLP-170/Viterbi-Genie)
  and QPSK (AF/DF-Hard/DF-Soft/Viterbi-Genie) relay comparison under symmetric ISI+Rayleigh+AWGN
  hops. Source: `e6_relay_comparison_symmetric.py`.
- `e6_mlp_qpsk_vs_viterbi.png` / `.npy` — 4-class MLP-QPSK classifier vs Viterbi-Genie, BER +
  latency, L=3 taps. Source: `e6_mlp_qpsk_vs_viterbi.py`.
  **Latency values in this .npy were patched after saving** — the in-run measurement (42.5x)
  was inflated by CPU contention from two other rescale jobs running concurrently; corrected
  via an isolated re-measurement (183.1x, `repeats=7`) once the other jobs finished. See the
  `latency_note` key in the .npy dict.
- `e6_viterbi_qpsk_partial_csi.png` / `.npy` — worst (5-pilot) / medium (20-pilot) / ideal
  (fading-aware genie) CSI-quality comparison for QPSK Viterbi. Source:
  `e6_viterbi_qpsk_partial_csi.py`.

## Headline numbers (@20dB SNR unless noted)

| Comparison | Result |
|---|---|
| Symmetric hops, BPSK | DF-Hard worst (0.384), AF/DF-Soft plateau (~0.338), MLP-170/Viterbi-Genie floor (~0.230/0.231) |
| Symmetric hops, QPSK | Same pattern, statistically indistinguishable from BPSK per relay (modulation-invariance) |
| MLP-QPSK vs Viterbi-Genie | BER 0.2367 vs 0.2298 (~3% gap); latency 8.93ms vs 1635ms (**183x**) |
| CSI quality (QPSK Viterbi) | Worst(5p): 0.2505±0.028, Medium(20p): 0.2281±0.0014, Ideal: 0.2272±0.0007 — worst-case CI still 5-40x wider than ideal even at 10 trials |

See `memory-bank/activeContext.md` and `memory-bank/progress.md` for full writeups, mechanisms,
and caveats (e.g. the unresolved L=4 tap-count anomaly, which was NOT rescaled — still flagged
as not-thesis-ready).
