# E6 Multi-Training Results (N_TRAIN=3)

> **Historical snapshot (2026-08-27); do not cite.** The thesis reads
> `e6_unknown_channel_results/`, where every `e6_*_ported.py` script now writes
> its output. The flat-channel figures below predate the October 2026 Hop-2
> noise correction (`thesis/RERUN_CHANGELOG.md`), which roughly halved the DF and
> MLP error rates. Kept for provenance.

Generated 2026-08-27. Each experiment was run with `N_TRAIN=3` independent
MLP training seeds, addressing the reviewer gap: "training done only once /
results are Monte Carlo based on single training seed."

## What changed vs. previous runs

Each `e6_*_ported.py` script now trains `N_TRAIN=3` MLP instances with
seeds `seed+0`, `seed+1`, `seed+2`. The BER Monte Carlo (`N_TRIALS`) runs
in full for every training instance, giving `N_TRAIN × N_TRIALS` total
columns. Mean ± 95% CI are pooled over all columns, so both training
variance and inference variance are captured.

Classical baselines (AF, DF, Viterbi) have no training variance; they are
re-run alongside each MLP instance for consistent pairing.

## Files

| File | Experiment | Scale | Key finding |
|---|---|---|---|
| `e6_sim_ported_results.npy` | E6_SIM: unknown ISI / nonlinear-bias | 3×10×100k | MLP <5e-5 @16dB S1; DF non-monotonic ✓ |
| `e6_flat_ported_results.npy` | E6_FLAT: flat memoryless channels | 3×10×100k | MLP-DF gap: F1=0.0064, F2=0.0026, F3=0.0036 |
| `e6_composite_ported_results.npy` | E6_COMPOSITE: ISI×PA×phase | 3×10×100k | MLP-169 ~0.0025 @20dB; Viterbi-diff ~2dB ahead |
| `e6_blind_ported_results.npy` | E6_BLIND: no pilots, random ISI | 3×50×20k | Viterbi-blind unstable (CI ±0.024 vs MLP ±0.002 @10dB) |
| `e6_partial_ported_results.npy` | E6_PARTIAL: pilot sweep + block-length sweep | 3×50×20k | Viterbi collapses at 5 pilots (BER=0.1235) |

## Reproducibility

```bash
cd /path/to/relaynet2
pip install -e .
python e6_sim_ported.py
python e6_flat_ported.py
python e6_composite_ported.py
python e6_blind_ported.py
python e6_partial_ported.py
```

The scripts now save to `e6_unknown_channel_results/`; this directory is not updated.
