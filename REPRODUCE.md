# Reproducing the thesis

Everything reported in *Deep Learning Architectures for Two-Hop Relay
Communication* can be regenerated from this repository and checked
automatically against the document.

```bash
git clone https://github.com/Gilzuk/relaynet2
cd relaynet2
make setup      # install pinned dependencies
make check      # what can this machine run?
make verify     # confirm every number in the thesis matches its data source
```

`make verify` takes about a minute, needs nothing but NumPy and SciPy, and is
the fastest way to convince yourself the document and the data agree.

---

## What "reproducible" means here

Two distinct claims, checkable independently:

1. **The thesis matches its data.** A declared subset of numerical table cells
   and prose numbers is compared against the file that produced it; it does not
   cover every number or figure. This is `make verify`: **873 values,
   0 discrepancies** (2026-10-17), exit code 1 on any mismatch.
2. **The data can be regenerated.** The experiments are re-runnable from
   scratch. The unknown-channel pipeline is fully seeded and reproduces its
   committed `.npy` files **bit-for-bit**; `run_experiments.py` is seeded via
   `--seed` (default 42).

---

## Tiers

Pick by how much compute you have. Each tier subsumes the checks of the one
above it.

| Tier | Command | Time | Needs | What it establishes |
|---|---|---|---|---|
| 0 | `make verify` | ~1 min | numpy, scipy | Every thesis number matches its data source |
| 0 | `make test` | < 1 min | + pytest | 351 tests over channels, modulation, coding, relays, statistics and the verification tooling |
| 1 | `make repro-unknown` | ~40 min | numpy, scipy | Recomputes the whole unknown-channel study (Ch. 6), then verifies |
| 1b | `make repro-qpsk` | ~20 min | numpy, scipy | Recomputes the QPSK unknown-channel study (ISI → AWGN and → Rayleigh) |
| 2 | `make repro-full` | hours | **+ torch** | Recomputes every experiment, including all neural relays |

### Tier 0 — verification

```bash
make verify
```

`verify_thesis_tables.py` reads the LaTeX sources in `thesis/chapters/`,
extracts each table cell and each numeric prose claim, and compares it against
its authoritative source: the JSON files under `results/`, the `.npy` files
under `e6_unknown_channel_results/`, or a closed-form expression evaluated in
the script. Deterministic transcriptions get a tight rounding tolerance;
re-simulated values get a Monte-Carlo tolerance.

Timing tables are reported as informational, not pass/fail — wall-clock is
machine-dependent by nature.

### Tier 1 — recompute the unknown-channel study

```bash
make repro-unknown      # all six BPSK experiments, then verify
make repro-qpsk         # the QPSK study on its own
```

This regenerates the `.npy` files and re-verifies with a Monte-Carlo tolerance.
No torch required: the unknown-channel relays (AF, DF, the minimal MLP, Viterbi
MLSE, CMA) are pure NumPy.

### Tier 2 — recompute everything

```bash
make repro-full
```

Requires PyTorch, for the VAE, CGAN, Transformer, Mamba S6 and Mamba-2 relays.

**Budget realistically.** On the CUDA machine used for the thesis, one full
nine-relay training pass took ~3.2 hours, of which the CGAN alone was ~2 hours
(64%). On CPU it is substantially slower. For a smoke test rather than a
publication-quality run, use `make quick`.

---

## October 2026 revision (Dr. Khina's review)

The experiments added or rerun in that round. Those the thesis quotes are
registered in `provenance_audit.py` and checked by `make verify`. Results are written into
`results/` (console logs as `*_console.txt`); nothing goes to `/tmp/`.

```bash
python coded_matched_metrics.py           # Table 5.9: reliability-weighted coded metrics (CMM_TRIALS, default 30)
CODED_METRIC=matched CODED_TRIALS=30 python coded_error_mechanism.py    # Table 5.6, matched rows
CODED_METRIC=matched CODED_TRIALS=30 python coded_reliable_regime.py    # Table 5.7 paragraph
CODED_METRIC=matched CODED_TRIALS=30 python coded_k_sweep_qpsk.py       # constraint-length paragraph
CODED_METRIC=matched CODED_TRIALS=30 python coded_k_sweep_qam16.py      # Table 5.10, matched columns
CODED_METRIC=matched CODED_TRIALS=30 python coded_rate_adaptation.py    # Tables 5.12 and 5.13
CODED_METRIC=matched CODED_TRIALS=30 python coded_latency_throughput.py
CODED_METRIC=matched CODED_TRIALS=30 python coded_latency_capacity.py
python coded_matched_envelopes.py         # matched block-DF envelopes, derived (no simulation)
python coded_min_size_matched.py          # coded size-sweep penalties against matched block DF
python af_block_and_conventional.py       # AF block length and conventional variable-gain AF
python factorized_mlse_equivalence.py     # two binary trellises vs the joint QPSK trellis
python e6_scalar_map.py                   # scalar MAP relay on the ISI channel
python e6_isi_penalty.py                  # ISI required-SNR penalties, derived from e6_matched_protocol.json
python e6_flat_ported.py                  # flat controls with the corrected Hop-2 noise
python hybrid_branch_selection.py         # which branch the canonical Hybrid took (replay)
```

With `CODED_METRIC` unset the coded scripts reproduce the published unweighted
results exactly; with `CODED_METRIC=matched` they write `*_matched.json`
alongside them.

The committee audit (October 2026) added a rerun of the sequence-on-memory study
behind Table 6.7 and Sec. 6.4.2. It needs torch. The rerun summary is
`docs/mamba_lookahead_rerun_2026-10-13.md`.

```bash
python seq_models_on_memory_readout.py        # Mamba centre (control) and full-window readouts
python seq_models_on_memory_bidirectional.py  # Mamba bidirectional readout
python seq_models_on_memory_rerun_controls.py # MLP-3K and Transformer-3K, same protocol
python seq_models_on_memory_repeat.py         # one fresh retrain of three configurations
python seq_models_on_memory_repeat_probe.py Mamba-S6-3K isi 1   # weights hash + machine, appended
```

---

## Requirements

Core (Tiers 0 and 1) — `pip install -r requirements-repro.txt`:

```
numpy  scipy  matplotlib  pytest
```

Optional (Tier 2 only):

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

PyTorch is genuinely optional. If it is absent, the channels package still
imports and every NumPy-only experiment runs; only the neural relays that use
torch are unavailable. `make check` reports exactly which tiers your machine can
run.

Building the PDF (`make thesis`) additionally needs TeX Live with **XeLaTeX**
and `latexmk`. Compile with XeLaTeX, not pdfLaTeX: the thesis uses `fontspec`
and `polyglossia` for the Hebrew abstract. See `thesis/OVERLEAF.md`.

---

## Layout

```
relaynet/                     Simulation framework (channels, modulation, relays, stats)
run_experiments.py            Unified runner for the main experiment sweep
e6_*_ported.py                Unknown-channel study (Ch. 6), NumPy only
e6_qpsk_unknown_channel.py    QPSK unknown-channel study, both hop-2 variants
calibrate_simulator.py        E1 calibration: simulator vs closed-form BER (Eb/N0 axis)
verify_thesis_tables.py       Checks the thesis against its data sources
results/                      Committed experiment outputs (JSON + figures)
e6_unknown_channel_results/   Committed unknown-channel outputs (.npy + figures)
tests/                        351 tests (pytest)
thesis/                       LaTeX sources, figures, fonts, CHANGELOG
  main.tex                      Compile with XeLaTeX; includes the chapters below
  chapters/ch0N_*.tex           One file per chapter. After ch05 the file numbers run
                                one ahead of the document: ch07 renders as Ch. 6
                                (unknown channels), ch08 as Ch. 7 (discussion), ch09
                                as Ch. 8 (summary); ch06 (higher-order modulation)
                                is not included by main.tex
  chapters/_*.tex               Drafting history; not included by main.tex
scripts/build_bundles.py      Rebuilds the two Overleaf zips from thesis/
scripts/plot_e6_studies.py    Regenerates the Ch. 7 composite/blind/partial figures from .npy
scripts/plot_awgn_companion.py Regenerates the Ch. 5 AWGN companion figure from JSON
scripts/strip_rev.py          Brace-matching remover for inline REV annotations
scripts/check_env.py          Environment doctor
```

---

## Determinism and caveats

- **Seeding.** Trial `t` under seed `s` initialises as `seed(s*1000 + t)`. The
  unknown-channel scripts reproduce their committed outputs bit-for-bit; this
  has been verified by re-running them from scratch and diffing every value.
- **Timing numbers will differ.** Table 5.11 and Figure 6.7(b) report
  wall-clock, which depends on your hardware. The verifier checks Table 5.11
  only against the readouts committed beside it, not against your machine.
- **A GPU changes speed, not results, for the simulations.** For fixed data
  and fixed weights the BER numbers are hardware independent; only the timing
  tables move. Training a neural relay is the exception (next item but one).
- **Floating-point across platforms.** Different BLAS builds can perturb the
  last displayed digit of an evaluation, and the verifier's rounding tolerance
  absorbs this.
- **Seeded training is exact only on one machine.** Retraining the
  sequence-on-memory relays with the same seed gave identical weights in every
  fresh process on one machine. On another machine with the same installed
  software it gave a different relay: Mamba-S6 moved by 0.21 dB and the
  Transformer by 0.89 dB, while the MLP was identical. Per-seed penalties and
  three-seed spreads of trained sequence models therefore change from machine to
  machine (Sec. 6.4.2). Rerun them on your machine and compare
  best-of-three values, not individual seeds.
- **What is *not* reproduced.** One experiment is deliberately not run and is
  flagged as open in the thesis: the VAE with deterministic decoding. Blind CMA
  at short block lengths, previously also open, is now measured in the
  partial-posterior sweep. Exact retraining of the canonical learned relays is
  not possible: their PyTorch initialization was unseeded (thesis Sec. 4.3.3).
  The canonical Hop-1 draws themselves are reproducible, because `Source(seed)`
  reseeds NumPy's global generator (`hybrid_branch_selection.py` replays them).
  See `thesis/CHANGELOG.md`.

---

## If verification fails

`make verify` exits 1 and prints each mismatch as
`table | cell | published | source | difference`.

- After re-running an experiment, a small difference within Monte-Carlo noise
  is expected; the tolerance should absorb it. A large difference means the
  thesis and the data genuinely disagree.
- If a table changed, update the LaTeX to match the data source, never the
  reverse.
- `make check` first if something looks structurally wrong — a missing results
  directory reports as a skip rather than a mismatch.
