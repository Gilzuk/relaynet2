# Reproducing the thesis

This repository preserves the source, saved results and analysis tools for
*Deep Learning Architectures for Two-Hop Relay Communication*. The verifier
checks a declared subset of the thesis against those results. Recomputing a
saved comparison and retraining an original neural checkpoint are different
tasks; exact canonical retraining is not possible from the retained record.

For the four October validation jobs, use the [researcher protocol](docs/round2_reproduction_protocol.md): exact environment, one-script launch, per-study seeds/budgets, weight inventory and explicit missing records. `requirements-round2.txt` is the environment of those jobs; the legacy setup below uses a different dependency set.

```bash
git clone https://github.com/Gilzuk/relaynet2
cd relaynet2
make setup      # install pinned dependencies
make check      # what can this machine run?
make verify     # check the declared numerical coverage against saved data
```

`make verify` takes about a minute, needs nothing but NumPy and SciPy, and is
the fastest check of the document's declared numerical coverage. Use the
immutable revision linked in thesis Section 4.3.4 for the archived code/data;
the merged PR also contains the final manuscript and rebuilt bundles.

---

## What "reproducible" means here

Two distinct claims, checkable independently:

1. **The thesis matches its data.** A declared subset of numerical table cells
   and prose numbers is compared against the file that produced it; it does not
   cover every number or figure. This is `make verify`: **892 values,
   0 discrepancies** (2026-10-17), exit code 1 on any mismatch.
2. **Experiments can be rerun under stated limits.** Seed formulas and budgets
   belong to each driver. `--seed 42` does not seed every PyTorch RNG in the
   canonical driver. Original learned checkpoints and library versions were
   not retained in the archive; AF/DF evaluations reproduce bit for bit, but
   this does not establish exact learned-column reproduction. Later replicated
   studies have their own seed controls and cross-machine limitations.

---

## Tiers

Pick by how much compute you have. Each tier subsumes the checks of the one
above it.

| Tier | Command | Time | Needs | What it establishes |
|---|---|---|---|---|
| 0 | `make verify` | ~1 min | numpy, scipy | Declared numerical coverage matches saved data |
| 0 | `make test` | depends on environment | + pytest | Implementation and analysis regression tests; optional Mamba-2 tests may skip |
| 1 | `make repro-unknown` | budget dependent; potentially hours | numpy, scipy | Runs six legacy BPSK drivers, then verifies; does not run later fixed-budget confirmations or every Chapter 6 study |
| 1b | `make repro-qpsk` | ~20 min | numpy, scipy | Recomputes the QPSK unknown-channel study (ISI → AWGN and → Rayleigh) |
| 2 | `make repro-full` | hours | **+ torch** | Runs the legacy driver, then verifies; revision studies below are separate |

### Tier 0 — verification

```bash
make verify
```

`verify_thesis_tables.py` reads the LaTeX sources in `thesis/chapters/`,
extracts the registered table cells and prose values, and compares them against
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

This regenerates the six legacy `.npy` outputs and re-verifies with a Monte-Carlo
tolerance. It overwrites those outputs; use a separate checkout when comparing
with the archive. The legacy ISI driver includes an adaptive rare-event tail
with a large exposure limit, not the later predeclared fixed-budget validation.
Use the experiment ledger and `provenance_audit.py` for those separate drivers.
No torch required: the unknown-channel relays (AF, DF, the minimal MLP, Viterbi
MLSE, CMA) are pure NumPy.

### Tier 2 — run the legacy experiment driver

```bash
make repro-full
```

Requires PyTorch, for the VAE, CGAN, Transformer, Mamba S6 and Mamba-2 relays.
It may reuse available cached weights and includes experiments not reported in
the thesis. It is not an exact recreation of Tables 5.2–5.3. Their actual
commands, code/data commits, training and evaluation budgets are in Table 4.2
and ledger C4–C5. Inspect `--list` and select the intended experiment.

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
python coded_matched_metrics.py           # Table 5.4: reliability-weighted coded metrics (CMM_TRIALS, default 30)
CODED_METRIC=matched CODED_TRIALS=30 python coded_error_mechanism.py    # main matched discussion; Table 9.9 is unweighted
CODED_METRIC=matched CODED_TRIALS=30 python coded_reliable_regime.py    # main matched discussion; Table 9.10 is unweighted
CODED_METRIC=matched CODED_TRIALS=30 python coded_k_sweep_qpsk.py       # constraint-length paragraph
CODED_METRIC=matched CODED_TRIALS=30 python coded_k_sweep_qam16.py      # Table 5.5, matched columns
CODED_METRIC=matched CODED_TRIALS=30 python coded_rate_adaptation.py    # Tables 5.7 and 5.8
CODED_METRIC=matched CODED_TRIALS=30 python coded_latency_throughput.py
CODED_METRIC=matched CODED_TRIALS=30 python coded_latency_capacity.py
python coded_matched_envelopes.py         # matched block-DF envelopes, derived (no simulation)
CODED_METRIC=matched python coded_min_size_matched.py  # matched block DF only: 10 trials x 100 frames
python af_block_and_conventional.py       # AF block length and conventional variable-gain AF
python factorized_mlse_equivalence.py     # two binary trellises vs the joint QPSK trellis
python e6_scalar_map.py                   # scalar MAP relay on the ISI channel
python e6_isi_penalty.py                  # ISI required-SNR penalties, derived from e6_matched_protocol.json
python e6_flat_ported.py                  # flat controls with the corrected Hop-2 noise
python hybrid_branch_selection.py         # which branch the canonical Hybrid took (replay)
```

The scripts that import `coded_metric` select the unweighted pipeline by default;
with `CODED_METRIC=matched` they write `*_matched.json`. The size sweep retains
the saved learned curves with unweighted destination decoding and reruns only
the block-DF baseline. The adaptation envelopes are derived from saved data.
The standalone high-budget and soft-decision diagnostics do not implement that
environment switch; the principal hard/soft matched comparison is C10.

Round-2 arithmetic and audit checks require no retraining:

```bash
python penalty_common_targets.py          # common-target penalties and censored summaries
python canonical_paired_tests.py          # paired tests from saved canonical trials
python canonical_run_provenance.py        # AF/DF replay and architecture inventory; needs torch
python scripts/plot_e6_complexity_cost.py  # factorized arithmetic panel, no simulation
python plot_seed_stability.py             # saved-curve plots with censored penalties marked
```

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
tests/                        Implementation and evidence checks (pytest)
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

- **Seeding.** There is no project-wide seed formula. Canonical evaluation
  uses trials 0–9 at every SNR; other drivers use their own formulas. Read the
  archived driver and result metadata for the experiment being repeated.
- **Timing numbers will differ.** Table 5.6 and Figure 6.7(b) report
  wall-clock, which depends on your hardware. The verifier checks Table 5.6
  only against the readouts committed beside it, not against your machine.
- **Hardware can affect numerical results.** Fixed data and weights remove
  training variation, but floating-point/backend differences can still change
  a decision close to a boundary. Bit identity must be checked, not assumed.
- **Floating-point across platforms.** BLAS/backend differences can affect
  optimization and decisions; the effect is not guaranteed to stay in the last
  displayed digit. Preserve discrepancies rather than expanding a tolerance.
- **Repeatability is an empirical check, not a seed guarantee.** Earlier
  tested configurations repeated on one host but differed on another. The
  later full-budget MLP seed-42 control repeated exactly on its host. The new
  sequence study also differs for the MLP, despite repeated seed IDs. Preserve
  every seed curve and compare common-target penalties, censored cases and
  spread by execution environment; best-of-three agreement alone is insufficient.
  These differences do not by themselves invalidate the original measurements.
- **Canonical inference and missing records.** The VAE already decodes its
  latent mean deterministically; sampling is used in training, not evaluation.
  Exact retraining of the canonical learned relays is not possible: PyTorch
  RNG state, the original checkpoints and historical versions are unavailable
  in the archive (thesis Sec. 4.3.3). Later dependency pins do not recover them.
  The canonical Hop-1 draws themselves are reproducible, because `Source(seed)`
  reseeds NumPy's global generator (`hybrid_branch_selection.py` replays them).
  See `thesis/CHANGELOG.md`.

---

## If verification fails

`make verify` exits 1 and prints each mismatch as
`table | cell | published | source | difference`.

- First distinguish a missing/corrupted historical input from a new training
  realization. A new-run difference is not automatically a transcription error
  or evidence that the original measurement was wrong.
- Preserve both records and inspect run configuration, RNG policy, environment
  and comparison fields. Change manuscript claims only with identified evidence;
  never overwrite historical results or tune tolerances to force agreement.
- Report missing inputs and skipped checks explicitly; an incomplete check does
  not establish a pass for the omitted evidence.
