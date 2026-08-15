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

1. **The thesis matches its data.** Every numerical table cell and every
   quantitative claim in the prose is compared against the file that produced
   it. This is `make verify`: **304 values, 0 discrepancies**, exit code 1 on
   any mismatch.
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
| 0 | `make test` | ~2 s | + pytest | 108 unit tests over channels, modulation, relays, statistics |
| 1 | `make repro-unknown` | ~40 min | numpy, scipy | Recomputes the whole unknown-channel study (Ch. 7), then verifies |
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
e6_*_ported.py                Unknown-channel study (Ch. 7), NumPy only
e6_qpsk_unknown_channel.py    QPSK unknown-channel study, both hop-2 variants
verify_thesis_tables.py       Checks the thesis against its data sources
results/                      Committed experiment outputs (JSON + figures)
e6_unknown_channel_results/   Committed unknown-channel outputs (.npy + figures)
tests/                        108 unit tests
thesis/                       LaTeX sources, figures, fonts, CHANGELOG
  main.tex                      Compile with XeLaTeX; includes the chapters below
  chapters/ch0N_*.tex           One file per chapter, numbered to match the document
                                (ch05 experiments, ch06 higher-order modulation,
                                 ch07 unknown channels, ch08 discussion, ch09 summary)
  chapters/_*.tex               Drafting history; not included by main.tex
scripts/build_bundles.py      Rebuilds the two Overleaf zips from thesis/
scripts/plot_e6_studies.py    Regenerates the Ch. 7 composite/blind/partial figures from .npy
scripts/strip_rev.py          Brace-matching remover for inline REV annotations
scripts/check_env.py          Environment doctor
```

---

## Determinism and caveats

- **Seeding.** Trial `t` under seed `s` initialises as `seed(s*1000 + t)`. The
  unknown-channel scripts reproduce their committed outputs bit-for-bit; this
  has been verified by re-running them from scratch and diffing every value.
- **Timing numbers will differ.** Tables 5.3 and 8.1 report wall-clock, which
  depends on your hardware. The verifier treats them as informational.
- **A GPU changes speed, not results.** The BER numbers are hardware
  independent; only the timing tables move.
- **Floating-point across platforms.** Different BLAS builds can perturb the
  last displayed digit. The verifier's rounding tolerance absorbs this.
- **What is *not* reproduced.** One experiment is deliberately not run and is
  flagged as open in the thesis: the VAE with deterministic decoding. Blind CMA
  at short block lengths, previously also open, is now measured in the
  partial-posterior sweep. See `thesis/CHANGELOG.md`.

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
