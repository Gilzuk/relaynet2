# Re-measurement record — August 2026

Every simulation result in the thesis was re-measured. This file records what
was re-run, what the numbers were before and after, and which written
conclusions changed as a result. It is separate from `CHANGELOG.md`, which
tracks editorial changes to the document; this one covers only the data.

Baseline for every comparison below is commit `c05611f`, the state of the
thesis before any re-run output landed.

---

## Why the results were re-measured

Three independent reasons, discovered in this order.

**1. A 3 dB error in the noise convention.** `relaynet/channels/awgn.py` set
`sigma = sqrt(noise_power)` on its real-valued branch instead of
`sqrt(noise_power/2)`, so the channel delivered 3 dB more noise than its label
claimed. Fixed in `97ca397`.

The fix reaches further than it appears. Every learned relay generates its
*training* data through `awgn_channel`
(`relaynet/utils/activations.py:214`), so correcting the channel moved the
trained weights — and therefore moved results on Rayleigh and on QPSK/16-QAM,
whose own evaluation paths the fix never touched. "The channel was not
modified, so the result stands" is false in this codebase.

**2. Tables resting on configurations the thesis no longer evaluates.** The
pairing rule — a real constellation on a real channel, a complex constellation
on a complex one — left three tables measured on pairings that had been
withdrawn. Each was found the same way: a default argument nobody had looked
at.

| Table | Was measured on | Cause |
|---|---|---|
| `tbl:table2` | BPSK / Rayleigh | real constellation on a complex channel |
| `tbl:table8` | BPSK / Rayleigh | `evaluate_relays` defaults to `modulation="bpsk"` |
| `tbl:table24` | 16-QAM / AWGN | `run_monte_carlo` defaults to `awgn_channel` when `channel_fn` is omitted |

**3. Result files older than the code that produced them.** `relaynet/relays/vae.py`
and `relaynet/relays/cgan.py` were both created on 2026-04-01, while the
results carrying those relays were dated 2026-03-23. Those numbers came from a
superseded implementation.

---

## What was re-run

Budget throughout: 10 trials × 10,000 bits per SNR point, SNR 0:2:20, matching
`ch04_methods.tex`. Runtimes are wall-clock on 4 CPU cores, no CUDA.

| Stage | Produces | Runtime |
|---|---|---|
| 7.1 channel validation | calibration figures | 5 s |
| 7.10 modulation | `tbl:table2`, `tbl:table14` | 2,892 s |
| 7.8 normalized-3K | `tbl:table8` | 747 s |
| 7.17 16-class | `tbl:table24` | 4,548 s |
| 7.11 activation | `tbl:table15` | 6,608 s |

`7.17` was run twice. The first pass put its twelve learned variants on
Rayleigh but left the AF and DF baselines on AWGN, because a *second*
`run_monte_carlo` call for the baselines also omitted `channel_fn`. It was
caught before transcription: the run reported AF at 0.00063 and DF at exactly
0.00000, against 0.0450 and 0.0375 for the same relays on the same channel in
`tbl:table14` — impossible, and the tell that the two halves of the table were
measured on different channels.

**Not re-run, deliberately:**

- **The cGAN**, dropped on instruction. Its column is absent from
  `tbl:table2`, `tbl:table14`, `tbl:table15` and `tbl:table24`, and the
  captions say so. It cannot be reinstated from the old numbers (superseded
  implementation, above), nor run alone and pasted in: training it consumes
  the shared RNG stream, so its presence shifts every other relay's numbers
  and the table would carry rows from two different draws.
- **`tbl:table13` and `tbl:table25`**, which report wall-clock timings
  measured on different hardware. The verifier treats them as informational.

---

## Numbers, before and after

### `tbl:table2` — canonical relay comparison

Reconfigured as well as re-measured: BPSK/Rayleigh (9 relays) → **QPSK/Rayleigh
(8 relays)**, QPSK having become the canonical constellation. Rows are not
comparable one-to-one; the whole table was replaced.

| Relay | 0 dB | 8 dB | 20 dB |
|---|---|---|---|
| AF | 0.4076 | 0.1852 | 0.0110 |
| DF | 0.3337 | 0.1218 | 0.0097 |
| MLP (169p) | 0.3349 | 0.1229 | 0.0099 |
| Hybrid | 0.3329 | 0.1220 | 0.0097 |
| VAE | 0.3359 | 0.1235 | 0.0097 |
| Transformer | 0.4014 | 0.1424 | 0.0101 |
| Mamba-S6 | 0.4071 | 0.1448 | 0.0100 |
| Mamba-2 SSD | 0.4032 | 0.1416 | 0.0098 |

The DF closed form was re-derived for the new constellation. Converting the
abscissa from `Es/N0` to `Eb/N0` (3.01 dB lower at k=2), `2P(1−P)` predicts
0.0098 at 20 dB against a measured 0.00972, and 0.0239 at 16 dB against
0.02451 — within 1% at both.

### `tbl:table8` — normalized 3K

BPSK/Rayleigh → **QPSK/Rayleigh**, and the study now takes its constellation
from `CANONICAL_PAIRS` rather than the `evaluate_relays` default.

### `tbl:table14` — 16-QAM

Was BPSK, QPSK *and* 16-QAM all on AWGN — two of those pairings withdrawn, the
third duplicating Chapter 5. Now **16-QAM on Rayleigh alone**, which states the
chapter's finding directly: AF and DF fall with SNR (DF 0.4197 → 0.0375) while
every learned relay flattens at 0.25–0.26.

### `tbl:table15` — output activation

AWGN + Rayleigh → **Rayleigh only**. The claimed benefit shrank: bounding the
output improves the sequence models by a factor of **2.4**, not the 5× reported
from the AWGN run (Transformer 0.2595 → 0.1073), narrowing their gap to DF from
3.1× to 1.3×. The feedforward relays are unmoved.

### `tbl:table24` — 4-class versus 16-class

16-QAM/AWGN → **16-QAM/Rayleigh**.

| Architecture | 4-class | 16-class | Ratio |
|---|---|---|---|
| MLP | 0.05491 | 0.03840 | 1.43× |
| VAE | 0.05382 | 0.03762 | 1.43× |
| Hybrid | 0.05362 | 0.03748 | 1.43× |
| Transformer | 0.05360 | 0.03814 | 1.41× |
| Mamba-S6 | 0.05421 | 0.03934 | 1.38× |
| Mamba-2 SSD | 0.05381 | 0.04953 | 1.09× |

On AWGN the 16-class heads reached ~0 and the improvement read as 51× for the
MLP. On Rayleigh the channel itself limits DF to 0.03748, so raw BER cannot
approach zero and the ratio is meaningless in isolation. The floor is properly
measured as **excess over DF**: 0.0161–0.0174 for the per-axis relays,
0.0001–0.0019 for the joint ones — a reduction of 95% or more for five of six.
The finding survives; its magnitude and the way it must be stated do not.

---

## Written conclusions that changed

**H4 — architecture convergence at equal scale.** Read "confirmed: at 3K
params, all methods within ~1% BER (except VAE)". Neither half survived. The
spread is SNR-dependent, falling monotonically from 17.4% at 0 dB to 4.1% at
20 dB; and at low SNR it is not a spread but a clean split by family, the
feedforward models at 0.336–0.342 near DF and the sequence models at
0.400–0.407 near AF. Since all six carry the same parameter budget this is
architecture mattering, not capacity. Now recorded as **confirmed at high SNR,
rejected at low**.

**The VAE failure.** The thesis reported the VAE "pinned at 0.25–0.40 at every
SNR, meaning it never learns the task". It does not reproduce: the VAE now
tracks DF, reaching 0.00972 at 20 dB against DF's 0.00972.

The obvious explanation was the 3 dB bug, since learned relays train through
the corrected channel. **A controlled A/B refutes it.** Evaluation runs on
Rayleigh, untouched by the fix, so the only difference between arms is the
convention used to synthesise training data; the patch was probe-verified to
bite (a nominal 10 dB arrives as 7.00 dB in the old arm, 10.01 dB in the new).
Both arms behave — 0.3349 and 0.3368 at 0 dB — so training on 3 dB-hot data
does not break the VAE. The cause is that `relaynet/relays/vae.py` postdates
the published numbers. Recorded in `results/vae_convention_ab.json`.

Chapter 2's conclusion, that the generative paradigm confers no advantage
here, survives and is better supported: the VAE arrives where a 169-parameter
regressor already is, at greater cost, rather than failing.

**16-QAM joint classification.** "Removes the structural floor, achieving
near-zero BER" becomes parity with DF — the floor is removed, but the endpoint
is the classical baseline rather than past it. This is H2 reappearing one
constellation higher.

---

## Verification

`verify_thesis_tables.py` checks every published cell against the file that
produced it: **349 cells, 0 inconsistencies**. Coverage was extended during
this work, and three checks were found to be silently disabled:

- `check_table8` aborted on a `KeyError` for `GenAI-3K` after the relay was
  renamed `MLP-3K`, losing 40 cells while reporting "skipped".
- `check_table14ray` failed the same way and lost all 68 of its cells.
- `check_table24` matched row labels literally, so `cGAN`, `Mamba-S6`,
  `Mamba-2 SSD`, `AF` and `DF` were never compared — five of nine rows
  unchecked, while the table reported OK on the four that were.

Name resolution is now central (`RELAY_ALIASES`, `resolve()`), row labels are
normalised, and a table that cannot cover a row reports partial coverage
instead of dropping it from the count. `tbl:table15` had never had any coverage
at all, which is why it sat on a pre-correction AWGN run unnoticed; it now has
24 cells.

## Reproducing

```
python3 run_experiments.py --exp 7.10 7.8 7.17 7.11 \
    --num-trials 10 --bits-per-trial 10000 --skip-relays cgan --retrain
python3 verify_thesis_tables.py
```

`CANONICAL_PAIRS` in `run_experiments.py` is the single definition of which
(constellation, channel) configurations are evaluated.
