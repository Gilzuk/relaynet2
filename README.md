# RelayNet2 — Learned and Classical Two-Hop Relay Communication

[![CI](https://github.com/Gilzuk/relaynet2/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilzuk/relaynet2/actions/workflows/ci.yml)

Code, experiment records and the M.Sc. thesis *Deep Learning Architectures for
Two-Hop Relay Communication* (Gil Zukerman, Tel Aviv University, 2026).
The system has one half-duplex relay and no direct source–destination path.

- **Read the thesis:** [PDF](thesis/main.pdf), [LaTeX source](thesis/main.tex).
- **Reproduce or inspect the evidence:** [researcher protocol](docs/round2_reproduction_protocol.md),
  [experiment ledger](memory-bank/table_provenance.md), [reproduction archive](results/reproduction/README.md).
- **Edit in Overleaf:** [generated thesis repository](https://github.com/Gilzuk/relaynet2-thesis),
  [build instructions](thesis/OVERLEAF.md).

## Scope and principal findings

The canonical comparison is **uncoded Gray-QPSK, two-hop SISO, i.i.d. per-symbol
Rayleigh fading and perfect receiver-side CSI**. It evaluates AF, symbol-wise DF
and six learned configurations. The coherent front end uses CSI; the learned
canonical relays receive zero-forced samples without explicit channel magnitudes.
Separate studies change the channel, receiver information or coding. They include
BPSK and faded QPSK with ISI, nonlinear composite channels, finite-pilot and blind
receivers, and coded QPSK/16-QAM controls. These studies are not a single controlled
ablation. The older standalone 16-QAM classifier extension is outside the thesis.

| Comparison | Supported conclusion | Evidence |
|---|---|---|
| Canonical uncoded relays | No learned advantage over DF is resolved in the reported paired tests. At 8 dB, historical DF BER is 0.1218 versus MLP 0.1229. Against conventional finite-power AF, the MLP's 0-dB margin is only 0.003–0.007, depending on gain rule. | Tables 5.2–5.3; [canonical data](results/modulation/qpsk_rayleigh.json); [AF controls](results/af_block_and_conventional.json) |
| Matched coded decoding | With reliability-weighted metrics, block DF has lower BER than **both hard and soft MLP read-outs**, at all six SNRs and in all 30 paired trials per point. Matched Mamba was **not evaluated**. The earlier high-SNR MLP advantage concerns the unweighted decoder. | Principal §5.5.1; [historical matched data](results/coded_matched_metrics.json); [additional-run assessment](results/round2_rerun_assessment.json) |
| Fixed three-tap BPSK ISI | The 170-parameter MLP improves strongly on the evaluated AF/zero-threshold DF, but trails matched genie MLSE over the resolved range: 0.0065 versus 0.001354 at 8 dB. MLP validation ends at 16 dB; its 18–20-dB tail remains unvalidated. | Chapter 6; [matched protocol](results/e6_matched_protocol.json); [fixed-budget tail](results/e6_matched_highsnr.json) |
| Finite pilots and blind processing | At 10 dB the MLP trails LS+MLSE at 20 pilots, is not resolved from it at 10, and leads at 5. In the separate blind study it leads the evaluated CMA through 12 dB, with no resolved difference from 14 dB. | Figure 6.5; [pilot/blind methods](thesis/chapters/ch07_unknown_and_mismatch_channels.tex) |
| Computation on the real-tap coded-QPSK channel | Exact I/Q factorization gives two binary MLSE trellises: 16/64/256 multiplications per symbol at 3/5/7 taps. The separate 220-parameter MLP uses 208 MACs; the arithmetic crossover is about 6.7 taps. MLSE has lower measured BER. | Main Tables 6.8–6.9; [factorization check](factorized_mlse_equivalence.py) |

These are configuration-specific measurements. They do not establish universal
learned-relay superiority, overfitting as a mechanism, a universal minimum network,
new-family generalization, or hardware latency/energy savings. Training-family
knowledge is retained in the learned weights. MLSE minimizes sequence error;
the BCJR control measures **relay-output** BER only, not end-to-end BER.

A zero observed error count is not zero true BER. Similar rounded values,
unchanged rankings and failure to reject a paired test do not establish statistical
equivalence. Reported paired tests are exploratory and uncorrected for multiple
comparisons; fixed-checkpoint Monte Carlo intervals exclude retraining variance.

## Reproduction: choose the intended operation

Use a full git checkout and record `git rev-parse HEAD`. The following environment
matches the **October validation**; it does not recover missing historical versions.
Linux is required by the coded driver's multiprocessing `fork`. The archived run's
CPU, BLAS, package and thread records remain part of its execution context.

```bash
python3.12 -m venv .venv-round2
. .venv-round2/bin/activate
python -m pip install -r requirements-round2.txt
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python --version
```

The recorded Python version is **3.12.14**; the dependency file pins NumPy 2.3.5,
SciPy 1.17.0, Matplotlib 3.10.8 and pytest 9.1.1. CUDA is not required. The older
`requirements-repro.txt` describes a different environment.

### 1. Check committed evidence without training

```bash
python scripts/assess_round2_reruns.py --check
python scripts/audit_round2_models.py --check
python verify_thesis_tables.py
python provenance_audit.py
```

The numerical verifier currently checks **892 cells/prose values** against their
declared artifacts. The model audit verifies **48 saved checkpoints**. These checks
have defined coverage; they do not certify every sentence or scientific conclusion.
The [protocol](docs/round2_reproduction_protocol.md) maps each experiment to its
architectures, scaling, training/evaluation budgets, RNG policy and output files.

### 2. Train additional round-2 realizations with one script

```bash
python scripts/reproduce_round2.py --list
python scripts/reproduce_round2.py --run-id independent-round2-001 --jobs sequence_memory coded_matched canonical_native canonical_3k --threads 2 --timeout-hours 5.5 --background
cat results/reproduction/independent-round2-001/status.json
```

Choose a **new run ID** for each attempt. The script freezes committed source,
uses isolated worktrees, and retains logs, comparisons and available weights.
Local jobs are sequential; `--background` detaches the launcher but cannot survive
host shutdown. The coded job can fork six SNR workers, so two numerical threads
per worker is not a two-core cap on the whole run. The timeout is a limit, not a
runtime estimate. See each job's `job.json`, `comparison.json` and logs when
classifying completion: a learned-result difference is not a worker crash.

The focused runner declares Python/NumPy/Torch bootstrap seed **20261017**, requests
deterministic Torch algorithms, and preserves original driver-local seeds.
Canonical training data use `42 + training SNR`; evaluation trials use 0–9.
Sequence initialization IDs are 0/1/2; coded initialization is 0 with shuffle seed
42. A default data seed of 42 alone never established deterministic training.

### 3. Inspect or execute the broader thesis plan

```bash
python scripts/reproduce_all.py                     # plan only; no experiments
python scripts/reproduce_all.py --json              # machine-readable plan
# Full execution is expensive; run only when the host/budget is appropriate:
python scripts/reproduce_all.py --run --run-id independent-full-001 --seed 42 --threads 2
```

This runner has **73 entries: 68 producers, two archive analyses and three
historical archive references**. Its default global bootstrap seed is **42**, while
driver-local seeds still take precedence. It retains per-step outputs, hashes,
RNG states and available checkpoints, rejects stale/missing outputs, and propagates
failures. Partial plans may use committed upstream inputs; those inputs are recorded.
The complete plan has **not** been executed end to end. A plan entry or an archived
reference is not evidence of a new experiment. See the [runner review](docs/full_runner_review_2026-10-17.md).

### What can and cannot be reproduced

| Evidence | Status and limit |
|---|---|
| Historical canonical Tables 5.2–5.3 | Actual architectures, training paths, scaling and budgets are documented in §4.3.3/Table 4.2 and [canonical provenance](results/canonical_run_provenance.json). Historical Torch RNG states, learned weights, library versions and epoch losses are unavailable. Exact learned replay is not possible from the retained record. |
| Classical canonical controls | AF/DF per-trial arrays reproduce exactly under the checked protocol. This does not certify learned replay. |
| Four October hosted jobs | All workers completed. Learned outputs differ from the historical records; classical controls match. [Assessment JSON](results/round2_rerun_assessment.json) preserves numerical and statistical discrepancies. These are additional training-realization/seed-sensitivity observations, not proof that the originals were incorrect or that every difference was caused by a changed initial seed. |
| Saved weights | Twelve new canonical and 36 sequence checkpoints are retained. The hosted coded classifier weights were not saved; only its trial results and configuration are available. |
| Explicit seed-42 control | Two full-budget native MLP controls matched initial/final weights and all 110 trial BERs on one host. This does not establish every-model or cross-platform bitwise repeatability. |
| Secondary studies | Pilot sweeps, the full minimum-size training grid, adaptive rare-event tails and matched Mamba were not newly executed by the four-job validation. |

Immutable references: experimental/derived-result snapshot
[`80b4545adcc6`](https://github.com/Gilzuk/relaynet2/tree/80b4545adcc66e4be1f435364f7ec927e87f6414);
hosted execution source
[`c0394f0da5da`](https://github.com/Gilzuk/relaynet2/tree/c0394f0da5dac449594b89a9794fe251fad75e76);
hosted evidence archive
[`dc7c02c00cbe`](https://github.com/Gilzuk/relaynet2/tree/dc7c02c00cbebdfb566362a810ab4b45cb6abbe0/results/reproduction).
Later RNG instrumentation does not apply retroactively to that hosted run.

## Implemented relay models

Counts below are the native framework definitions, not every study's architecture.
The canonical experiment excludes cGAN; the ISI and coded MLPs have different input,
hidden and output dimensions. Per-study settings take precedence over this overview.

| Relay | Native parameters | Implementation / context |
|---|---:|---|
| AF / symbol-wise DF | 0 | Block-normalized amplification / constellation detection |
| MLP / Hybrid | 169 each | 5→24→1; Hybrid has its own MLP and a fixed-threshold blockwise switch |
| VAE | 1,777 | Seven-sample window; latent sampling during training, latent-mean decoding at inference |
| Transformer | 17,697 | 11-sample window, width 32, four heads, two layers; attention within the window |
| Simplified Mamba-S6 | 24,001 | Width 32, state size 16, two layers; causal scan read at the centre of an 11-sample window |
| Simplified Mamba-2 SSD | 26,179 | Width 32, state size 16, two layers; causal centre readout |
| cGAN prototype | 2,946 | WGAN-GP; includes training-only critic parameters; excluded from the canonical tables |

The two Mamba implementations are simplified research implementations, not the
reference fused Mamba kernels. Their centre readout uses **six samples up to the
current sample**, versus all eleven for the canonical Transformer. S6 omits the
reference convolution/gate and has an unused C-projection still included in its
nominal parameter count. Sharing a window length or parameter budget does not
isolate architecture, capacity or optimization effects.

Local causality is distinct from transmission causality: block power normalization
couples transmitted symbols across the whole block, and Hybrid also selects its
branch from a block statistic. The canonical learned relays process each QPSK
axis without input rescaling (±1/√2 versus training ±1), normalize each output axis
to unit power, and use measured forwarded power to set Hop-2 noise. See §4.1.5 and
§4.3.3 for the exact normalization and SNR conventions.

## Framework examples and historical experiments

`run_experiments.py` is the **legacy framework runner**, with 18 registered jobs.
Its numeric identifiers such as `7.10` and `7.17` are experiment IDs, **not current
thesis section numbers**. It is distinct from the full-thesis and focused round-2
runners above. Quick mode changes budgets and cannot reproduce published tables.

```bash
python run_experiments.py --list
python run_experiments.py --exp 7.2 --quick
```

A small reproducible CPU example, uncoded BPSK/AWGN:

```python
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import compute_confidence_interval
from relaynet.channels.awgn import awgn_channel

relay = DecodeAndForwardRelay(prefer_gpu=False)
snrs, means, trials = run_monte_carlo(
    relay, snr_range=range(0, 9, 2), channel_fn=awgn_channel,
    num_bits_per_trial=10000, num_trials=10, seed_offset=0,
    modulation="bpsk",
)
lo, hi = compute_confidence_interval(trials)
for snr, ber, lower, upper in zip(snrs, means, lo, hi):
    print(f"SNR={snr:2d} dB  BER={ber:.4e}  CI=[{lower:.4e}, {upper:.4e}]")
```

These are across-trial Student-t intervals, not a rare-event confidence procedure.
The framework BPSK bit map is 0→−1, 1→+1; its QPSK map uses 0→+1 on each axis.
Individual experiment drivers can use their own bit labels. Preserve their mapping,
call order and seeds for sample-level replay.

Older framework demonstrations remain in [results/](results/), [checkpoints/](checkpoints/)
and git history. In particular, [16-QAM classifier data](results/all_relays_16class/all_relays_16class.json)
are outside the current thesis. The former README's 20-dB classifier numbers do
not match that current JSON and are not repeated as current results. Finite-SNR
values do not prove an irreducible error floor; separate I/Q classification is not
inherently a structural limitation for separable square-QAM likelihoods. Neither
zero observed errors nor rounded equality supports infinite improvement or
statistical equivalence to DF. Historical development labels such as “overfitting
found” and “best parameter efficiency” are not established thesis conclusions.

The old `e6_unknown_channel_results/e6_viterbi_awgn.npy` contains a superseded
real-noise convention and is retained for provenance. Use the matched-protocol
JSON files linked above for the published MLSE comparison. Re-running legacy
scripts in place can overwrite historical outputs; use the isolated runners for
reproduction evidence.

## Thesis build and repository layout

| Location | Purpose |
|---|---|
| `relaynet/` | Channel, modulation, relay, simulation and statistics library |
| `run_experiments.py`, `e6_*.py`, `coded_*.py` | Experiment producers; their individual settings define each study |
| `scripts/reproduce_round2.py`, `scripts/reproduce_all.py` | Isolated execution and evidence retention |
| `results/`, `e6_unknown_channel_results/` | Current and historical records; consult the ledger before comparison |
| `thesis/chapters/` | Chapters 1–4 methods/background; 5 core experiments; 6 unknown/mismatched channels; 7–8 discussion/summary; 9 appendices 9.1–9.7 |
| `thesis/main.pdf`, `thesis_preview.pdf` | Built thesis and identical top-level preview |
| `thesis_overleaf.zip`, `thesis_overleaf_clean.zip` | Generated self-contained Overleaf packages with bundled fonts |
| `tests/`, `docs/`, `memory-bank/` | Tests, review/protocol records and project history |

Build with **XeLaTeX** (including Hebrew support):

```bash
cd thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
cp thesis/main.pdf thesis_preview.pdf
python scripts/build_bundles.py
```

In Overleaf select XeLaTeX and `main.tex`. Both bundles derive from the same active
source graph; no revision annotations remain in the current manuscript. The
[thesis mirror](https://github.com/Gilzuk/relaynet2-thesis) is generated by
`scripts/overleaf_project.py` / `scripts/overleaf_sync.py`; edit this source repository
and regenerate it. Do not edit the mirror directly. A linked Overleaf project may
still need a GitHub pull after the mirror updates.

The PDF has **119 Arabic-numbered body pages and 135 physical pages**, including
13 front-matter and three Hebrew back-matter pages. These are separate counts;
formal acceptance of the faculty's page-count convention is an administrative matter.

## Tests

```bash
python -m pytest tests/ -q
```

Use the reported pass/skip counts for the checked-out revision and installed
dependencies. Tests cover channels, modulation, relays, simulation, statistical
analyses, evidence checks and reproduction orchestration. They do not substitute
for full-budget experiment execution or scientific review.

## Appendix — Proof of Claims

Each headline claim of the unknown-channel study, verified against theory and primary literature. The fixed-ISI channel is the normalized 3-tap FIR $h = [1, 0.7, 0.5]/\lVert\cdot\rVert \approx [0.758, 0.531, 0.379]$ (`e6_viterbi_ported.py`).

### Claim 1 — Local bit-MAP is not an end-to-end relay benchmark

Bit-MAP minimizes expected bit error under a specified model, observation and side information. Symbol-MAP is not generally bit-MAP, and Viterbi MLSE minimizes sequence error [1, 2, 16]. The QPSK BCJR control measures **relay-output** BER, not destination BER; it does not close an end-to-end BCJR-versus-MLP comparison.

### Claim 2 — The selected zero-threshold slicer has a 0.25 limit

For the selected BPSK taps, the cursor is approximately 0.758 and the interferer sum is 0.910. One of four patterns gives -0.152; the others preserve the sign. This gives the zero-threshold rule a 1/4 limit, not a bound for every memoryless detector. Zero-amplitude ties for other taps contribute one half.

### Claim 3 — Separate MACs and FLOPs

The real 11-to-13-to-1 relay has 170 parameters, 156 MACs and 14 bias additions, approximately 330 FLOPs excluding activation cost. Full-state joint MLSE has M^L branches for channel length L; with real taps a square M-QAM trellis separates exactly into two √M-ary trellises (2M^(L/2) branch metrics, 2^(L+1) for QPSK; `factorized_mlse_equivalence.py`). Reduced-state alternatives [9] are not benchmarked. The coded-QPSK 208-MAC relay is a different, 220-parameter architecture.

### Claim 4 — Blind results are implementation-specific

The random nonlinear composite cascade cannot be characterized by the zeros of one fixed filter. CMA and decision-directed MLSE results concern the tested algorithms and adaptation budgets, not universal convergence guarantees [10–13].

### Claim 5 — Pilot count changes measured estimation quality

Five observations can identify three taps if the pilot design has full column rank. The measured degradation is not proof of rank deficiency. Standard LS covariance assumes a correctly specified linear model [14]; the composite cascade is nonlinear. Other few-pilot estimators [15] remain outside this comparison.

### Claim 6 — The matched gap is target-dependent

Corrected BPSK required-SNR penalties are 0.26, 1.18, 1.88 and 2.53 dB at BER targets 1e-1 through 1e-4. No windowed bit-MAP ablation separates window, capacity, training and optimization effects. Historical fixed-offset claims are superseded.

### References

1. G. D. Forney, Jr., "Maximum-likelihood sequence estimation of digital sequences in the presence of intersymbol interference," *IEEE Trans. Inf. Theory*, vol. 18, no. 3, pp. 363–378, 1972.
2. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed. McGraw-Hill, 2008.
3. N. Farsad and A. Goldsmith, "Neural network detection of data sequences in communication systems," *IEEE Trans. Signal Process.*, vol. 66, no. 21, pp. 5663–5678, 2018.
4. N. Shlezinger, Y. C. Eldar, N. Farsad, and A. Goldsmith, "ViterbiNet: A deep learning based Viterbi algorithm for symbol detection," *IEEE Trans. Wireless Commun.*, vol. 19, no. 5, pp. 3319–3331, 2020.
5. M. Honkala, D. Korpi, and J. M. J. Huttunen, "DeepRx: Fully convolutional deep learning receiver," *IEEE Trans. Wireless Commun.*, vol. 20, no. 6, pp. 3925–3940, 2021.
6. H. Ye, G. Y. Li, and B.-H. Juang, "Power of deep learning for channel estimation and signal detection in OFDM systems," *IEEE Wireless Commun. Lett.*, vol. 7, no. 1, pp. 114–117, 2018.
7. T. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," *IEEE Trans. Cogn. Commun. Netw.*, vol. 3, no. 4, pp. 563–575, 2017.
8. R. W. Lucky, J. Salz, and E. J. Weldon, *Principles of Data Communication*. McGraw-Hill, 1968.
9. M. V. Eyuboğlu and S. U. H. Qureshi, "Reduced-state sequence estimation with set partitioning and decision feedback," *IEEE Trans. Commun.*, vol. 36, no. 1, pp. 13–20, 1988.
10. D. N. Godard, "Self-recovering equalization and carrier tracking in two-dimensional data communication systems," *IEEE Trans. Commun.*, vol. 28, no. 11, pp. 1867–1875, 1980.
11. J. R. Treichler and B. G. Agee, "A new approach to multipath correction of constant modulus signals," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 31, no. 2, pp. 459–472, 1983.
12. Z. Ding, R. A. Kennedy, B. D. O. Anderson, and C. R. Johnson, Jr., "Ill-convergence of Godard blind equalizers in data communication systems," *IEEE Trans. Commun.*, vol. 39, no. 9, pp. 1313–1327, 1991.
13. R. Raheli, A. Polydoros, and C.-K. Tzou, "Per-survivor processing: A general approach to MLSE in uncertain environments," *IEEE Trans. Commun.*, vol. 43, no. 2/3/4, pp. 354–364, 1995.
14. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*. Prentice Hall, 1993.
15. S. Park, H. Jang, O. Simeone, and J. Kang, "Learning to demodulate from few pilots via offline and online meta-learning," *IEEE Trans. Signal Process.*, vol. 69, pp. 226–239, 2021.
16. L. R. Bahl, J. Cocke, F. Jelinek, and J. Raviv, "Optimal decoding of linear codes for minimizing symbol error rate," *IEEE Trans. Inf. Theory*, vol. 20, no. 2, pp. 284–287, 1974.

---

## Citation and license

```bibtex
@mastersthesis{zukerman2026relay,
  title={Deep Learning Architectures for Two-Hop Relay Communication:
         A Comparative Study of Classical and Neural Network Relay Strategies},
  author={Zukerman, Gil},
  school={Tel Aviv University},
  year={2026},
  url={https://github.com/Gilzuk/relaynet2}
}
```

`setup.py` declares an MIT license classifier, but this revision contains no
standalone license text. The former README link to `LICENSE` was broken.
