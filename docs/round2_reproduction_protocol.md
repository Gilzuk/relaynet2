# Round-2 reproduction protocol for independent researchers

This companion separates checking saved evidence, evaluating saved weights, and
training another realization. It does not claim recovery of the missing historical
canonical weights. Use the commit containing this document and record its full
`git rev-parse HEAD`; PR #184 is a moving review branch, not an immutable identifier.
The reviewed starting revision is `d29f8ee96664e5df8baa21a7e02e2a91ca9c665d`.

## Environment and commands

Use a full git checkout on Linux (the coded driver requires multiprocessing
`fork`), Python **3.12.14**, and CPU execution. These are the additional validation
environment, not recovered historical versions. The exact hosted package/CPU/BLAS
records are in each archived run's `environment.json`.

```bash
python3.12 -m venv .venv-round2
. .venv-round2/bin/activate
python -m pip install -r requirements-round2.txt
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python scripts/assess_round2_reruns.py --check
python scripts/audit_round2_models.py --check
python verify_thesis_tables.py
python provenance_audit.py
```

Check the patch version with `python --version`. The dependency file specifies
NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and pytest 9.1.1. The separate legacy
`requirements-repro.txt` is a different dependency set; it does not recreate the
hosted environment. BLAS, CPU, package builds and thread behavior can still differ.
No universal same-seed bit-identity guarantee follows from these pins.

The first two commands verify saved artifacts and recompute analysis/model checks;
they do not train. The model audit loads 48 checkpoints with strict state-dict
matching and restricted `weights_only=True` loading. Archived sequence metadata
contains Torch's `TorchVersion` string subclass, for which the script provides a
specific allowlist; unrestricted pickle loading is unnecessary.

To run new full-budget experiments with the current explicit RNG policy:

```bash
python scripts/reproduce_round2.py --list
python scripts/reproduce_round2.py --run-id independent-round2-001 --jobs sequence_memory coded_matched canonical_native canonical_3k --threads 2 --timeout-hours 5.5 --background
cat results/reproduction/independent-round2-001/status.json
```

Choose a new run ID for every attempt. The launcher freezes the committed HEAD,
creates isolated worktrees and saves outputs under that ID. It must run on a host
that stays alive; detaching from a shell is not persistence through host shutdown.
Jobs execute sequentially locally; the archived hosted workflow ran them in
parallel. The coded job forks up to six SNR workers, so two numerical threads per
worker is not a two-core cap on the entire experiment. Runtimes and peak memory are
host dependent; peak memory was not retained as a benchmark.

## Exact experiment mapping

Historical source and data commits for the two canonical tables are recorded in
[`canonical_run_provenance.json`](../results/canonical_run_provenance.json).
The hosted additional runs executed **c0394f0da5dac449594b89a9794fe251fad75e76**;
their immutable archive is **dc7c02c00cbebdfb566362a810ab4b45cb6abbe0**.
Later deterministic-algorithm flags and RNG snapshots do not apply retroactively
to that hosted execution. Reproducing its source requires checking out c0394f0;
that will use its older RNG policy, not the current one.

| Job | Training configuration | Evaluation | Historical reference / new output |
|---|---|---|---|
| `canonical_native` | Six learned models, 100 epochs; architecture, window, sample count, SNR, learning rate, batch and checkpoint rule per Table 4.2 and the provenance JSON. BPSK AWGN training; QPSK zero-forced Rayleigh deployment. | 0–20 dB in 2-dB steps; 10 trials × 10,000 bits; seeds 0–9 at every SNR. | `results/modulation/qpsk_rayleigh.json` |
| `canonical_3k` | Six nominally 3K models; 25,000 requested samples, 24,969 interior windows; sequence gradient subset 21,224; 100 epochs. Other settings per provenance JSON. | Same canonical grid, bits and seeds. | `results/normalized_3k/3k_rayleigh.json` |
| `sequence_memory` | Four architectures × three channels × seeds 0,1,2. 40 epochs, final weights, MSE, Adam 0.001, batch 64, no scheduler or clipping. 6,666 symbols at each of 5/10/15 dB: 19,998 BPSK windows or 39,996 QPSK axis windows. | 0/4/8/12/16/20 dB; 20 trials × 20,000 bits; seeds 0–19. | Historical `results/seq_models_on_memory.json`; new `results/reproduction_seq_memory.json`; corrected penalties in `results/penalty_common_targets.json`. |
| `coded_matched` | One NumPy float64 classifier, 42→16→4, tanh/softmax, 756 parameters. 2,000 frames per SNR at 5/10/15 dB, 202 symbols/frame: 1,212,000 windows. Cross-entropy, 25 epochs, batch 512, Adam 0.003, betas 0.9/0.999, epsilon 1e-8; final weights, no holdout/scheduler/clipping. | Six SNRs as above; 30 trials × 500 frames × 200 information bits; seed `800000 + 1000*SNR + trial`. | `results/coded_matched_metrics.json`; hard/soft MLP and classical relays, no matched Mamba. |

Canonical Adam is configured with learning rate only; other options were library
defaults. Historical library versions are missing, so exact historical default
implementation behavior is not asserted. In the retained Torch 2.8.0 environment,
Adam defaults include betas (0.9, 0.999), epsilon 1e-8, weight decay 0 and AMSGrad
off. Only canonical Mamba2 clips gradients (norm 1). Canonical sequence wrappers
hold out 15% of windows, but select on **training** loss with improvement >1e-5,
no early stopping. MLP/Hybrid/VAE and the separate sequence-memory loop use final
weights. Canonical nominal sample counts are floored per training SNR before
interior windows are extracted; the provenance JSON gives actual counts.

## RNG, inputs and inference details that affect a replay

- The current suite bootstraps Python, global NumPy and Torch with 20261017,
  sets `PYTHONHASHSEED=0`, disables CUDA, fixes numerical threads and requests
  deterministic Torch algorithms. Job-local seed rules still take precedence.
- Canonical training data use seed `42 + integer training SNR`. The historical
  canonical Torch state is unknown. A newly chosen Torch seed is not that state.
- Sequence construction and training reset Torch to 0, 1 or 2. Training payload
  generators use 1000, 1001 or 1002. Hop-1 objects begin at constructor seed 1;
  their state is consumed by the original baseline/AF/training/evaluation order.
  Preserve the entire job order, rather than reseeding each channel call.
- Coded classifier initialization and training payload/fading/noise use seed 0;
  batch shuffling uses `default_rng(42)`. Repeated seed labels in later runs are
  not independent new initialization samples.
- Canonical training is ±1 BPSK; evaluation supplies QPSK axes ±1/√2 without
  rescaling. Each learned output axis is normalized separately to unit block
  power; the combined complex power is 2. Hop-2 noise uses measured power.
  AF/DF act on complex symbols and have power 1. A block-wide normalization is
  distinct from the neural local map and prevents symbol-causal transmission.
- Canonical training uses interior windows and evaluation repeats edge samples.
  Sequence-memory training zero-pads, while inherited inference repeats edges.
  S6/SSD center readouts use the past half of the window; the Transformer uses
  unmasked attention. Mamba implementations are simplified, not reference kernels.
- The coded classifier concatenates the full I window then the Q window; it does
  not alternate I/Q entries. Window length is 21, with zeros only at the ends of
  each concatenated SNR training stream, not at every frame. The code is rate 1/2,
  K=3, octal generators (7,5), zero-tail terminated by two bits. Hard output is
  the four-class argmax; soft output is the probability-weighted constellation
  mean normalized over the block. The weighted destination metric models the
  second-hop channel, not the full distribution of relay decision errors.
- VAE evaluation decodes the latent mean; sampling occurs during training.
  Sequence evaluation disables dropout. Checkpoint reuse requires evaluation mode.

## Evidence, interpretation and missing data

Each job directory under `results/reproduction/gha-39024026594-1-*` contains
`job.json`, `comparison.json`, `run.log`, reference snapshots and outputs. The
parent contains the source manifest and environment. The assessment JSON links
to immutable archived evidence. SHA-256 manifests check bytes, not scientific
validity; the model audit additionally checks that weights load into the declared
architecture with the stated parameter count/window and finite tensors.

| Record or claim | Status and limit |
|---|---|
| Classical controls | Exact stored-array agreement in all four hosted jobs. |
| New canonical and sequence weights | 12 + 36 retained and load-checked; not the missing historical canonical weights. |
| New coded classifier weights | **Not retained.** Trial results, source and environment are available, but weight-only replay of that realization is impossible. A future coded run should export its weights before evaluation. |
| Historical canonical RNG, checkpoints, per-epoch losses, package versions | **Unavailable.** Protocol known to the documented extent; exact historical learned-column recovery cannot be claimed. |
| New-run optimizer state and training-data tensors | Not complete resume records. Regeneration uses seeds and original call order; mid-training resume is not implemented. |
| Same-host repeatability | Full-budget seed-42 MLP control matched twice on one host. It does not establish repeatability of every model or across hardware. |
| Learned differences | Preserve them as additional training-realization/seed-sensitivity evidence. No automatic invalidation of originals, causal attribution to seed alone, or equivalence claim. |
| Broader study coverage | Minimum-size grid, pilots, adaptive tails and matched Mamba were not added to these four reruns. Their historical sources remain in the thesis ledger. |

`RETRAINING_DIFFERENCE_REQUIRES_REVIEW` with worker exit 0 is a completed
experiment whose learned fields differ, not a crash. Comparisons use relative
1e-12 / absolute 1e-14 tolerances on declared fields, not a statistical acceptance
band. Common BER targets stay fixed by the reference; missed targets are explicitly
censored, never dropped. Fresh paired tests condition on each trained model;
uncorrected exploratory p-values and unchanged rankings do not prove equivalence.
The canonical and archived coded reanalyses reconstruct integer error counts
before ranking differences, so floating-point BER subtraction cannot split
ties. Their common trial denominators are 10,000 and 100,000 bits, respectively.
SciPy 1.17.0 uses `wilcoxon(..., zero_method="wilcox", correction=False,
alternative="two-sided", method="auto")`; zeros are omitted and all-zero pairs
return p=1. Ten-pair results agree with independent exhaustive sign enumeration;
the 30-trial coded results can use the asymptotic method. These numerical
settings do not remove the signed-rank distributional assumptions.
Keep historical and new JSONs separate, investigate configuration/RNG/environment
differences, and document any change to a manuscript claim with its evidence.
