# RelayNet2 — Generative AI for Two-Hop Relay Communication

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-187%20passed-brightgreen.svg)](#testing)
[![Experiments](https://img.shields.io/badge/Experiments-18-blue.svg)](#recent-experiments-summary)

A framework for comparing **classical and AI-based relay strategies** in two-hop cooperative communication, over **AWGN and Rayleigh fading** SISO channels, with **BPSK, QPSK and 16-QAM** modulation.

> **Thesis vs. framework scope.** This repository contains both the `relaynet` simulation **framework** (whose full capabilities are described below) and the M.Sc. **thesis** it supports, under [`thesis/`](thesis/). The thesis deliberately fixes a **single canonical setup** — SISO on both hops, i.i.d. Rayleigh fast fading, complex baseband, Gray-coded QPSK, uncoded BER — and varies only the relay function. Its **principal contribution** is **learned relaying under unknown/mismatched channels** (carried on BPSK); the higher-order-modulation extension is not part of the current build. MIMO, Rician relay comparison, and 16-PSK were removed from both the thesis and the framework, and are recorded as *future work* (Rician is retained only to draw the fading-distribution figure). See the [Thesis](#thesis-msc) section.

---

## Table of Contents

- [Overview](#overview)
- [Thesis (M.Sc.)](#thesis-msc)
- [Channel Types](#channel-types)
- [Antenna Topologies](#antenna-topologies)
- [Relay Strategies](#relay-strategies)
- [Architecture](#architecture)
- [Key Findings](#key-findings)
- [Unknown-Channel Contribution](#unknown-channel-contribution)
- [Verifying the Thesis Against Its Data](#verifying-the-thesis-against-its-data)
- [Appendix — Proof of Claims](#appendix--proof-of-claims)
- [Recent Experiments Summary](#recent-experiments-summary)
- [BER Results — Original Models](#ber-results--original-models)
- [Normalized 3K-Parameter Comparison](#normalized-3k-parameter-comparison)
- [AI Model Architectures](#ai-model-architectures)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Checkpoints Summary](#checkpoints-summary)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements and compares **9 relay strategies** (2 classical + 7 AI-based) across **2 channel types** (AWGN, Rayleigh) and **3 modulation schemes** (BPSK, QPSK, 16-QAM) to evaluate the potential of generative AI and modern sequence models for cooperative relay communication.

### Relay Methods

| Method | Type | Architecture | Parameters |
|--------|------|-------------|-----------|
| AF | Classical | Amplify-and-Forward | 0 |
| **DF** | Classical | Decode-and-Forward | 0 |
| **MLP (Minimal)** | Supervised | Feedforward NN | 169 |
| **Hybrid** | SNR-Adaptive | MLP + DF switching | 169 |
| **VAE** | Generative | Variational Autoencoder | 1,777 |
| **CGAN** | Adversarial | Conditional GAN (WGAN-GP) | 2,946 |
| **Transformer** | Attention | Multi-head Self-Attention | 17,697 |
| **Mamba S6** | State Space | Selective State Space Model | 24,001 |
| **Mamba-2 SSD** | State Space | Structured State Space Duality | 26,179 |

### Channel Types

| Channel | Key Characteristic |
|---------|--------------------|
| **AWGN** | Additive white Gaussian noise only |
| **Rayleigh** | Flat fading, no line-of-sight (NLOS) |

## Thesis (M.Sc.)

The M.Sc. thesis *"Deep Learning Architectures for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network Relay Strategies"* (Gil Zukerman, Tel Aviv University, 2026) lives under [`thesis/`](thesis/).

| Item | Location |
|------|----------|
| LaTeX source (chapters, bibliography, figures) | `thesis/main.tex`, `thesis/chapters/`, `thesis/results/` |
| Compiled PDF | `thesis/main.pdf`, `thesis_preview.pdf` |
| **Overleaf-ready package** (self-contained, bundled fonts) | `thesis_overleaf.zip` |

**Building.** Compile with **XeLaTeX** (required for `fontspec` + `polyglossia` Hebrew); all fonts are bundled in `thesis/fonts/`, so no system-font installation is needed. See `thesis/OVERLEAF.md`. To use Overleaf: upload `thesis_overleaf.zip`, set **Menu → Compiler → XeLaTeX**, main document `main.tex`.

**Structure.** The thesis fixes one canonical setup — SISO, i.i.d. Rayleigh fast fading on both hops, complex baseband, Gray-coded **QPSK**, uncoded — and varies only the relay function:

| Chapter | Content |
|---------|---------|
| Ch 1–4 | Introduction, background, research objectives (H1–H5), methods |
| Ch 5 | **Core experiments** on the canonical setup: channel-model validation, relay comparison, parameter normalization and complexity, plus the coded block-DF study |
| Ch 6 | **Principal contribution:** learned relaying under **unknown and mismatched channels** (H5) — see below |
| Ch 7–8 | Discussion and conclusions; summary |
| Ch 9 | Appendices (reproducibility, per-experiment budgets, minimum-relay-size sweep) |

The higher-order-modulation extension (16-QAM, joint 2D $N$-class classification) is **not** part of the current build; its source remains under `thesis/chapters/` but is commented out of `main.tex`. QPSK is the canonical modulation rather than an extension.

The unknown-channel study (Ch 6) is reproduced in this framework by the `e6_*_ported.py` scripts (see [Unknown-Channel Contribution](#unknown-channel-contribution)).

---

## Channel Types

### AWGN (Additive White Gaussian Noise)

The simplest channel model — a clean additive noise channel:

```
y = x + n,    n ~ N(0, σ²)
```

### Rayleigh Fading

Models non-line-of-sight (NLOS) propagation with multiplicative fading:

```
y = h · x + n,    h ~ CN(0, 1),    n ~ CN(0, σ²)
Equalization: x̂ = y / h   (perfect CSI)
```

## Relay Strategies

### Classical Relays

- **AF (Amplify-and-Forward):** Amplifies the received noisy signal and retransmits — simple but propagates noise.
- **DF (Decode-and-Forward):** Demodulates, re-modulates, and retransmits — eliminates noise at the relay but introduces demodulation errors.

### AI-Based Relays

- **MLP (Minimal):** A compact 2-layer feedforward neural network (window_size=5, hidden=24). Uses a sliding window to process each bit with its neighbors. Only 169 parameters.
- **Hybrid:** SNR-adaptive relay that switches between MLP (low SNR) and DF (high SNR) based on a learned threshold. Combines the best of both worlds.
- **VAE (Variational Autoencoder):** Probabilistic generative model that learns a latent representation of the clean signal. Encoder maps to a latent space; decoder reconstructs the signal.
- **CGAN (Conditional GAN):** Wasserstein GAN with gradient penalty. The generator learns to denoise conditioned on the noisy input; the critic provides adversarial training signal.
- **Transformer:** Multi-head self-attention over a sliding window of symbols. Captures global dependencies with O(n²) complexity. Architecture: d_model=32, heads=4, layers=2.
- **Mamba S6 (Selective State Space):** Linear-time sequence model with input-dependent state transitions. Captures long-range dependencies with O(n) complexity. Architecture: d_model=32, d_state=16, layers=2.
- **Mamba-2 SSD (Structured State Space Duality):** Successor to Mamba S6 using the SSD formulation, which restricts the state transition to a scalar-times-identity structure to unlock matmul-based training. Architecture: d_model=32, d_state=16, layers=2.

---

## Architecture

```
                     Two-Hop Relay Network
                     =====================

    Source ──── Channel ────► Relay ──── Channel ────► Destination
                  │              ▲              │
             [AWGN/Fading]    [AI or        [AWGN/Fading]
                           Classical]
    Topology:  SISO (1x1)
    Channels:  AWGN  |  Rayleigh
    Relays:    AF │ DF │ MLP │ Hybrid │ VAE │ CGAN │ Transformer │ Mamba S6 │ Mamba-2 SSD
```

Each relay strategy is evaluated on both SISO channels (AWGN and Rayleigh) using Monte Carlo simulation with 95% confidence intervals (10 trials × 10,000 bits per SNR point).

---

## Key Findings

### Original Models (varying parameter counts)

1. **The best AI relay is channel-dependent** — CGAN leads on AWGN; DF leads on Rayleigh
2. **State space models beat attention** for signal processing (O(n) vs O(n²))
3. **DF dominates at medium/high SNR** (≥6 dB) — no training required
4. **Hybrid relay** provides the best practical trade-off: AI at low SNR, DF at high SNR
5. **All AI relays dramatically outperform AF** across all channels

### Normalized 3K Comparison (equal parameter budgets)

When all 6 AI models are constrained to ≈3,000 parameters:

1. **All architectures converge in performance** — the architecture gap narrows at small scale; DF remains the strongest baseline on most channels
2. **MLP/Hybrid remain competitive** — simple feedforward networks match sequence models at equal param budgets
3. **VAE consistently underperforms** — probabilistic overhead hurts at all scales
4. **Architecture matters less than expected** — at 3K params, all models are within ~1 dB of each other

---

## Unknown-Channel Contribution

The thesis's principal contribution (Ch 6, hypothesis **H5**) studies **learned relaying when the channel is unknown to, or mismatched with, the classical relay's model class** — the regime where a fixed minimal MLP earns its place. It is reproduced in this framework by the `e6_*_ported.py` scripts, with figures/data in [`e6_unknown_channel_results/`](e6_unknown_channel_results/).

Three relay architectures appear in these studies, and two of them coincidentally have the same parameter count, so the labels are worth reading carefully: the canonical relay is $5 \to 24 \to 1$ (169 parameters), the unknown-ISI and flat-memory relays are $11 \to 13 \to 1$ (**170**), and the composite, blind and pilot-budget relays take complex I/Q pairs, $22 \to 7 \to 1$ (169 again).

| Study | Script | Key result |
|-------|--------|-----------|
| Unknown ISI + control | `e6_sim_ported.py` | AF/DF pinned at the **analytic 0.25 ISI floor** (DF *non-monotonic* in SNR); the 170-param MLP restores reliable relaying |
| Viterbi MLSE benchmark | `e6_viterbi_ported.py` | Genie-CSI Viterbi is ~1–1.5 dB better than the MLP; a 200-pilot LS estimate matches genie |
| Flat (memoryless) control | `e6_flat_ported.py` | Unknown phase/gain/asymmetry: classical **does not fail**, MLP only matches it (gap ≤ 0.0036) — isolates *memory*, not unknownness, as the cause |
| Composite cascade | `e6_composite_ported.py` | ISI × PA-nonlinearity × unknown phase: MLP recovers from raw I/Q, ~2 dB behind pilot-aided Viterbi |
| Posterior-free (blind) | `e6_blind_ported.py` | MLP matches blind CMA while avoiding decision-directed MLSE's instability |
| Partial posterior | `e6_partial_ported.py` | Pilot-budget crossover: Viterbi wins with ≥10 pilots, collapses at 5; MLP is pilot-free and flat |
| Complexity | `e6_complexity_ported.py` | Viterbi cost grows as $M^L$; the relay's cost is constant **for a fixed architecture** (~330 flops/sym) and 30–90× faster in wall-clock. Holding the window fixed as memory grows is a choice, not a law: spanning longer memory generally widens the window, and the relay's cost then grows roughly linearly in it |

**Bottom line (H5):** the learned relay **never beats a correctly matched classical receiver**, but occupies a well-defined niche — *identification-free, fixed-complexity* mitigation of structural model-class mismatch (memory, nonlinearity, absent pilots), where the memoryless classical relays fail outright.

The scope is narrower than "family-agnostic": the network is trained on the same impairment family it is tested on, so its weights carry prior information about that family. What it does without is **per-block** channel state — no pilots, no explicit identification, no online adaptation, on a realization it has not seen. It is not evaluated on a structurally different family absent from training.

---

## Verifying the Thesis Against Its Data

Every number the thesis publishes is checked against the file that produced it. Two tools do this, and both are expected to exit `0`:

```bash
python verify_thesis_tables.py    # published cells vs their data sources
python provenance_audit.py        # every result file is committed and newer than its script
```

`verify_thesis_tables.py` reads the LaTeX, extracts each published value, and compares it against the `.json`/`.npy` that generated it, with a tolerance set by the number of decimals shown. It currently checks **498 cells across 26 tables and prose claims**.

`provenance_audit.py` links each experiment to its script, its output files and the commit that produced them, and fails if a result is uncommitted or predates the script that generates it. `python provenance_audit.py --tables` regenerates the per-table reproduction ledger in `memory-bank/table_provenance.md`.

**The checks are themselves tested.** A verifier that examines nothing reports the same "OK" as one that examines everything and finds no problem, and that gap has hidden real defects here — a check that validated constants against their own arithmetic and could never fail, and an earlier one that read three of ten rows while printing OK. So:

- `MIN_CELLS` records the coverage each check is expected to reach; a shortfall fails rather than passing quietly.
- A check that cannot run at all fails unless explicitly allowlisted in `ALLOWED_SKIPS`.
- `tests/test_verifier_catches_drift.py` perturbs a published number in a scratch copy of the thesis and asserts the owning check flags it. Without this, nothing proves a check *can* fail.

Run the whole suite with `pytest`.

---

## Appendix — Proof of Claims

Each headline claim of the unknown-channel study, verified against theory and primary literature. The channel used throughout is the normalized 3-tap FIR $h = [1, 0.7, 0.5]/\lVert\cdot\rVert \approx [0.758, 0.531, 0.379]$ (`e6_viterbi_ported.py`).

### Claim 1 — The learned relay never beats a correctly matched classical receiver

The bound is the **symbol-MAP (BCJR/APP)** detector, and the distinction from MLSE matters. Given the true channel model and the same observation, symbol-MAP minimizes *bit* error probability, so no learned function of that observation can beat it on the BER metric used here — against *that* comparator the claim is a theorem. **MLSE minimizes *sequence* error probability and is not BER-optimal** [1], [2], [16], so against genie-CSI Viterbi the claim is an *empirical* finding, not a theorem: a learned relay with lower BER than Viterbi would contradict no optimality result. No comparison in this thesis is made against a BER-optimal detector at either modulation order; that benchmark remains open. The literature is consistent: learned receivers that "beat classical" beat *mismatched or suboptimal* baselines. SBRNN approaches Viterbi-with-CSI and passes it only under imperfect CSI [3]; ViterbiNet matches the model-based algorithm and wins only under CSI uncertainty [4]; DeepRx beats practical LMMSE receivers, which are not MAP-optimal under the studied impairments [5]; Ye–Li–Juang beat MMSE under pilot shortage, CP removal and clipping — mismatch again [6]; end-to-end autoencoders beat classical *schemes* by redesigning the transmitter, not a receiver-only counterexample [7].

### Claim 2 — The analytic 0.25 BER floor and DF's non-monotonicity

With the normalized taps, the ISI magnitude sum $h_1 + h_2 \approx 0.910$ exceeds the cursor $h_0 \approx 0.758$: the eye is **closed**. Of the four equiprobable BPSK interferer sign patterns, exactly one (both interferers opposing) yields $0.758 - 0.910 = -0.152 < 0$, a deterministic sign flip; the other three ($0.606$, $0.910$, $1.668$) stay correct. A noise-free memoryless slicer therefore errs on exactly one pattern in four: BER $\to$ exactly $1/4$. Non-monotonicity follows from the same geometry: at moderate SNR, noise occasionally pushes the flipped sample back across zero, so the error rate on the bad pattern is below 1 there and rises toward 1 as SNR $\to \infty$ — total BER climbs toward 0.25 from below. This is standard closed-eye behavior [8], [2, ch. 9].

### Claim 3 — Complexity: $M^L$ trellis vs. a fixed forward pass

MLSE maintains $M^{L-1}$ trellis states and evaluates $M^L$ branch metrics per symbol [1], [2, ch. 10]. The $11 \to 13 \to 1$ MLP costs $\approx 2(11 \cdot 13 + 13)$ MACs plus activations $\approx 330$ flops/symbol, constant for the fixed architecture — with the stated caveat that spanning longer memory generally widens the window, after which cost grows roughly linearly in it. One fairness note: reduced-complexity sequence estimation (RSSE [9], DFSE, M-algorithm) also breaks the $M^L$ scaling, so full Viterbi is the steepest classical comparator.

### Claim 4 — Blind regime: CMA converges; decision-directed blind MLSE does not

CMA performs blind equalization of constant-modulus signals [10], [11]. The channel is minimum-phase (zeros at $|z| \approx 0.707$), so a short FIR equalizer can approximately invert it; finite length and noise enhancement leave a residual BER of order $10^{-3}$ at 20 dB, matching the measured $3.3\times10^{-3}$. Known CMA caveats — local minima for under-length equalizers [12] — support "matches but does not excel". Decision-directed blind MLSE, bootstrapping taps from its own decisions, is a crude form of per-survivor processing [13]; misconvergence, sign/shift ambiguities and error propagation are documented failure modes, matching its observed instability.

### Claim 5 — Pilot-budget crossover: reliable at ≥10 pilots, collapse at 5

LS estimation of 3 unknown taps is identifiable from 5 pilots ($5 > 3$), but convolution edge effects leave a near-square, ill-conditioned system whose LS variance $\propto \sigma^2 \operatorname{tr}((X^H X)^{-1})$ explodes; Viterbi with badly wrong taps then error-propagates catastrophically. The collapse is an **estimation-variance plus error-propagation** effect, consistent with CRB scaling $\sigma^2 L / N_p$ [14] — not strict non-identifiability. The result is specific to the classical LS+Viterbi pipeline: meta-learned demodulators adapt from very few pilots [15], which does not contradict the claim but bounds its scope.

### Claim 6 — Genie-CSI MLSE leads the minimal MLP by only 1–1.5 dB

A finite-window symbol-wise detector cannot beat the MAP detector over that same window, which cannot beat one given the whole sequence: a window truncates the observation, and truncation cannot add information. That ordering is all theory supplies — it fixes no particular gap. No windowed-MAP detector was implemented here, so the measured 1–1.5 dB is **not** decomposed into the window's share and the 170-parameter approximation's. That published learned detectors close the gap further — SBRNN within fractions of a dB of Viterbi [3], ViterbiNet essentially to zero with enough capacity [4] — is consistent with the residual gap being a property of the deliberately minimal budget, but this thesis does not separate the two causes.

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

## BER Results — Original Models

### AWGN Channel (0–20 dB)

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|----------|----|----|-------|--------|-----|------|-------------|----------|-------------|
| 0 | 0.291 | 0.268 | 0.264 | 0.262 | 0.376 | **0.261** | 0.267 | 0.269 | 0.270 |
| 4 | 0.154 | 0.112 | 0.112 | 0.113 | 0.330 | **0.111** | 0.114 | 0.111 | **0.111** |
| 8 | 0.044 | **0.010** | 0.015 | **0.010** | 0.291 | 0.013 | 0.014 | **0.010** | 0.012 |
| 12 | 0.0027 | **1.67e-04** | **1.67e-04** | **1.67e-04** | 0.269 | 3.33e-04 | 3.33e-04 | **1.67e-04** | **1.67e-04** |
| 16 | **0** | **0** | **0** | **0** | 0.258 | **0** | **0** | **0** | **0** |
| 20 | **0** | **0** | **0** | **0** | 0.250 | **0** | **0** | **0** | **0** |

> At 8+ dB, DF (0 parameters) matches or beats all AI methods. CGAN achieves the best AI performance at low SNR (0–4 dB) on AWGN.

### Results Plots

Per-channel BER comparison plots with 95% confidence intervals are in the `results/` directory:

| Channel | Plot |
|---------|------|
| AWGN | `results/awgn_comparison_ci.png` |
| Rayleigh | `results/fading_comparison.png` |
| Model Complexity | `results/complexity_comparison_all_relays.png` |

---

## Normalized 3K-Parameter Comparison

To enable a fair **apples-to-apples** comparison, all 6 AI models were scaled to ≈3,000 parameters:

| Model | Parameters | Configuration |
|-------|-----------|---------------|
| MLP-3K | 3,004 | window=11, hidden=231 |
| Hybrid-3K | 3,004 | window=11, hidden=231 (+ DF switching) |
| VAE-3K | 3,037 | window=11, latent=10, hidden=(44, 20) |
| CGAN-3K | 3,004 | window=11, noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
| Transformer-3K | 3,007 | window=11, d_model=18, heads=2, layers=1 |
| Mamba-3K | 3,027 | window=11, d_model=16, d_state=6, layers=1 |

### 3K BER Results Across All Channels

#### AWGN

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.65e-1 | 2.65e-1 | 2.67e-1 | 2.69e-1 | **2.61e-1** | 2.60e-1 |
| 10 | 2.68e-3 | 1.44e-3 | 9.48e-3 | 2.00e-3 | 1.88e-3 | **1.84e-3** |
| 20 | **0** | **0** | **0** | **0** | **0** | **0** |

#### Rayleigh Fading

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.59e-1 | 2.58e-1 | 2.70e-1 | 2.54e-1 | 2.50e-1 | **2.49e-1** |
| 10 | 4.87e-2 | 4.84e-2 | 5.60e-2 | 4.74e-2 | 4.65e-2 | **4.64e-2** |
| 20 | 5.84e-3 | 5.68e-3 | 7.08e-3 | 5.64e-3 | 5.64e-3 | **5.60e-3** |

### Normalized 3K Plots

| Plot | Description |
|------|-------------|
| `results/normalized_3k_awgn.png` | AWGN channel, all 6 models at ~3K params |
| `results/normalized_3k_rayleigh.png` | Rayleigh fading, all 6 models at ~3K params |
| `results/normalized_3k_all_channels.png` | **Consolidated 2×3 grid** of all channels |

---

## AI Model Architectures

### Mamba S6 (Selective State Space Model)

```
State equation: x_k = exp(Δ·A) · x_{k-1} + Δ·B · u_k
Output:         y_k = C · x_k + D · u_k
Selective:      Δ, B, C = f(input)    ← input-dependent!
Complexity:     O(n) — linear in sequence length
```

Original: d_model=32, d_state=16, layers=2 → **24,001 params**
3K: d_model=16, d_state=6, layers=1 → **3,027 params**

### Transformer (Multi-Head Self-Attention)

```
Attention:  softmax(Q·Kᵀ / √d_k) · V
Complexity: O(n²) — quadratic in sequence length
```

Original: d_model=32, heads=4, layers=2 → **17,697 params**
3K: d_model=18, heads=2, layers=1 → **3,007 params**

### MLP (Minimal Feedforward)

```
Input:   window of noisy symbols (size 5 or 11)
Layer 1: input → hidden (ReLU)
Output:  hidden → 1 (Tanh)
```

Original: window=5, hidden=24 → **169 params**
3K: window=11, hidden=231 → **3,004 params**

### VAE (Variational Autoencoder)

```
Encoder:  x → μ, log(σ²)    (latent space)
Sample:   z = μ + σ · ε      (reparameterization trick)
Decoder:  z → x̂              (reconstruction)
Loss:     MSE + β · KL(q||p)
```

Original: hidden=(32, 16), latent=8 → **1,777 params**
3K: hidden=(44, 20), latent=10 → **3,037 params**

### CGAN (Conditional GAN / WGAN-GP)

```
Generator:  (noisy_signal, noise_z) → denoised_signal
Critic:     (signal, condition) → realness_score
Training:   Wasserstein loss + gradient penalty (λ=10)
```

Original: g_hidden=(32, 32, 16), c_hidden=(32, 16), noise=8 → **2,946 params**
3K: g_hidden=(30, 30, 16), c_hidden=(32, 16), noise=8 → **3,004 params**

### Hybrid (SNR-Adaptive)

```python
def process(signal, snr_db):
    if snr_db < threshold:
        return mlp_relay.process(signal)   # AI for low SNR
    else:
        return df_relay.process(signal)      # Classical for high SNR
```

---

## Project Structure

```
relaynet2/
├── relaynet/                         # Core library package
│   ├── channels/
│   │   ├── awgn.py                       # AWGN channel
│   │   ├── fading.py                     # Rayleigh & Rician fading (Rician used for the fading-PDF figure only)
│   ├── modulation/
│   │   └── bpsk.py                       # BPSK modulation/demodulation
│   ├── relays/
│   │   ├── af.py                         # Amplify-and-Forward
│   │   ├── df.py                         # Decode-and-Forward
│   │   ├── genai.py                      # Minimal MLP (feedforward NN)
│   │   ├── hybrid.py                     # SNR-adaptive Hybrid relay
│   │   ├── vae.py                        # Variational Autoencoder relay
│   │   ├── cgan.py                       # Conditional GAN relay (WGAN-GP)
│   │   └── base.py                       # Abstract relay base class
│   ├── simulation/
│   │   ├── runner.py                     # Monte Carlo BER simulation
│   │   └── statistics.py                 # CI, significance tests
│   ├── visualization/
│   │   └── plots.py                      # BER plotting utilities
│   └── utils/
│       └── torch_compat.py               # PyTorch device helpers
│
├── checkpoints/                      # 22+ implementation checkpoints
│   ├── checkpoint_01_channel.py          # AWGN channel model
│   ├── checkpoint_02_modulation.py       # BPSK modulation
│   ├── checkpoint_03_nodes.py            # Source/Relay/Destination nodes
│   ├── checkpoint_04_simulation.py       # Simulation framework
│   ├── checkpoint_05_plotting.py         # BER plotting
│   ├── checkpoint_06_decode_forward.py   # DF relay
│   ├── checkpoint_07_comparative_plot.py # AF vs DF comparison
│   ├── checkpoint_08_genai_relay.py      # MLP relay
│   ├── checkpoint_09_final_comparison.py # 3-way comparison
│   ├── checkpoint_10_rl_relay.py         # RL (Q-Learning) relay
│   ├── checkpoint_11_enhanced_training.py# Enhanced MLP
│   ├── checkpoint_12_maximum_training.py # Maximum MLP
│   ├── checkpoint_13_minimal_complexity.py # Minimal 169-param MLP
│   ├── checkpoint_14_complexity_comparison_plot.py
│   ├── checkpoint_15_vae_relay.py        # VAE relay
│   ├── checkpoint_16_cgan_pytorch.py     # CGAN relay (PyTorch)
│   ├── checkpoint_17_final_comparison.py # 4-way comparison
│   ├── checkpoint_18_transformer_relay.py# Transformer relay
│   ├── checkpoint_19_transformer_comparison.py
│   ├── checkpoint_20_mamba_s6_relay.py   # Mamba S6 relay
│   ├── checkpoint_21_final_with_mamba.py # Full comparison
│   ├── checkpoint_22_master_ber_chart.py # Master BER charts
│   └── checkpoint_22_normalized_3k.py    # 3K-param model factories
│
├── scripts/
│   ├── run_full_comparison.py            # Full pipeline: train + evaluate all
│   └── plot_normalized_3k.py             # Standalone 3K comparison plots
│
├── tests/                            # 187 tests (pytest)
│   ├── test_channels.py                  # AWGN and Rayleigh channel tests
│   ├── test_modulation.py                # BPSK modulation tests
│   ├── test_relays.py                    # All relay strategy tests
│   ├── test_simulation.py                # Monte Carlo runner tests
│   └── test_statistics.py                # CI & significance tests
│
├── results/                          # Generated BER plots, JSON, charts
│   ├── bpsk_comparison/                  # §7.2–7.7 BPSK relay comparisons
│   ├── normalized_3k/                    # §7.8 equal-parameter comparison
│   ├── modulation/                       # §7.10 BPSK→QPSK→QAM16
│   ├── qam16_activation/                 # §7.11 activation study
│   ├── classify_vs_regress/              # §7.13 classification formulation
│   ├── classify_activations/             # §7.13 activation sweep
│   ├── classify_closing_gap/             # §7.13 closing the DF gap
│   ├── csi/                              # §7.14–7.15 CSI injection
│   ├── e2e/                              # §7.16 end-to-end autoencoder
│   ├── all_relays_16class/               # §7.17 16-class 2D (all 7 archs)
│   ├── classify_16class/                 # §7.17 MLP 16-class variants
│   ├── channel_analysis/                 # §7.1 channel model analysis
│   ├── activation_comparison/            # legacy activation comparison
│   ├── logs/                             # experiment failure logs
│   └── ...
│
├── thesis/                           # M.Sc. thesis (canonical restructured version)
│   ├── main.tex                          # XeLaTeX root (compiler set via magic comment)
│   ├── main.pdf                          # compiled thesis
│   ├── chapters/                         # ch01–ch09 + frontmatter, appendices
│   ├── results/                          # thesis figures
│   ├── fonts/                            # bundled TTF/OTF (Times New Roman, Arial, Courier New, David CLM)
│   └── OVERLEAF.md                       # Overleaf compile instructions
├── thesis_preview.pdf                # compiled thesis (top-level copy)
├── thesis_overleaf.zip               # self-contained Overleaf upload package
│
├── e6_sim_ported.py                  # Unknown-channel study (Ch 6) ported to relaynet:
├── e6_viterbi_ported.py              #   ISI, Viterbi-MLSE benchmark, flat control,
├── e6_flat_ported.py                 #   composite cascade, blind/partial posterior,
├── e6_composite_ported.py            #   and complexity — see Unknown-Channel Contribution
├── e6_blind_ported.py
├── e6_partial_ported.py
├── e6_complexity_ported.py
├── e6_unknown_channel_results/       # figures + .npy data for the ported unknown-channel studies
│
├── run_experiments.py                # Unified experiment runner (18 experiments)
├── make.ps1                          # PowerShell build script (Windows)
├── Makefile                          # GNU Make build script (Linux/macOS)
├── EXPERIMENT_GUIDELINES.md          # Developer guide for experiments
├── CHART_GUIDELINES.md               # Publication chart rules (22 rules)
├── README.md
├── TECHNICAL_REPORT.md               # Mathematical details
├── MAMBA_FINAL_REPORT.md             # Mamba S6 analysis
├── FINAL_SUMMARY.md                  # Project summary
├── IMPLEMENTATION_PLAN.md            # Development plan
└── CHECKPOINT_LOG.md                 # Development log
```

---

## Quick Start

### Requirements

```bash
pip install numpy matplotlib torch scipy
```


```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Run Experiments (Unified Runner)

All 18 experiments are managed through `run_experiments.py`:

```bash
# List all available experiments
python run_experiments.py --list

# Run all experiments (quick mode — reduced samples/epochs)
python run_experiments.py --quick

# Run a specific experiment
python run_experiments.py --exp 7.17 --quick

# Force retrain (ignore cached checkpoints)
python run_experiments.py --exp 7.2 --retrain

# Regenerate all charts from saved JSON (no retraining)
python run_experiments.py --regen-charts
```

### Build Script (PowerShell)

A `make.ps1` script provides shorthand targets:

```powershell
.\make.ps1 help            # Show all targets
.\make.ps1 list            # List experiments
.\make.ps1 quick           # Run all experiments (quick)
.\make.ps1 exp -s 7.17     # Run specific experiment
.\make.ps1 retrain -s 7.2  # Force retrain
.\make.ps1 charts          # Regenerate charts from JSON
.\make.ps1 clean           # Clean all generated outputs
```

A GNU `Makefile` is also included for Linux/macOS.

### Legacy Comparison Scripts

```bash
# Full pipeline: train + evaluate all (legacy)
python scripts/run_full_comparison.py --include-sequence-models --include-normalized

# Standalone 3K comparison plots
python scripts/plot_normalized_3k.py --full
```

### Run a Single Relay Programmatically

```python
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.awgn import awgn_channel

relay = DecodeAndForwardRelay()
results = run_monte_carlo(relay, snr_range=range(0, 21, 2),
                          channel=awgn_channel, num_bits=10000, num_trials=10)
for snr, ber, ci_lo, ci_hi in results:
    print(f"SNR={snr:2d} dB  BER={ber:.4e}  CI=[{ci_lo:.4e}, {ci_hi:.4e}]")
```

---

## Testing

All 187 tests pass:

```bash
python -m pytest tests/ -q
# 187 passed
```

Tests cover:
- **Channels:** AWGN noise power, Rayleigh and Rician fading statistics
- **Modulation:** BPSK modulate/demodulate correctness
- **Relays:** All 6 AI relays + 2 classical relays (training, inference, parameter counts)
- **Simulation:** Monte Carlo runner, BER computation
- **Statistics:** Confidence intervals, Wilcoxon significance tests

---

## Checkpoints Summary

| CP | Description | Key Result |
|----|-------------|-----------|
| 01 | AWGN Channel | Noise model implementation |
| 02 | BPSK Modulation | Modulation/demodulation |
| 03 | Network Nodes | Source/Relay/Destination |
| 04 | Simulation | Two-hop relay framework |
| 05 | Plotting | BER visualization |
| 06 | DF Relay | Classical baseline |
| 07 | AF vs DF | DF >> AF |
| 08 | MLP Relay | First AI relay |
| 09 | 3-way Comparison | MLP beats AF |
| 10 | RL Relay | Q-Learning approach |
| 11 | Enhanced MLP | Better training |
| 12 | Maximum MLP | Overfitting found |
| 13 | **Minimal (169p)** | **Best parameter efficiency** |
| 14 | Complexity Plot | Params vs performance |
| 15 | VAE Relay | Probabilistic generative model |
| 16 | CGAN Relay | Adversarial generative model |
| 17 | 4-way Comparison | DF/Minimal/VAE/CGAN |
| 18 | Transformer | Multi-head attention relay |
| 19 | Transformer vs DF | Attention-based comparison |
| 20 | **Mamba S6** | **Selective state space relay** |
| 21 | Full Comparison | All 9 methods compared |
| 22 | Master BER Charts | Final visualization + **3K normalized comparison** |

---

## Recent Experiments Summary

All experiments are managed through the unified `run_experiments.py` runner (18 experiments, §7.1–§7.17 + constellation diagrams). Every experiment supports `--quick` mode, saves `.pt` checkpoints, exports `.json` for chart regeneration, and logs failures automatically.

| Section | Experiment | Results Directory |
|---------|-----------|-------------------|
| §7.1 | Channel Model Analysis | `results/channel_analysis/` |
| §7.2 | BPSK AWGN Relay Comparison | `results/bpsk_comparison/` |
| §7.3 | BPSK Rayleigh Relay Comparison | `results/bpsk_comparison/` |
| §7.8 | Normalized 3K Comparison | `results/normalized_3k/` |
| §7.9 | Master 2×3 Chart | `results/bpsk_comparison/` |
| §7.10 | Modulation Comparison (BPSK → QPSK → QAM16) | `results/modulation/` |
| §7.11 | QAM16 Activation Study | `results/qam16_activation/` |
| §7.13 | Classification vs Regression + Activations + Closing Gap | `results/classify_vs_regress/`, `results/classify_activations/`, `results/classify_closing_gap/` |
| §7.17 | 16-Class 2D QAM16 (all 7 architectures) | `results/all_relays_16class/` |

### Modulation Extension — §7.10 (BPSK → QPSK → 16-QAM)

- **QPSK**: All BPSK findings generalise fully via I/Q splitting — BER curves are identical to BPSK across all 9 relays.
- **16-QAM**: BPSK-trained relays exhibit an irreducible BER floor (~0.18–0.25 at 16 dB) due to `tanh` compressing the 4-level PAM amplitudes.

### Classification vs Regression — §7.13

Three sub-studies explore the classification formulation for 16-QAM relaying:

- **Classify vs Regress**: Classification MLP (4-class) achieves ~1.3× lower BER than regression MLP at 20 dB
- **Activation Sweep**: 8 hidden/output activation combinations; H:Sigmoid wins overall, O:Sigmoid is the only catastrophic failure
- **Closing the DF Gap**: 6 progressive enhancements (window, SNR range, hidden size) reduce the classification gap to DF from 8.1× to 1.0× at 20 dB

### 16-Class 2D Classification — §7.17 (Key Breakthrough)

**Problem**: Per-axis I/Q splitting classifies 4 levels independently per axis, producing a structural BER floor of ~0.0081 at 20 dB.

**Solution**: Treat the relay as a **joint 2D classifier** over all 16 QAM constellation points. The model receives (y_I, y_Q) and outputs 16 logits. All 7 relay architectures tested in both 4-class and 16-class modes (14 variants total).

| Relay | 4-cls BER @ 20 dB | 16-cls BER @ 20 dB | Improvement |
|---|---|---|---|
| MLP (472p) | 0.00811 | **0.00002** | 405× |
| VAE (2,112p) | 0.00810 | **0.00000** | ∞ |
| CGAN (3,361p) | 0.35340 | 0.28370 | 1.2× |
| Hybrid (472p) | 0.00811 | **0.00000** | ∞ |
| Transformer (17,984p) | 0.00810 | **0.00001** | 810× |
| Mamba S6 (24,288p) | 0.00810 | **0.00009** | 90× |
| Mamba-2 SSD (26,466p) | 0.00811 | **0.00197** | 4.1× |
| DF (classical) | 0.00000 | — | — |

**Key finding**: The top-3 16-class variants (Hybrid, VAE, MLP) **match classical DF** at 20 dB — the first time neural relays achieve near-zero BER on 16-QAM. The structural floor is an artefact of I/Q splitting, not a fundamental limitation of neural architectures. CGAN remains the poorest performer; Mamba-2 lags behind its S6 predecessor.

---

## Citation

If you use this work, please cite:

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

## License

MIT License — see [LICENSE](LICENSE) for details.
