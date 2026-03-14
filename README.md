# RelayNet2 — Generative AI for Two-Hop Relay Communication

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-60%20passed-brightgreen.svg)](#testing)

A comprehensive framework for comparing **classical and AI-based relay strategies** in two-hop cooperative communication across **5 channel models**: AWGN, Rayleigh fading, Rician fading, 2×2 MIMO with Zero-Forcing, and 2×2 MIMO with MMSE equalization.

---

## Table of Contents

- [Overview](#overview)
- [Channel Models](#channel-models)
- [Relay Strategies](#relay-strategies)
- [Architecture](#architecture)
- [Key Findings](#key-findings)
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

This project implements and compares **8 relay strategies** (2 classical + 6 AI-based) across **5 wireless channel models** to evaluate the potential of generative AI and modern sequence models for cooperative relay communication.

### Relay Methods

| Method | Type | Architecture | Parameters |
|--------|------|-------------|-----------|
| AF | Classical | Amplify-and-Forward | 0 |
| **DF** | Classical | Decode-and-Forward | 0 |
| **GenAI (Minimal)** | Supervised | Feedforward NN | 169 |
| **Hybrid** | SNR-Adaptive | GenAI + DF switching | 169 |
| **VAE** | Generative | Variational Autoencoder | 1,777 |
| **CGAN** | Adversarial | Conditional GAN (WGAN-GP) | 2,946 |
| **Transformer** | Attention | Multi-head Self-Attention | 17,697 |
| **Mamba S6** | State Space | Selective State Space Model | 24,001 |

### Channel Models

| Channel | Type | Key Characteristic |
|---------|------|--------------------|
| **AWGN** | SISO | Additive white Gaussian noise only |
| **Rayleigh** | SISO fading | Flat fading, no line-of-sight (NLOS) |
| **Rician (K=3)** | SISO fading | Fading with line-of-sight component |
| **2×2 MIMO ZF** | MIMO | Spatial multiplexing, Zero-Forcing equalization |
| **2×2 MIMO MMSE** | MIMO | Spatial multiplexing, MMSE equalization |

---

## Channel Models

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

### Rician Fading (K = 3)

Models channels with a dominant line-of-sight (LOS) component:

```
h = √(K/(K+1)) · e^{jθ} + √(1/(K+1)) · h_scatter
```

The Rician K-factor controls the ratio of LOS to scattered power. Higher K yields less severe fading.

### 2×2 MIMO with Zero-Forcing (ZF)

Spatial multiplexing with two transmit and two receive antennas:

```
y = H·x + n,    H ∈ ℂ^{2×2},    H_ij ~ CN(0, 1)
ZF equalization: x̂ = H⁻¹·y
```

ZF completely removes inter-stream interference but amplifies noise when H is ill-conditioned.

### 2×2 MIMO with MMSE

Regularized linear equalization that trades a small residual interference for reduced noise enhancement:

```
x̂ = (H^H·H + σ²·I)⁻¹ · H^H · y
```

MMSE consistently outperforms ZF, especially at low SNR.

> **GPU Acceleration:** Both MIMO channels use vectorized PyTorch batched `torch.linalg.solve` instead of per-symbol Python loops, achieving >100× speed-up on CPU and further gains on CUDA GPUs.

---

## Relay Strategies

### Classical Relays

- **AF (Amplify-and-Forward):** Amplifies the received noisy signal and retransmits — simple but propagates noise.
- **DF (Decode-and-Forward):** Demodulates, re-modulates, and retransmits — eliminates noise at the relay but introduces demodulation errors.

### AI-Based Relays

- **GenAI (Minimal):** A compact 2-layer feedforward neural network (window_size=5, hidden=24). Uses a sliding window to process each bit with its neighbors. Only 169 parameters.
- **Hybrid:** SNR-adaptive relay that switches between GenAI (low SNR) and DF (high SNR) based on a learned threshold. Combines the best of both worlds.
- **VAE (Variational Autoencoder):** Probabilistic generative model that learns a latent representation of the clean signal. Encoder maps to a latent space; decoder reconstructs the signal.
- **CGAN (Conditional GAN):** Wasserstein GAN with gradient penalty. The generator learns to denoise conditioned on the noisy input; the critic provides adversarial training signal.
- **Transformer:** Multi-head self-attention over a sliding window of symbols. Captures global dependencies with O(n²) complexity. Architecture: d_model=32, heads=4, layers=2.
- **Mamba S6 (Selective State Space):** Linear-time sequence model with input-dependent state transitions. Captures long-range dependencies with O(n) complexity. Architecture: d_model=32, d_state=16, layers=2.

---

## Architecture

```
                     Two-Hop Relay Network
                     =====================

    Source ──── Channel ────► Relay ──── Channel ────► Destination
                  │              ▲              │
             [AWGN/Fading/    [AI or        [AWGN/Fading/
              MIMO]         Classical]       MIMO]

    Channels:  AWGN │ Rayleigh │ Rician K=3 │ 2×2 MIMO ZF │ 2×2 MIMO MMSE
    Relays:    AF │ DF │ GenAI │ Hybrid │ VAE │ CGAN │ Transformer │ Mamba S6
```

Each relay strategy is evaluated across all 5 channels using Monte Carlo simulation with 95% confidence intervals (10 trials × 10,000 bits per SNR point).

---

## Key Findings

### Original Models (varying parameter counts)

1. **Mamba S6 is the best AI method** — wins all low-SNR scenarios across all channels
2. **State space models beat attention** for signal processing (O(n) vs O(n²))
3. **DF dominates at medium/high SNR** (≥6 dB) — no training required
4. **Hybrid relay** provides the best practical trade-off: AI at low SNR, DF at high SNR
5. **All AI relays dramatically outperform AF** across all channels
6. **MIMO MMSE consistently outperforms MIMO ZF** at every SNR for all relay types

### Normalized 3K Comparison (equal parameter budgets)

When all 6 AI models are constrained to ≈3,000 parameters:

1. **Mamba S6 and Transformer converge in performance** — the architecture gap narrows at small scale
2. **GenAI/Hybrid remain competitive** — simple feedforward networks match sequence models at equal param budgets
3. **VAE consistently underperforms** — probabilistic overhead hurts at small scale
4. **Architecture matters less than expected** — at 3K params, all models are within ~1 dB of each other
5. **CGAN matches Transformer/Mamba** on most channels despite fundamentally different training

---

## BER Results — Original Models

### AWGN Channel (0–20 dB)

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|----------|----|----|-------|--------|-----|------|-------------|----------|
| 0 | 0.480 | 0.265 | 0.259 | 0.259 | 0.261 | 0.265 | 0.259 | **0.255** |
| 2 | 0.420 | 0.186 | 0.180 | 0.180 | 0.181 | 0.185 | 0.181 | **0.176** |
| 4 | 0.360 | 0.104 | 0.103 | 0.103 | 0.104 | 0.105 | 0.104 | **0.102** |
| 6 | 0.290 | **0.045** | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 |
| 8 | 0.210 | **0.012** | 0.013 | 0.013 | 0.013 | 0.012 | 0.013 | 0.014 |
| 10 | 0.140 | **0.002** | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 |

> At 6+ dB, DF (0 parameters) matches or beats all AI methods. AI excels only at low SNR (0–4 dB).

### Results Plots

Per-channel BER comparison plots with 95% confidence intervals are in the `results/` directory:

| Channel | Plot |
|---------|------|
| AWGN | `results/awgn_comparison_ci.png` |
| Rayleigh | `results/fading_comparison.png` |
| Rician K=3 | `results/rician_comparison_ci.png` |
| 2×2 MIMO ZF | `results/mimo_2x2_comparison_ci.png` |
| 2×2 MIMO MMSE | `results/mimo_2x2_mmse_comparison_ci.png` |
| Model Complexity | `results/complexity_comparison_all_relays.png` |

---

## Normalized 3K-Parameter Comparison

To enable a fair **apples-to-apples** comparison, all 6 AI models were scaled to ≈3,000 parameters:

| Model | Parameters | Configuration |
|-------|-----------|---------------|
| GenAI-3K | 3,004 | window=11, hidden=231 |
| Hybrid-3K | 3,004 | window=11, hidden=231 (+ DF switching) |
| VAE-3K | 3,037 | window=11, latent=10, hidden=(44, 20) |
| CGAN-3K | 3,004 | window=11, noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
| Transformer-3K | 3,007 | window=11, d_model=18, heads=2, layers=1 |
| Mamba-3K | 3,027 | window=11, d_model=16, d_state=6, layers=1 |

### 3K BER Results Across All Channels

#### AWGN

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.65e-1 | 2.65e-1 | 2.67e-1 | 2.69e-1 | **2.61e-1** | 2.60e-1 |
| 10 | 2.68e-3 | 1.44e-3 | 9.48e-3 | 2.00e-3 | 1.88e-3 | **1.84e-3** |
| 20 | **0** | **0** | **0** | **0** | **0** | **0** |

#### Rayleigh Fading

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.59e-1 | 2.58e-1 | 2.70e-1 | 2.54e-1 | 2.50e-1 | **2.49e-1** |
| 10 | 4.87e-2 | 4.84e-2 | 5.60e-2 | 4.74e-2 | 4.65e-2 | **4.64e-2** |
| 20 | 5.84e-3 | 5.68e-3 | 7.08e-3 | 5.64e-3 | 5.64e-3 | **5.60e-3** |

#### Rician (K=3)

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.05e-1 | 2.05e-1 | 2.18e-1 | 2.05e-1 | 2.00e-1 | **2.00e-1** |
| 10 | 1.54e-2 | 1.47e-2 | 1.98e-2 | 1.48e-2 | **1.45e-2** | 1.46e-2 |
| 20 | 9.20e-4 | 8.80e-4 | 1.24e-3 | 8.80e-4 | **6.80e-4** | 7.20e-4 |

#### 2×2 MIMO ZF

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 2.52e-1 | 2.52e-1 | 2.64e-1 | 2.52e-1 | 2.47e-1 | **2.45e-1** |
| 10 | 4.82e-2 | 4.80e-2 | 5.55e-2 | 4.67e-2 | 4.64e-2 | **4.64e-2** |
| 20 | 5.40e-3 | **5.12e-3** | 5.92e-3 | 5.16e-3 | **5.12e-3** | 5.16e-3 |

#### 2×2 MIMO MMSE

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|----------|----------|-----------|--------|---------|----------------|----------|
| 0 | 1.65e-1 | 1.65e-1 | 1.79e-1 | 1.63e-1 | **1.62e-1** | 1.64e-1 |
| 10 | 2.68e-2 | 2.51e-2 | 3.37e-2 | 2.54e-2 | 2.56e-2 | 2.60e-2 |
| 20 | 2.92e-3 | 2.60e-3 | 3.84e-3 | 2.76e-3 | 2.72e-3 | **2.56e-3** |

### Normalized 3K Plots

| Plot | Description |
|------|-------------|
| `results/normalized_3k_awgn.png` | AWGN channel, all 6 models at ~3K params |
| `results/normalized_3k_rayleigh.png` | Rayleigh fading, all 6 models at ~3K params |
| `results/normalized_3k_rician_k3.png` | Rician K=3, all 6 models at ~3K params |
| `results/normalized_3k_2x2_mimo_zf.png` | 2×2 MIMO ZF, all 6 models at ~3K params |
| `results/normalized_3k_2x2_mimo_mmse.png` | 2×2 MIMO MMSE, all 6 models at ~3K params |
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

### GenAI (Minimal Feedforward)

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
        return genai_relay.process(signal)   # AI for low SNR
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
│   │   ├── fading.py                     # Rayleigh & Rician fading
│   │   └── mimo.py                       # 2×2 MIMO ZF & MMSE (GPU-accelerated)
│   ├── modulation/
│   │   └── bpsk.py                       # BPSK modulation/demodulation
│   ├── relays/
│   │   ├── af.py                         # Amplify-and-Forward
│   │   ├── df.py                         # Decode-and-Forward
│   │   ├── genai.py                      # Minimal GenAI (feedforward NN)
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
│   ├── checkpoint_08_genai_relay.py      # GenAI relay
│   ├── checkpoint_09_final_comparison.py # 3-way comparison
│   ├── checkpoint_10_rl_relay.py         # RL (Q-Learning) relay
│   ├── checkpoint_11_enhanced_training.py# Enhanced GenAI
│   ├── checkpoint_12_maximum_training.py # Maximum GenAI
│   ├── checkpoint_13_minimal_complexity.py # Minimal 169-param GenAI
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
├── tests/                            # 60 tests (pytest)
│   ├── test_channels.py                  # AWGN, Rayleigh, Rician, MIMO tests
│   ├── test_modulation.py                # BPSK modulation tests
│   ├── test_relays.py                    # All relay strategy tests
│   ├── test_simulation.py                # Monte Carlo runner tests
│   └── test_statistics.py                # CI & significance tests
│
├── results/                          # Generated BER plots (23 files)
│   ├── awgn_comparison_ci.png
│   ├── fading_comparison.png
│   ├── rician_comparison_ci.png
│   ├── mimo_2x2_comparison_ci.png
│   ├── mimo_2x2_mmse_comparison_ci.png
│   ├── normalized_3k_awgn.png
│   ├── normalized_3k_rayleigh.png
│   ├── normalized_3k_rician_k3.png
│   ├── normalized_3k_2x2_mimo_zf.png
│   ├── normalized_3k_2x2_mimo_mmse.png
│   ├── normalized_3k_all_channels.png    # Consolidated 2×3 grid
│   └── ...
│
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

For GPU-accelerated MIMO channels (optional):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Run the Full Comparison Pipeline

```bash
# All channels, all relays (including Transformer + Mamba), plus normalized 3K comparison
python scripts/run_full_comparison.py --include-sequence-models --include-normalized

# Quick mode (lower fidelity, faster)
python scripts/run_full_comparison.py --include-sequence-models --include-normalized --quick
```

### Generate Normalized 3K Plots Only

```bash
python scripts/plot_normalized_3k.py

# High-fidelity mode (more samples, more trials)
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

### Run with MIMO Channel

```python
from relaynet.channels.mimo import mimo_2x2_mmse_channel

# GPU-accelerated MIMO MMSE (auto-detects CUDA)
results = run_monte_carlo(relay, snr_range=range(0, 21, 2),
                          channel=lambda s, snr: mimo_2x2_mmse_channel(s, snr, device="auto"),
                          num_bits=10000, num_trials=10)
```

---

## Testing

All 60 tests pass:

```bash
python -m pytest tests/ -q
# 60 passed in ~7s
```

Tests cover:
- **Channels:** AWGN noise power, Rayleigh/Rician fading statistics, MIMO ZF & MMSE equalization
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
| 08 | GenAI Relay | First AI relay |
| 09 | 3-way Comparison | GenAI beats AF |
| 10 | RL Relay | Q-Learning approach |
| 11 | Enhanced GenAI | Better training |
| 12 | Maximum GenAI | Overfitting found |
| 13 | **Minimal (169p)** | **Best parameter efficiency** |
| 14 | Complexity Plot | Params vs performance |
| 15 | VAE Relay | Probabilistic generative model |
| 16 | CGAN Relay | Adversarial generative model |
| 17 | 4-way Comparison | DF/Minimal/VAE/CGAN |
| 18 | Transformer | Multi-head attention relay |
| 19 | Transformer vs DF | Attention-based comparison |
| 20 | **Mamba S6** | **Best AI method (low SNR)** |
| 21 | Full Comparison | All 8 methods compared |
| 22 | Master BER Charts | Final visualization + **3K normalized comparison** |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{relaynet2_2026,
  title={Generative AI for Two-Hop Relay Communication},
  author={Zukerma, Gil},
  year={2026},
  url={https://github.com/Gilzuk/relaynet2}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
