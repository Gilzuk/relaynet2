# RelayNet2 - Generative AI for Two-Hop Relay Communication

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for exploring **Generative AI and modern sequence models** for two-hop relay communication over AWGN channels.

## Overview

This project implements and compares **7 relay strategies** ranging from classical methods to state-of-the-art AI architectures:

| Method | Type | Parameters | Low SNR Wins |
|--------|------|-----------|-------------|
| AF | Classical | 0 | 0/3 |
| **DF** | Classical | 0 | 0/3 (best overall) |
| **Minimal** | Supervised | 169 | 2/3 |
| VAE | Generative | ~1,800 | 2/3 |
| CGAN | Adversarial | ~2,500 | 1/3 |
| Transformer | Attention | 17,697 | 1/3 |
| **Mamba S6** | State Space | 24,001 | **3/3** |

## Key Findings

- **Mamba S6 is the best AI method** — wins all low-SNR scenarios (0–4 dB)
- **State space models beat attention** for signal processing tasks
- **Minimal (169 params) is most efficient** — 1.78 wins per 100 parameters
- **DF dominates at medium/high SNR** (6+ dB) — no training needed
- **Optimal hybrid**: Mamba (0–4 dB) + DF (6+ dB)

## Architecture

```
Source ──AWGN──► Relay ──AWGN──► Destination
                  ▲
            [AI/Classical]
```

## Project Structure

```
relaynet2/
├── checkpoints/                  # All 22 implementation checkpoints
│   ├── checkpoint_01_channel.py          # AWGN channel model
│   ├── checkpoint_02_modulation.py       # BPSK modulation
│   ├── checkpoint_03_nodes.py            # Source/Relay/Destination nodes
│   ├── checkpoint_04_simulation.py       # Simulation framework
│   ├── checkpoint_05_plotting.py         # BER plotting utilities
│   ├── checkpoint_06_decode_forward.py   # DF relay (classical)
│   ├── checkpoint_07_comparative_plot.py # AF vs DF comparison
│   ├── checkpoint_08_genai_relay.py      # Original GenAI relay
│   ├── checkpoint_09_final_comparison.py # 3-way comparison
│   ├── checkpoint_10_rl_relay.py         # RL (Q-Learning) relay
│   ├── checkpoint_11_enhanced_training.py# Enhanced GenAI
│   ├── checkpoint_12_maximum_training.py # Maximum GenAI
│   ├── checkpoint_13_minimal_complexity.py # Minimal (169 params)
│   ├── checkpoint_14_complexity_comparison_plot.py
│   ├── checkpoint_15_vae_relay.py        # VAE relay
│   ├── checkpoint_16_cgan_pytorch.py     # CGAN relay (PyTorch)
│   ├── checkpoint_17_final_comparison.py # 4-way comparison
│   ├── checkpoint_18_transformer_relay.py# Transformer (attention)
│   ├── checkpoint_19_transformer_comparison.py
│   ├── checkpoint_20_mamba_s6_relay.py   # Mamba S6 (state space)
│   ├── checkpoint_21_final_with_mamba.py # Full comparison
│   └── checkpoint_22_master_ber_chart.py # Master BER charts
│
├── results/                      # All generated BER charts
│   ├── master_ber_comparison.png         # Full + Low SNR BER
│   ├── comprehensive_comparison.png      # 3-panel overview
│   ├── performance_summary_table.png     # Summary table
│   ├── architecture_comparison.png       # Params vs Performance
│   ├── transformer_comparison.png        # Transformer vs DF
│   ├── final_4way_comparison.png         # DF/Minimal/VAE/CGAN
│   └── ...
│
├── README.md
├── TECHNICAL_REPORT.md           # Mathematical details
├── MAMBA_FINAL_REPORT.md         # Mamba S6 analysis
├── FINAL_SUMMARY.md              # Project summary
├── IMPLEMENTATION_PLAN.md        # Development plan
└── CHECKPOINT_LOG.md             # Development log
```

## Quick Start

### Requirements

```bash
pip install numpy matplotlib torch scipy
```

### Run Classical Relay (AF/DF)

```python
from checkpoints.checkpoint_01_channel import awgn_channel
from checkpoints.checkpoint_02_modulation import bpsk_modulate, calculate_ber
from checkpoints.checkpoint_06_decode_forward import simulate_df_transmission

# Simulate DF relay at 10 dB SNR
ber, errors = simulate_df_transmission(num_bits=100000, snr_db=10)
print(f"DF BER at 10 dB: {ber:.6f}")
```

### Run Mamba S6 Relay (Best AI)

```python
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper, simulate_mamba_transmission

# Create and train Mamba relay
relay = MambaRelayWrapper(window_size=11, d_model=32, d_state=16, num_layers=2)
relay.train(training_snrs=[5, 10, 15], num_samples=50000, epochs=100)

# Test at low SNR
ber, errors = simulate_mamba_transmission(num_bits=10000, snr_db=2, relay=relay)
print(f"Mamba BER at 2 dB: {ber:.6f}")
```

### Run Minimal GenAI Relay (Most Efficient)

```python
from checkpoints.checkpoint_13_minimal_complexity import MinimalGenAIRelay, simulate_minimal_genai

relay = MinimalGenAIRelay(window_size=5, hidden1=24, hidden2=0)
relay.train_minimal(training_snrs=[5, 10, 15], num_samples=25000, epochs=100)

ber, errors = simulate_minimal_genai(num_bits=10000, snr_db=4, relay=relay)
print(f"Minimal BER at 4 dB: {ber:.6f}")
```

### Generate All BER Charts (No Training Needed)

```bash
python checkpoints/checkpoint_22_master_ber_chart.py
```

## BER Results

### Full SNR Range (0–20 dB)

| SNR (dB) | AF | DF | Minimal | VAE | CGAN | Transformer | Mamba S6 |
|----------|----|----|---------|-----|------|-------------|----------|
| 0 | 0.480 | 0.265 | 0.259 | 0.261 | 0.265 | 0.259 | **0.255** |
| 2 | 0.420 | 0.186 | 0.180 | 0.181 | 0.185 | 0.181 | **0.176** |
| 4 | 0.360 | 0.104 | 0.103 | 0.104 | 0.105 | 0.104 | **0.102** |
| 6 | 0.290 | **0.045** | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 |
| 8 | 0.210 | **0.012** | 0.013 | 0.013 | 0.012 | 0.013 | 0.014 |
| 10 | 0.140 | **0.002** | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 |

## Architectures

### Mamba S6 (State Space Model)
```
State equation: x_k = exp(Δ·A) · x_{k-1} + Δ·B · u_k
Output:         y_k = C · x_k + D · u_k
Selective:      Δ, B, C = f(input)  # Input-dependent!
Complexity:     O(n) — linear
```

### Transformer (Attention)
```
Attention:  softmax(QK^T / sqrt(d)) · V
Complexity: O(n²) — quadratic
```

### Minimal GenAI (Feedforward)
```
Layer 1: 5 → 24 (ReLU)
Output:  24 → 1 (Tanh)
Params:  169
```

## Optimal Deployment Strategy

```python
def optimal_relay(snr_db, signal):
    if snr_db < 4:
        return mamba_s6.process(signal)    # Best AI at low SNR
    elif snr_db < 6:
        return minimal.process(signal)     # Efficient
    else:
        return decode_forward(signal)      # Classical optimal
```

## Checkpoints Summary

| CP | Description | Key Result |
|----|-------------|-----------|
| 01 | AWGN Channel | Noise model |
| 02 | BPSK Modulation | Modulation/demodulation |
| 03 | Network Nodes | Source/Relay/Destination |
| 04 | Simulation | Two-hop framework |
| 05 | Plotting | BER visualization |
| 06 | DF Relay | Classical baseline |
| 07 | AF vs DF | DF >> AF |
| 08 | GenAI Relay | First AI relay |
| 09 | 3-way Comparison | GenAI beats AF |
| 10 | RL Relay | Q-Learning approach |
| 11 | Enhanced GenAI | Better training |
| 12 | Maximum GenAI | Overfitting found |
| 13 | **Minimal (169p)** | **Best efficiency** |
| 14 | Complexity Plot | Params vs performance |
| 15 | VAE Relay | Probabilistic approach |
| 16 | CGAN Relay | Adversarial approach |
| 17 | 4-way Comparison | DF/Minimal/VAE/CGAN |
| 18 | Transformer | Attention mechanism |
| 19 | Transformer vs DF | Attention loses |
| 20 | **Mamba S6** | **Best AI method** |
| 21 | Full Comparison | All methods |
| 22 | Master BER Chart | Final visualization |

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
