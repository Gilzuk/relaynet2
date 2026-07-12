# Appendices {#sec:appendices}

## Appendix A: Mathematical Notation {#sec:appendix-a-mathematical-notation}

  Symbol                                    Description
  ----------------------------------------- -----------------------------------------------------
  $b$                                       Binary bit ($\in \{0, 1\}$)
  $x$                                       Modulated BPSK symbol ($\in \{-1, +1\}$)
  $y$                                       Received signal
  $n$                                       Noise sample
  $h$                                       Fading coefficient
  $\sigma^2$                                Noise variance
  $\text{SNR}$                              Signal-to-Noise Ratio (linear)
  $G$                                       AF amplification gain
  $\mathbf{H}$                              MIMO channel matrix ($\in \mathbb{C}^{2 \times 2}$)
  $\mathbf{W}$                              Neural network weight matrix
  $\mathbf{b}$                              Bias vector
  $P_e$                                     Bit error rate
  $Q(\cdot)$                                Gaussian Q-function
  $K$                                       Rician K-factor
  $\beta$                                   VAE KL weight
  $\lambda$                                 Regularization parameter
  $\eta$                                    Learning rate
  $\Delta$                                  State space discretization step
  $\mathbf{A}, \mathbf{B}, \mathbf{C}, D$   State space model matrices

## Appendix B: Model Architectures and Hyperparameters {#sec:appendix-b-model-architectures-and-hyperparameters}

**MLP (Minimal):** - Input: 5-symbol sliding window - Hidden: 24 neurons, ReLU activation - Output: 1 neuron, Tanh activation - Parameters: 169 - Training: MSE loss, lr=0.01, 100 epochs, 25K samples at SNR=\[5, 10, 15\] dB - Implementation: NumPy (CPU)

**Hybrid:** - Architecture: Same as MLP (169 params) - SNR threshold: Learned (default \~5 dB) - Below threshold → MLP processing; Above threshold → DF processing - Implementation: NumPy (CPU)

**VAE:** - Encoder: 7 → 32 (ReLU) → 16 (ReLU) → μ(8), log σ²(8) - Decoder: 8 → 16 (ReLU) → 32 (ReLU) → 1 (Tanh) - Parameters: 1,777 - Training: β-VAE loss (β=0.1), Adam lr=1e-3, 100 epochs - Implementation: PyTorch (CUDA)

**CGAN (WGAN-GP):** - Generator: (7+8) → 32 (LeakyReLU) → 32 (LeakyReLU) → 16 (LeakyReLU) → 1 (Tanh) - Critic: (1+7) → 32 (LeakyReLU) → 16 (LeakyReLU) → 1 - Parameters: 2,946 - Training: WGAN-GP (λ_GP=10, λ_L1=100), Adam lr=1e-4, 200 epochs, 5 critic updates per generator update - Implementation: PyTorch (CUDA)

**Transformer:** - Input projection: 1 → 32 - Positional encoding: Sinusoidal - Encoder: 2 layers, 4 heads, d_model=32, d_ff=128 - Output projection: 32 → 1 (Tanh) - Parameters: 17,697 - Training: MSE loss, Adam lr=1e-3, 100 epochs - Implementation: PyTorch (reported experiments on CUDA; CPU fallback supported)

**Mamba S6:** - Input projection: 1 → 32 - Mamba blocks: 2 layers, each: LayerNorm → expand (32→64) → S6 (d_state=16) → contract (64→32) → residual - S6 selective parameters: Δ, B, C = f(input) via learned linear projections - Output projection: 32 → 1 (Tanh) - Parameters: 24,001 - Training: MSE loss, Adam lr=1e-3, 100 epochs - Implementation: PyTorch (reported experiments on CUDA; CPU fallback supported)

**Mamba2 (SSD):** - Input projection: 1 → 32 - Mamba-2 blocks: 2 layers, each: LayerNorm → parallel branches (SiLU gate ∥ SSD layer) → gated output → contract (64→32) → residual - SSD layer: chunk_size=8, builds lower-triangular causal kernel $M \in \mathbb{R}^{L \times L}$ per chunk via cumulative log-decay, applies $Y = M \cdot V$ as batched matmul - Inter-chunk state: running state $(B, N, D)$ updated once per chunk - S4D-style A initialisation: $A_{\log} = \log(1, 2, \dots, d_{\text{state}})$ - Selective parameters: Δ, B, C = f(input); value projection V = f(input) - Output projection: 32 → 16 (SiLU) → 1 (Tanh) - Parameters: 26,179 - Training: MSE loss, Adam lr=1e-3, 100 epochs, gradient clipping (max_norm=1.0) - Implementation: PyTorch (CUDA)

## Appendix C: Software Architecture {#sec:appendix-c-software-architecture}

The project is implemented as a modular Python package (`relaynet`) with the following structure:

    relaynet/
    ├── channels/          # Channel models
    │   ├── awgn.py            # AWGN channel
    │   ├── fading.py          # Rayleigh & Rician fading
    │   └── mimo.py            # 2×2 MIMO + ZF/MMSE/SIC equalization
    ├── modulation/
    │   ├── bpsk.py            # BPSK modulate/demodulate
    │   ├── qpsk.py            # QPSK Gray-coded modulate/demodulate
    │   ├── qam.py             # 16-QAM Gray-coded modulate/demodulate
    │   └── psk.py             # 16-PSK modulate/demodulate
    ├── relays/            # Relay strategies
    │   ├── base.py            # Abstract base class
    │   ├── af.py              # Amplify-and-Forward
    │   ├── df.py              # Decode-and-Forward
    │   ├── genai.py           # Minimal MLP (feedforward NN)
    │   ├── hybrid.py          # SNR-adaptive Hybrid
    │   ├── vae.py             # Variational Autoencoder
    │   ├── cgan.py            # Conditional GAN (WGAN-GP)
    │   ├── transformer.py     # Transformer relay
    │   ├── mamba.py           # Mamba S6 relay
    │   └── mamba2.py          # Mamba-2 SSD relay

    ├── simulation/
    │   ├── runner.py          # Monte Carlo BER simulation
    │   └── statistics.py      # CI computation, significance tests
    ├── visualization/
    │   └── plots.py           # BER plotting utilities
    └── utils/
        └── torch_compat.py    # Device detection helpers

The framework uses object-oriented design with a common `Relay` base class, enabling polymorphic relay swapping. Monte Carlo simulation is implemented in `runner.py` with configurable trial count, bit count, and SNR range. All MIMO operations use vectorized PyTorch for GPU acceleration.

**Testing:** 126 automated tests (pytest) cover all channels, modulation (BPSK, QPSK, 16-QAM, 16-PSK), relay strategies, simulation, statistics, and modulation-comparison modules with 100% pass rate.

**Reproducibility:** Random seeds are controlled at the source (bit generation) and noise (per-trial seeding) levels to ensure reproducible results.

## Appendix D: Normalized 3K-Parameter Configurations {#sec:appendix-d-normalized-3k-parameter-configurations}

+----------------+--------------+--------------+---------------------------------------------------+
| ::: minipage   | ::: minipage | ::: minipage | ::: minipage                                      |
| Model          | Parameters   | Window       | Hidden / Architecture                             |
| :::            | :::          | :::          | :::                                               |
+:===============+:=============+:=============+:==================================================+
| MLP-3K         | ,004         |              | hidden=231                                        |
+----------------+--------------+--------------+---------------------------------------------------+
| Hybrid-3K      | ,004         |              | hidden=231 (+ DF switch)                          |
+----------------+--------------+--------------+---------------------------------------------------+
| VAE-3K         | ,037         |              | latent=10, hidden=(44, 20)                        |
+----------------+--------------+--------------+---------------------------------------------------+
| CGAN-3K        | ,004         |              | noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
+----------------+--------------+--------------+---------------------------------------------------+
| Transformer-3K | ,007         |              | d_model=18, heads=2, layers=1                     |
+----------------+--------------+--------------+---------------------------------------------------+
| Mamba-3K       | ,027         |              | d_model=16, d_state=6, layers=1                   |
+----------------+--------------+--------------+---------------------------------------------------+
| Mamba2-3K      | ,004         |              | d_model=15, d_state=6, chunk_size=8, layers=1     |
+----------------+--------------+--------------+---------------------------------------------------+

All 3K configurations use a window size of 11 (vs. 5 for original MLP/Hybrid, and 11 for original sequence models) to provide a common input context. The parameter counts are within ±1.2% of the 3,000 target.

::: center

------------------------------------------------------------------------
:::
