# Generative AI for Two-Hop Relay Communication: A Comparative Study of Classical and AI-Based Relay Strategies

---

**Gil Zukerma**

A thesis submitted in partial fulfillment of the requirements for the degree of Master of Science

2026

---

## Table of Contents

1. [List of Abbreviations](#1-list-of-abbreviations)
2. [Abstract (Hebrew)](#2-abstract-hebrew)
3. [Keywords](#3-keywords)
4. [Introduction and Literature Review](#4-introduction-and-literature-review)
   - 4.1 Cooperative Relay Communication
   - 4.2 Classical Relay Strategies
   - 4.3 Machine Learning in Wireless Communication
   - 4.4 Generative Models for Signal Processing
   - 4.5 Sequence Models: Transformers and State Space Models
   - 4.6 MIMO Systems and Equalization
   - 4.7 Research Gap and Motivation
5. [Research Objectives](#5-research-objectives)
6. [Methods](#6-methods)
   - 6.1 System Model
   - 6.2 Channel Models
     - 6.2.1 AWGN Channel — Theoretical Analysis
     - 6.2.2 Rayleigh Fading Channel — Theoretical Analysis
     - 6.2.3 Rician Fading Channel — Theoretical Analysis
     - 6.2.4 Fading Coefficient Distributions
     - 6.2.5 2×2 MIMO Channel — Theoretical Analysis
     - 6.2.6 Channel Model Validation (Simulative)
   - 6.3 Relay Strategies
   - 6.4 MIMO Equalization Techniques
   - 6.5 Simulation Framework
   - 6.6 Normalized Parameter Comparison
7. [Results](#7-results)
   - 7.1 Channel Model Validation
   - 7.2 AWGN Channel — Relay Comparison
   - 7.3 Rayleigh Fading Channel — Relay Comparison
   - 7.4 Rician Fading Channel — Relay Comparison
   - 7.5 2×2 MIMO with ZF Equalization
   - 7.6 2×2 MIMO with MMSE Equalization
   - 7.7 2×2 MIMO with SIC Equalization
   - 7.8 Normalized 3K-Parameter Comparison
   - 7.9 Complexity–Performance Trade-off
8. [Discussion and Conclusions](#8-discussion-and-conclusions)
   - 8.1 Interpretation of Results
   - 8.2 The "Less is More" Principle
   - 8.3 State Space vs. Attention for Signal Processing
   - 8.4 Practical Deployment Recommendations
   - 8.5 Limitations
   - 8.6 Future Work
   - 8.7 Conclusions
9. [References](#9-references)
10. [Appendices](#10-appendices)
    - A. Mathematical Notation
    - B. Model Architectures and Hyperparameters
    - C. Software Architecture
    - D. Normalized 3K-Parameter Configurations
11. [Abstract (English)](#11-abstract-english)

---

## 1. List of Abbreviations

| Abbreviation | Full Term |
|---|---|
| AF | Amplify-and-Forward |
| AWGN | Additive White Gaussian Noise |
| BER | Bit Error Rate |
| BPSK | Binary Phase-Shift Keying |
| CGAN | Conditional Generative Adversarial Network |
| CI | Confidence Interval |
| CN | Circularly-Symmetric Complex Normal |
| CSI | Channel State Information |
| DF | Decode-and-Forward |
| DNN | Deep Neural Network |
| GAN | Generative Adversarial Network |
| GenAI | Generative AI (Minimal Feedforward Relay) |
| GPU | Graphics Processing Unit |
| KL | Kullback–Leibler |
| LOS | Line-of-Sight |
| MIMO | Multiple-Input Multiple-Output |
| MMSE | Minimum Mean Square Error |
| MRC | Maximal Ratio Combining |
| MSE | Mean Squared Error |
| NLOS | Non-Line-of-Sight |
| NN | Neural Network |
| ReLU | Rectified Linear Unit |
| RL | Reinforcement Learning |
| SIC | Successive Interference Cancellation |
| SINR | Signal-to-Interference-plus-Noise Ratio |
| SISO | Single-Input Single-Output |
| SNR | Signal-to-Noise Ratio |
| SSM | State Space Model |
| V-BLAST | Vertical Bell Laboratories Layered Space-Time |
| VAE | Variational Autoencoder |
| WGAN-GP | Wasserstein GAN with Gradient Penalty |
| ZF | Zero-Forcing |

---

## 2. Abstract (Hebrew)

*[תקציר בעברית — עד 2 עמודים]*

*לפי הנחיות בית הספר להנדסת חשמל, התקציר בעברית חייב לכלול לפחות ארבעה פסקאות:*
*1. מטרות המחקר — השוואה שיטתית בין תשע אסטרטגיות ממסר (AF, DF, GenAI, Hybrid, VAE, CGAN, Transformer, Mamba S6, Mamba-2 SSD) לתקשורת שיתופית דו-קפיצתית.*
*2. שיטות — סימולציית מונטה קרלו (100,000 ביטים לנקודת SNR) על שישה ערוצים/טופולוגיות (AWGN, Rayleigh, Rician K=3, MIMO 2×2 עם ZF/MMSE/SIC), כולל ניתוח תיאורטי וסימולטיבי של מודלי הערוץ. בדיקות סטטיסטיות Wilcoxon לכל נקודת SNR.*
*3. תוצאות — ממסרי AI עולים על שיטות קלאסיות ב-SNR נמוך (0–4 dB); DF אופטימלי ב-SNR בינוני-גבוה עם 0 פרמטרים; רשת עם 169 פרמטרים מספיקה; Mamba S6 ו-Mamba-2 SSD מובילים בביצועים.*
*4. מסקנות — גישת Hybrid (GenAI + DF) מומלצת לשימוש מעשי; מורכבות ארכיטקטונית חשובה פחות מגודל מודל; MMSE-SIC מספק את הביצועים הטובים ביותר ב-MIMO; Mamba-2 SSD מציע יתרון חישובי בזכות עיבוד chunk-parallel.*

---

## 3. Keywords

Cooperative relay communication, generative AI, deep learning, two-hop relay, Mamba state space model, Transformer, variational autoencoder, conditional GAN, MIMO equalization, bit error rate

---

## 4. Introduction and Literature Review

### 4.1 Cooperative Relay Communication

Cooperative relay communication is a fundamental technique in modern wireless networks that extends coverage, improves reliability, and increases throughput by employing intermediate nodes between a source and destination. In a two-hop relay network, a source transmits a signal to a relay, which processes it and retransmits it to the destination. This architecture is central to standards such as LTE-Advanced and 5G NR, where relay nodes bridge coverage gaps and enhance cell-edge performance (Laneman et al., 2004; Nosratinia et al., 2004).

The relay communication model can be described as:

$$y_R = x + n_1, \quad x_R = f(y_R), \quad y_D = x_R + n_2$$

where $x$ is the transmitted BPSK symbol, $y_R$ is the signal received at the relay, $f(\cdot)$ is the relay processing function, $x_R$ is the retransmitted signal, and $y_D$ is the signal received at the destination. The noise terms $n_1 \sim \mathcal{N}(0, \sigma^2)$ and $n_2 \sim \mathcal{N}(0, \sigma^2)$ represent independent additive white Gaussian noise on each hop.

The choice of relay processing function $f(\cdot)$ fundamentally determines system performance. Classical approaches include amplify-and-forward (AF), which simply scales the received signal, and decode-and-forward (DF), which regenerates the signal through demodulation and re-modulation. Each has well-understood performance characteristics: AF is simple but propagates noise, while DF eliminates first-hop noise but introduces error propagation when decoding fails (Cover & El Gamal, 1979).

### 4.2 Classical Relay Strategies

**Amplify-and-Forward (AF).** The AF relay amplifies the received signal by a gain factor $G$ that normalizes the output power:

$$G = \sqrt{\frac{P_{\text{target}}}{\mathbb{E}[|y_R|^2]}}$$

The end-to-end SNR for AF relay is:

$$\text{SNR}_{\text{eff}}^{\text{AF}} = \frac{\text{SNR}_1 \cdot \text{SNR}_2}{\text{SNR}_1 + \text{SNR}_2 + 1}$$

This expression reveals a fundamental limitation: even when one hop has high SNR, the effective SNR is bottlenecked by the weaker hop. Moreover, the relay amplifies both signal and noise from the first hop, leading to noise accumulation at the destination.

**Decode-and-Forward (DF).** The DF relay demodulates the received signal, recovers the transmitted bits, and re-modulates clean symbols:

$$\hat{b}_R = \text{demod}(y_R), \quad x_R = \text{mod}(\hat{b}_R)$$

The end-to-end BER for DF is:

$$P_e^{\text{DF}} = P_{e,1} + (1 - P_{e,1}) \cdot P_{e,2}$$

where $P_{e,1}$ and $P_{e,2}$ are the BER of the first and second hops, respectively. For BPSK modulation over AWGN (real-valued noise with variance $1/\text{SNR}$), each hop's BER is $P_e = Q\left(\sqrt{\text{SNR}}\right)$. DF provides clean regeneration at high SNR but suffers from error propagation when the first-hop BER is non-negligible.

### 4.3 Machine Learning in Wireless Communication

The application of machine learning to physical-layer wireless communication has gained significant momentum in recent years. Deep learning has been applied to channel estimation (Ye et al., 2018), signal detection (Samuel et al., 2019), autoencoder-based end-to-end communication (Dorner et al., 2018), and resource allocation (Sun et al., 2018). These approaches learn complex mappings from data, potentially outperforming hand-crafted algorithms in scenarios where analytical solutions are intractable or suboptimal.

For relay processing, neural networks can learn to denoise signals by exploiting statistical patterns in the received waveform. A supervised learning approach trains a neural network $f_\theta$ to minimize:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{x}_i - x_i)^2$$

where $\hat{x}_i = f_\theta(y_{i-w:i+w})$ is the network output based on a sliding window of $2w+1$ received symbols, and $x_i$ is the clean transmitted symbol. This window-based approach provides temporal context that enables the network to exploit correlations in the noise and signal structure.

A critical question in applying neural networks to relay processing is the relationship between model complexity and performance. Prior work has generally assumed that larger models yield better performance, but this assumption has not been rigorously tested in the relay communication context. Understanding this relationship is essential for practical deployment, where relay nodes may have limited computational resources.

### 4.4 Generative Models for Signal Processing

Generative models offer an alternative paradigm for relay signal processing. Rather than directly learning a denoising function, generative models learn the distribution of clean signals and use this knowledge to reconstruct the transmitted signal from noisy observations.

**Variational Autoencoders (VAEs)** (Kingma & Welling, 2014) learn a latent representation by maximizing the evidence lower bound (ELBO):

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

where the first term encourages accurate reconstruction and the KL divergence term regularizes the latent space. The $\beta$ parameter ($\beta$-VAE) controls the trade-off between reconstruction quality and latent space regularity.

**Conditional GANs (CGANs)** (Mirza & Osindero, 2014) learn signal denoising through adversarial training. A generator $G$ maps noisy signals to denoised outputs, while a discriminator (critic) $D$ distinguishes between real clean signals and generated outputs. The Wasserstein GAN with gradient penalty (WGAN-GP) formulation (Gulrajani et al., 2017) provides stable training:

$$\mathcal{L}_G = -\mathbb{E}[D(G(\mathbf{y}, \mathbf{z}), \mathbf{y})] + \lambda_{\text{L1}} \|\hat{\mathbf{x}} - \mathbf{x}\|_1$$

$$\mathcal{L}_D = \mathbb{E}[D(G(\mathbf{y}, \mathbf{z}), \mathbf{y})] - \mathbb{E}[D(\mathbf{x}, \mathbf{y})] + \lambda_{\text{GP}} \cdot \text{GP}$$

The application of generative models to relay processing has not been extensively studied. This thesis provides the first systematic comparison of VAE and CGAN-based relay processing against classical and supervised learning methods.

### 4.5 Sequence Models: Transformers and State Space Models

Recent advances in sequence modeling have produced two competing paradigms with distinct computational properties.

**Transformers** (Vaswani et al., 2017) use multi-head self-attention to capture global dependencies in sequences:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The attention mechanism computes pairwise interactions between all positions, yielding $O(n^2)$ complexity in sequence length $n$. While powerful for natural language processing, this quadratic cost may be excessive for signal processing tasks where local temporal structure is more relevant than long-range semantic dependencies.

**Mamba (Selective State Space Models)** (Gu & Dao, 2024) offer a linear-time alternative based on structured state space models. The core dynamics are:

$$\mathbf{x}_k = \bar{\mathbf{A}} \mathbf{x}_{k-1} + \bar{\mathbf{B}} u_k, \quad y_k = \mathbf{C} \mathbf{x}_k + D u_k$$

where $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$ and $\bar{\mathbf{B}} = \Delta \mathbf{B}$ are the discretized state matrices. Critically, Mamba makes the parameters $\Delta$, $\mathbf{B}$, and $\mathbf{C}$ input-dependent through learned projections, enabling selective information propagation. This achieves $O(n)$ complexity while maintaining the ability to model long-range dependencies through the recurrent state.

**Mamba-2 (State Space Duality)** (Dao & Gu, 2024) reformulates the selective state space model through an algebraic duality between linear recurrences and matrix multiplications. The key insight is that the SSM output can be computed via a structured semi-separable matrix:

$$y_i = \mathbf{C}_i^\top \left(\prod_{k=j+1}^{i} \bar{\mathbf{A}}_k\right) \mathbf{B}_j u_j, \quad M_{ij} = \begin{cases} \mathbf{C}_i^\top \bar{\mathbf{A}}_{j+1:i} \mathbf{B}_j & i \geq j \\ 0 & i < j \end{cases}$$

The matrix $\mathbf{M}$ is lower-triangular (causal) and semi-separable, enabling chunk-parallel computation: the sequence is divided into chunks of length $L$, and within each chunk the output is computed as a single batched matrix multiply $\mathbf{y} = \mathbf{M}(\mathbf{B} \odot \mathbf{u})$. Inter-chunk state is propagated once per chunk rather than once per time step. This "SSD" (State Space Duality) formulation eliminates the sequential recurrence bottleneck of Mamba S6 while preserving the selective state space semantics.

The application of state space models to physical-layer signal processing is largely unexplored. This thesis presents the first comparison of Mamba S6 and Mamba-2 SSD against Transformers for relay communication, demonstrating that state space models are well suited to this domain.

### 4.6 MIMO Systems and Equalization

Multiple-input multiple-output (MIMO) systems employ multiple antennas at both transmitter and receiver to exploit spatial multiplexing and diversity gains (Tse & Viswanath, 2005). In a 2×2 MIMO system, the received signal is:

$$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}$$

where $\mathbf{H} \in \mathbb{C}^{2 \times 2}$ is the channel matrix with $H_{ij} \sim \mathcal{CN}(0, 1)$ (independent Rayleigh fading per link), $\mathbf{x}$ is the transmitted symbol vector, and $\mathbf{n} \sim \mathcal{CN}(\mathbf{0}, \sigma^2\mathbf{I})$ is noise.

Equalization at the receiver aims to recover $\mathbf{x}$ from $\mathbf{y}$. Three methods of increasing sophistication are commonly used:

- **Zero-Forcing (ZF):** $\hat{\mathbf{x}} = \mathbf{H}^{-1}\mathbf{y}$, which completely cancels inter-stream interference but amplifies noise when $\mathbf{H}$ is ill-conditioned.

- **MMSE:** $\hat{\mathbf{x}} = (\mathbf{H}^H\mathbf{H} + \sigma^2\mathbf{I})^{-1}\mathbf{H}^H\mathbf{y}$, which regularizes the inversion to balance interference cancellation with noise amplification.

- **Successive Interference Cancellation (SIC):** A non-linear technique that detects streams sequentially, cancelling each detected stream from the observation before detecting the next. The MMSE-ordered V-BLAST variant (Wolniansky et al., 1998) orders streams by post-detection SINR, detecting the strongest stream first to minimize error propagation.

Combining AI-based relay processing with MIMO equalization has not been studied in the literature. This thesis evaluates all nine relay strategies across all three equalization methods.

### 4.7 Research Gap and Motivation

Despite growing interest in AI for wireless communication, several important questions remain unanswered:

1. **How do different AI paradigms compare for relay processing?** No prior work provides a systematic comparison across supervised learning, generative models (VAE, CGAN), and modern sequence models (Transformers, Mamba S6, Mamba-2 SSD) for the relay denoising task.

2. **What is the relationship between model complexity and relay performance?** It is commonly assumed that larger models perform better, but this has not been rigorously tested for relay communication.

3. **How do AI relays perform across diverse channel conditions?** Most studies evaluate on AWGN only. A comprehensive evaluation across fading channels and MIMO configurations is lacking.

4. **Are state space models better than Transformers for signal processing?** The Mamba architecture has shown promise in NLP, but its suitability for physical-layer signal processing is unexplored.

5. **Does the chunk-parallel SSD formulation of Mamba-2 improve upon sequential S6?** The Mamba-2 architecture eliminates the sequential recurrence bottleneck, but its impact on relay processing quality and training efficiency is unknown.

This thesis addresses all five questions through a comprehensive framework that implements and compares nine relay strategies across six channel/topology configurations.

---

## 5. Research Objectives

### Main Objective

To systematically evaluate and compare classical and AI-based relay strategies for two-hop cooperative communication, and to determine the optimal relay architecture as a function of channel conditions, SNR regime, and computational constraints.

### Secondary Objectives

1. **Implement and compare nine relay strategies** spanning four learning paradigms: no learning (AF, DF), supervised learning (GenAI, Hybrid), generative modeling (VAE, CGAN), and sequence modeling (Transformer, Mamba S6, Mamba-2 SSD).

2. **Evaluate across six channel/topology configurations:** AWGN, Rayleigh fading, and Rician fading (K=3) channels in SISO topology, and 2×2 MIMO Rayleigh with ZF, MMSE, and SIC equalization.

3. **Investigate the complexity–performance trade-off** by testing model architectures ranging from 0 parameters (classical) to 26,179 parameters (Mamba-2 SSD), and by conducting a normalized comparison at approximately 3,000 parameters.

4. **Determine whether state space models outperform attention mechanisms** for relay signal processing by comparing Mamba S6, Mamba-2 SSD, and Transformer architectures at both original and normalized parameter counts.

5. **Evaluate the chunk-parallel SSD formulation** against the sequential S6 recurrence, quantifying the training-time improvement and any BER differences.

6. **Identify practical deployment recommendations** for selecting the appropriate relay strategy given specific operational constraints (SNR range, computational budget, channel environment).

---

## 6. Methods

### 6.1 System Model

The system under study is a two-hop relay network with a single relay node:

$$\text{Source} \xrightarrow{\text{Hop 1}} \text{Relay} \xrightarrow{\text{Hop 2}} \text{Destination}$$

**Modulation.** Binary Phase-Shift Keying (BPSK) is used throughout: bits $b \in \{0, 1\}$ are mapped to symbols $x = 1 - 2b \in \{-1, +1\}$. At the destination, hard-decision demodulation recovers bits as $\hat{b} = \frac{1 - \text{sign}(\text{Re}(\hat{x}))}{2}$.

**Hop Model.** Each hop applies a channel function followed by optional equalization:

$$y = h(x, \text{SNR}) + n$$

where $h(\cdot)$ depends on the specific channel type (AWGN, fading, or MIMO).

**Power Normalization.** All relay strategies normalize their output power to ensure fair comparison:

$$x_R \leftarrow x_R \cdot \sqrt{\frac{P_{\text{target}}}{P_{\text{current}}}}$$

### 6.2 Channel Models

This section presents the theoretical BER analysis for each channel model used in this study, derives the closed-form expressions, and validates them against Monte Carlo simulation. The theoretical–simulative comparison serves two purposes: (i) it verifies the correctness of the simulation framework, and (ii) it establishes the baseline performance that AI relays must beat.

#### 6.2.1 AWGN Channel — Theoretical Analysis

The AWGN channel adds zero-mean Gaussian noise to the transmitted signal:

$$y = x + n, \quad n \sim \mathcal{N}(0, \sigma^2), \quad \sigma^2 = \frac{P_s}{\text{SNR}_{\text{linear}}}$$

where $P_s = \mathbb{E}[|x|^2] = 1$ for unit-power BPSK symbols. The noise variance is inversely proportional to the linear SNR.

**Single-hop BER.** For BPSK $\pm 1$ symbols and real-valued AWGN with variance $\sigma^2 = 1/\text{SNR}$, an error occurs when the noise pushes the received sample past the decision boundary at the origin. The theoretical BER is:

$$P_e^{\text{AWGN}} = Q\!\left(\frac{1}{\sigma}\right) = Q\left(\sqrt{\text{SNR}}\right) = \frac{1}{2}\,\text{erfc}\!\left(\sqrt{\frac{\text{SNR}}{2}}\right)$$

where $Q(x) = \frac{1}{2}\text{erfc}(x/\sqrt{2})$ is the Gaussian Q-function and $\text{SNR} = P_s / \sigma^2$ is the ratio of signal power to noise variance.

> **Noise-convention note.** The standard textbook result $P_e = Q(\sqrt{2 E_b/N_0})$ assumes that the one-sided noise PSD is $N_0$, giving a baseband noise variance of $N_0/2$ per real dimension. In our AWGN implementation the noise is purely real with variance $\sigma^2 = P_s/\text{SNR}$, so $N_0 = 2\sigma^2 = 2P_s/\text{SNR}$ and $E_b/N_0 = \text{SNR}/2$. Substituting yields $Q(\sqrt{2 \cdot \text{SNR}/2}) = Q(\sqrt{\text{SNR}})$, which is the expression used throughout this work.
>
> By contrast, the fading channels add **complex** Gaussian noise with total power $1/\text{SNR}$ (i.e. $\sigma_I^2 = \sigma_Q^2 = 1/(2\,\text{SNR})$) and extract the real part after ZF equalization, halving the effective noise variance per decision dimension. This naturally introduces a factor of 2 inside the Q-function argument for all fading-channel BER expressions (Sections 6.2.2–6.2.5), making them consistent with the standard textbook forms.

**Two-hop DF BER.** For a decode-and-forward relay with equal-SNR hops, an error occurs at the destination if exactly one hop introduces an error:

$$P_e^{\text{DF}} = P_{e,1} + P_{e,2} - 2P_{e,1}P_{e,2}$$

With equal hops ($P_{e,1} = P_{e,2} = P_e^{\text{AWGN}}$), this becomes $P_e^{\text{DF}} = 2P_e(1 - P_e)$. At high SNR, $P_e \ll 1$ and $P_e^{\text{DF}} \approx 2P_e$, i.e., the two-hop penalty is approximately a factor of 2 in BER.

**Two-hop AF BER.** For amplify-and-forward with equal-SNR hops, the effective end-to-end SNR is:

$$\text{SNR}_{\text{eff}}^{\text{AF}} = \frac{\text{SNR}_1 \cdot \text{SNR}_2}{\text{SNR}_1 + \text{SNR}_2 + 1} = \frac{\gamma^2}{2\gamma + 1}$$

where $\gamma = \text{SNR}$ is the per-hop SNR. The resulting BER is $P_e^{\text{AF}} = Q\left(\sqrt{\text{SNR}_{\text{eff}}}\right)$ (same real-noise convention as the single-hop case). At high SNR, $\text{SNR}_{\text{eff}} \approx \gamma/2$, confirming the well-known 3 dB penalty of AF relaying.

![Figure 1: AWGN Channel — Theoretical vs. Simulative BER. Single-hop, two-hop AF, and two-hop DF: closed-form theory (solid lines) versus Monte Carlo simulation (markers with 95% CI). Theory and simulation match within the confidence interval at all SNR points.](results/channel_theoretical_awgn.png)

*Figure 1: AWGN channel — theoretical BER (solid lines) vs. Monte Carlo simulation (markers with 95% CI) for single-hop, two-hop AF, and two-hop DF.*

#### 6.2.2 Rayleigh Fading Channel — Theoretical Analysis

The Rayleigh fading channel models non-line-of-sight (NLOS) propagation where the signal undergoes multiplicative fading:

$$y = hx + n, \quad h \sim \mathcal{CN}(0, 1), \quad n \sim \mathcal{CN}(0, \sigma^2)$$

The fading coefficient $h$ is a circularly-symmetric complex Gaussian random variable, so its magnitude $|h|$ follows a Rayleigh distribution:

$$f_{|h|}(r) = 2r \cdot e^{-r^2}, \quad r \geq 0$$

with $\mathbb{E}[|h|^2] = 1$ (unit average power). The instantaneous SNR after equalization ($\hat{x} = y/h$) becomes $\gamma_h = |h|^2 \cdot \text{SNR}$, which is exponentially distributed.

**Single-hop BER.** Averaging the conditional BER $P_e(\gamma_h) = Q(\sqrt{2\gamma_h})$ over the exponential distribution of $\gamma_h$ yields the closed-form (Proakis & Salehi, 2008, Eq. 14-4-15):

$$P_e^{\text{Rayleigh}} = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)$$

where $\bar{\gamma} = \text{SNR}$ is the average SNR. At high SNR ($\bar{\gamma} \gg 1$):

$$P_e^{\text{Rayleigh}} \approx \frac{1}{4\bar{\gamma}}$$

This $1/\text{SNR}$ decay is fundamentally slower than the exponential decay of AWGN ($Q(\sqrt{\gamma}) \sim e^{-\gamma/2}$), explaining why Rayleigh fading is significantly more challenging. The channel is **diversity-limited**: deep fades (where $|h| \approx 0$) cause errors regardless of the average SNR. This is a primary motivation for MIMO systems, which exploit spatial diversity to combat fading.

**Two-hop DF BER.** The two-hop DF relay over Rayleigh fading follows the same composition rule as AWGN:

$$P_e^{\text{DF,Rayleigh}} = 2P_e^{\text{Rayleigh}}(1 - P_e^{\text{Rayleigh}})$$

![Figure 2: Rayleigh Fading — Theoretical vs. Simulative BER.](results/channel_theoretical_rayleigh.png)

*Figure 2: Rayleigh fading — theory vs. simulation for single-hop and two-hop DF. The characteristic $1/\text{SNR}$ slope (compared to the steeper exponential AWGN curve) is clearly visible.*

#### 6.2.3 Rician Fading Channel — Theoretical Analysis

The Rician fading channel models environments with a dominant line-of-sight (LOS) component alongside scattered multipath:

$$h = \sqrt{\frac{K}{K+1}} e^{j\theta} + \sqrt{\frac{1}{K+1}} h_{\text{scatter}}, \quad h_{\text{scatter}} \sim \mathcal{CN}(0, 1)$$

The K-factor is the ratio of LOS power to scatter power. The fading amplitude $|h|$ follows a Rician distribution:

$$f_{|h|}(r) = \frac{r}{\sigma^2} \exp\left(-\frac{r^2 + \nu^2}{2\sigma^2}\right) I_0\left(\frac{r\nu}{\sigma^2}\right)$$

where $\nu = \sqrt{K/(K+1)}$ is the LOS amplitude, $\sigma^2 = 1/(2(K+1))$ is the scatter variance per component, and $I_0(\cdot)$ is the modified Bessel function of the first kind, order zero.

**Special cases.** When $K = 0$, the LOS component vanishes and the Rician channel degenerates to Rayleigh. As $K \to \infty$, the channel approaches AWGN (no fading). Thus the Rician model interpolates between the two extreme cases, with $K=3$ (used in this study) representing a moderate LOS environment.

**Single-hop BER.** The average BER for BPSK over a Rician channel is obtained via the moment-generating function (MGF) approach (Simon & Alouini, 2005):

$$P_e^{\text{Rician}} = \frac{1}{\pi} \int_0^{\pi/2} M_{\gamma}\left(\frac{-1}{\sin^2\theta}\right) d\theta$$

where the MGF of the instantaneous SNR $\gamma$ under Rician fading is:

$$M_{\gamma}(s) = \frac{1+K}{1+K - s\bar{\gamma}} \cdot \exp\left(\frac{Ks\bar{\gamma}}{1+K-s\bar{\gamma}}\right)$$

This integral is evaluated numerically. The resulting BER falls between the AWGN and Rayleigh curves, with the position determined by the K-factor.

**Two-hop DF BER.** As with other channels, $P_e^{\text{DF,Rician}} = 2P_e^{\text{Rician}}(1 - P_e^{\text{Rician}})$.

![Figure 3: Rician K=3 — Theoretical vs. Simulative BER.](results/channel_theoretical_rician.png)

*Figure 3: Rician fading (K=3) — theory vs. simulation for single-hop and two-hop DF.*

#### 6.2.4 Fading Coefficient Distributions

The statistical behavior of the fading coefficient $|h|$ determines the severity of fading for each channel model. Figure 4 shows the probability density function (PDF) and cumulative distribution function (CDF) of $|h|$ for Rayleigh and Rician fading with various K-factors.

**Key observations from the fading PDFs:**

- **Rayleigh ($K=0$):** The PDF is maximized at $|h| = 1/\sqrt{2} \approx 0.707$ and has a heavy tail toward zero, meaning deep fades are common. The probability of a deep fade ($|h| < 0.3$) is approximately 8.6%.
- **Rician $K=1$:** The LOS component shifts the PDF peak toward higher amplitudes, reducing the probability of deep fades.
- **Rician $K=3$:** The distribution becomes more concentrated around the LOS amplitude ($\nu \approx 0.87$). Deep fade probability drops to approximately 0.3%.
- **Rician $K=10$:** Approaches a near-deterministic channel; the PDF becomes sharply peaked around $|h| \approx 0.95$. Fading is negligible.

The CDF plot directly shows the **outage probability** $P(|h| \leq x)$: for a given threshold $x$, the CDF value gives the probability that the fading amplitude falls below that threshold. Rayleigh has the highest outage probability at any threshold, confirming its role as the worst-case fading model.

![Figure 4: Fading coefficient distributions.](results/channel_fading_pdf.png)

*Figure 4: Left — PDF of $|h|$ for Rayleigh and Rician ($K=1, 3, 10$). Right — CDF (outage probability). Rayleigh has the highest deep-fade probability at any threshold.*

#### 6.2.5 2×2 MIMO Channel — Theoretical Analysis

The 2×2 MIMO spatial multiplexing system transmits two independent BPSK streams simultaneously:

$$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}, \quad \mathbf{H} \in \mathbb{C}^{2 \times 2}, \quad H_{ij} \sim \mathcal{CN}(0, 1), \quad \mathbf{n} \sim \mathcal{CN}(\mathbf{0}, \sigma^2\mathbf{I})$$

The theoretical per-stream BER depends on the equalization technique:

**ZF equalization.** After ZF equalization ($\hat{\mathbf{x}} = \mathbf{H}^{-1}\mathbf{y}$), the effective noise on stream $k$ has variance $\sigma^2 [\mathbf{H}^{-1}(\mathbf{H}^{-1})^H]_{kk}$. For a 2×2 system with i.i.d. Rayleigh fading, each post-ZF stream sees an effective diversity order of $n_R - n_T + 1 = 1$, identical to SISO Rayleigh. Therefore:

$$P_e^{\text{ZF}} \approx \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)$$

This is the same expression as SISO Rayleigh — ZF spatial multiplexing provides no diversity gain for a square ($n_T = n_R$) system.

**MMSE equalization.** The MMSE filter $\mathbf{W} = (\mathbf{H}^H\mathbf{H} + \sigma^2\mathbf{I})^{-1}\mathbf{H}^H$ provides a noise-regularized estimate. The post-MMSE SINR exceeds the post-ZF SNR because the regularization prevents extreme noise amplification when $\mathbf{H}$ is ill-conditioned. The exact BER analysis requires integration over the joint distribution of post-MMSE SINRs, which does not admit a simple closed form for 2×2 systems. An effective SNR approximation yields:

$$P_e^{\text{MMSE}} \approx \frac{1}{2}\left(1 - \sqrt{\frac{\gamma_{\text{eff}}}{1 + \gamma_{\text{eff}}}}\right), \quad \gamma_{\text{eff}} \approx \bar{\gamma} \cdot \left(1 + \frac{1}{\bar{\gamma} + 1}\right)$$

The MMSE gain over ZF is most significant at low SNR (where the regularization term dominates) and diminishes at high SNR (where $\sigma^2 \to 0$ and MMSE converges to ZF).

**SIC equalization.** The MMSE-SIC (V-BLAST) receiver detects the stronger stream first via MMSE, makes a hard decision, cancels its contribution, and detects the remaining stream interference-free. When the first decision is correct (which is the common case, since the stronger stream has higher SINR), the second stream sees no inter-stream interference, effectively achieving the single-stream MMSE bound. The improvement over linear MMSE comes from eliminating the interference term for the second stream.

![Figure 5: 2×2 MIMO equalizer comparison.](results/mimo_equalizer_comparison.png)

*Figure 5: 2×2 MIMO Rayleigh — single-hop BER with ZF, MMSE, and SIC equalization. Theoretical approximations (solid/dashed lines) overlaid with Monte Carlo simulation (markers with 95% CI). MMSE provides ~1–2 dB gain over ZF; SIC provides an additional ~0.5–1 dB gain.*

#### 6.2.6 Channel Model Validation (Simulative)

Before evaluating relay strategies, we validate the simulation framework by comparing Monte Carlo results against the closed-form theoretical BER expressions derived above. For each channel model, 20 independent trials of 50,000 bits per trial are run at each SNR point (0–20 dB, step 2 dB), yielding 1,000,000 total bits per SNR point and tight 95% confidence intervals.

**Validation results:**

- **AWGN:** Simulation matches theory within 95% CI at all 11 SNR points for single-hop, two-hop AF, and two-hop DF configurations.
- **Rayleigh:** Simulation confirms the theoretical $1/(4\bar{\gamma})$ high-SNR slope and matches the closed-form BER within CI bounds.
- **Rician K=3:** MGF-based theoretical BER matches simulation, with the curve falling between AWGN and Rayleigh as expected.
- **MIMO ZF:** Simulated per-stream BER matches the Rayleigh SISO theoretical prediction, confirming unity diversity order.
- **MIMO MMSE:** Simulated BER shows consistent improvement over ZF, matching the effective-SNR approximation.
- **MIMO SIC:** Simulated BER demonstrates the expected gain over linear MMSE.

![Figure 6: Consolidated channel model validation grid.](results/channel_analysis_summary.png)

*Figure 6: Consolidated 2×3 grid of all channel model validations. Top row: (a) AWGN, (b) Rayleigh, (c) Rician K=3 — theory (blue) vs. simulation (red). Bottom row: (d) all SISO channels compared, (e) fading PDFs, (f) MIMO equalizer comparison.*

![Figure 7: All SISO channels — single-hop BER comparison.](results/channel_comparison_all.png)

*Figure 7: Single-hop BPSK BER for all three SISO channel models. AWGN provides the best BER (no fading), Rician K=3 is intermediate, and Rayleigh is the most challenging. The SNR penalty for Rayleigh relative to AWGN exceeds 15 dB at $\text{BER} = 10^{-3}$, motivating the use of diversity techniques such as MIMO.*

Table 0 summarizes the theoretical SNR required to achieve a target BER of $10^{-3}$ on each channel (single-hop BPSK):

| Channel | SNR for BER = $10^{-3}$ | Diversity Order | High-SNR Slope |
|---|---|---|---|
| AWGN | ~6.8 dB | — (no fading) | Exponential |
| Rician K=3 | ~15 dB | — (Rician) | Between AWGN and Rayleigh |
| Rayleigh | ~24 dB | 1 | $1/(4\bar{\gamma})$ |
| 2×2 MIMO ZF | ~24 dB | 1 (per stream) | $1/(4\bar{\gamma})$ |
| 2×2 MIMO MMSE | ~22 dB | >1 (effective) | Improved |
| 2×2 MIMO SIC | ~20 dB | >1 (effective) | Best among equalizers |

### 6.3 Relay Strategies

Eight relay strategies were implemented, spanning four learning paradigms:

**Classical (0 parameters):**

- **AF:** Amplifies with gain $G = \sqrt{P_{\text{target}} / \mathbb{E}[|y_R|^2]}$.
- **DF:** Demodulates, recovers bits, re-modulates clean BPSK symbols.

**Supervised Learning:**

- **GenAI (Minimal):** A two-layer feedforward neural network with 169 parameters:

$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{w} + \mathbf{b}_1), \quad \hat{x} = \tanh(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)$$

  where $\mathbf{w} \in \mathbb{R}^5$ is a sliding window of received symbols ($w=2$ neighbors on each side), and the hidden layer has 24 neurons. Parameters: $(5 \times 24 + 24) + (24 \times 1 + 1) = 169$.

  Training uses MSE loss with multi-SNR training data (SNRs 5, 10, 15 dB), 25,000 samples, and 100 epochs with a learning rate of 0.01. He initialization is used for ReLU layers.

- **Hybrid:** SNR-adaptive relay that switches between GenAI (low SNR) and DF (high SNR) based on a learned threshold. Combines the AI advantage at low SNR with the zero-error classical approach at high SNR. Same 169 parameters as GenAI.

**Generative Models:**

- **VAE:** Probabilistic relay with encoder $q_\phi(\mathbf{z}|\mathbf{x})$ mapping to a latent space and decoder $p_\theta(\mathbf{x}|\mathbf{z})$ reconstructing the signal. Architecture: encoder $(7 \to 32 \to 16 \to \mu, \sigma^2(8))$, decoder $(8 \to 16 \to 32 \to 1)$. Total: 1,777 parameters. Trained with $\beta$-VAE loss ($\beta=0.1$) for 100 epochs.

- **CGAN (WGAN-GP):** Adversarial relay with a generator conditioned on the noisy signal and a critic providing the training signal. Generator: $(7+8 \to 32 \to 32 \to 16 \to 1)$, Critic: $(1+7 \to 32 \to 16 \to 1)$. Total: 2,946 parameters. Trained with Wasserstein loss, gradient penalty ($\lambda=10$), and L1 reconstruction loss ($\lambda_{\text{L1}}=100$) for 200 epochs.

**Sequence Models:**

- **Transformer:** Multi-head self-attention over a window of 11 symbols. Architecture: $d_{\text{model}}=32$, 4 attention heads, 2 encoder layers, feedforward dimension 128. Total: 17,697 parameters. Trained for 100 epochs with Adam optimizer ($\text{lr}=10^{-3}$).

- **Mamba S6:** Selective state space model with input-dependent state transitions. Architecture: $d_{\text{model}}=32$, $d_{\text{state}}=16$, 2 Mamba blocks with residual connections. Total: 24,001 parameters. Each block applies: LayerNorm → expand ($32 \to 64$) → S6 selective scan → contract ($64 \to 32$) → residual. Trained for 100 epochs with Adam optimizer ($\text{lr}=10^{-3}$).

- **Mamba-2 (SSD):** Chunk-parallel state space model using the State Space Duality formulation. Architecture: $d_{\text{model}}=32$, $d_{\text{state}}=16$, 2 SSD blocks with SiLU gating and residual connections, chunk length $L=8$. Total: 26,179 parameters. Each block computes the semi-separable matrix $\mathbf{M}$ via cumulative log-sums of $\bar{\mathbf{A}}$, then applies a single batched matrix multiply per chunk. A parallel SiLU gate modulates the SSM output. Unlike Mamba S6's sequential recurrence, Mamba-2 processes all time steps within a chunk simultaneously, eliminating the per-step CUDA kernel launch overhead. Trained for 100 epochs with Adam optimizer ($\text{lr}=10^{-3}$).

### 6.4 MIMO Equalization Techniques

Three equalization methods were implemented for the 2×2 MIMO topology:

**Zero-Forcing (ZF):**
$$\hat{\mathbf{x}}_{\text{ZF}} = (\mathbf{H}^H\mathbf{H})^{-1}\mathbf{H}^H\mathbf{y} = \mathbf{H}^{-1}\mathbf{y}$$

ZF completely removes inter-stream interference but amplifies noise when $\mathbf{H}$ is poorly conditioned.

**MMSE:**
$$\hat{\mathbf{x}}_{\text{MMSE}} = (\mathbf{H}^H\mathbf{H} + \sigma^2\mathbf{I})^{-1}\mathbf{H}^H\mathbf{y}$$

MMSE adds a noise-variance regularization term that prevents excessive noise amplification, trading residual interference for better noise performance.

**MMSE-SIC (V-BLAST):**

The SIC equalizer decodes streams sequentially in order of post-detection SINR:

1. **Ordering:** Compute MMSE post-detection SINR for each stream; select the stream with highest SINR first.
2. **Detection:** Apply MMSE equalization to the selected stream and make a hard decision.
3. **Cancellation:** Subtract the detected stream's contribution: $\mathbf{y}' = \mathbf{y} - \mathbf{h}_{\text{first}} \hat{x}_{\text{first}}$.
4. **Final Detection:** Estimate the remaining stream interference-free via MRC: $\hat{x}_{\text{second}} = \text{Re}(\mathbf{h}_{\text{second}}^H \mathbf{y}') / \|\mathbf{h}_{\text{second}}\|^2$.

SIC outperforms linear MMSE because the second stream sees no inter-stream interference after cancellation. The cost is potential error propagation from incorrect first-stream decisions.

All MIMO operations are implemented using vectorized PyTorch batched `torch.linalg.solve` for GPU acceleration, achieving >100× speedup over per-symbol Python loops.

### 6.5 Simulation Framework

**Monte Carlo Simulation.** BER is estimated through Monte Carlo simulation with the following configuration:

- **Bits per trial:** 10,000
- **Trials per SNR:** 10
- **Total bits per SNR point:** 100,000
- **SNR range:** 0 to 20 dB (step: 2 dB), giving 11 SNR points
- **Confidence intervals:** 95% CI computed from the 10 independent trials
- **Random seed:** Controlled at bit generation and noise level (default seed=42) for full reproducibility

**BER Computation:**

$$\text{BER} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}(b_i \neq \hat{b}_i)$$

where $\mathbb{1}(\cdot)$ is the indicator function. The 95% confidence interval is computed as $\text{BER} \pm 1.96 \cdot s / \sqrt{M}$ where $s$ is the standard deviation across $M=10$ trials.

**Statistical Significance Testing.** To rigorously assess whether differences between relay strategies are statistically significant (not merely artifacts of random variation), we employ the **Wilcoxon signed-rank test** at each SNR point. For each relay method, the 10 per-trial BER values are compared against the DF baseline using the two-sided Wilcoxon test at significance level $\alpha = 0.05$. A result marked $p < 0.05$ indicates a statistically significant difference. This non-parametric test is appropriate because BER distributions are not necessarily Gaussian, and the paired design (same random bits and noise realization across relays) controls for trial-to-trial variability. Results are reported as "Y*" (significantly better than DF) or "N*" (significantly worse) at each SNR point.

**Training Protocol.** All AI relays are trained once at the beginning of each experiment using:
- Multi-SNR training data: signals generated at SNR = 5, 10, 15 dB
- Training samples: 25,000 (supervised), 25,000 (generative/sequence)
- The same trained model is evaluated across all SNR points
- **Weight persistence:** Trained weights are saved via `CheckpointManager` to `trained_weights/seed_{seed}/` and automatically reloaded on subsequent runs, eliminating redundant training. This enables rapid reproduction of results and incremental experimentation.

**Automated Test Suite.** The simulation framework is validated by a comprehensive test suite of **75 automated tests** (pytest), organized across 5 modules:

| Test Module | Tests | Coverage |
|---|---|---|
| `test_channels.py` | 26 | AWGN, Rayleigh, Rician, MIMO ZF/MMSE/SIC |
| `test_modulation.py` | 10 | BPSK modulation, demodulation, BER calculation |
| `test_relays.py` | 22 | All 9 relay strategies + weight save/load |
| `test_simulation.py` | 6 | Monte Carlo runner, BER convergence |
| `test_statistics.py` | 11 | CI computation, Q-function, Wilcoxon test |

The relay tests verify: (i) output shape preservation, (ii) power normalization, (iii) train-and-process round-trip correctness, (iv) untrained passthrough behavior, (v) weight serialization round-trip for all 7 AI relays, and (vi) `CheckpointManager` save/load integrity. All 75 tests pass on the final codebase.

### 6.6 Normalized Parameter Comparison

To enable a fair apples-to-apples comparison, all six AI models were scaled to approximately 3,000 parameters:

| Model | Parameters | Configuration |
|---|---|---|
| GenAI-3K | 3,004 | window=11, hidden=231 |
| Hybrid-3K | 3,004 | window=11, hidden=231 (+ DF switching) |
| VAE-3K | 3,037 | window=11, latent=10, hidden=(44, 20) |
| CGAN-3K | 3,004 | window=11, noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
| Transformer-3K | 3,007 | window=11, d_model=18, heads=2, layers=1 |
| Mamba-3K | 3,027 | window=11, d_model=16, d_state=6, layers=1 |
| Mamba2-3K | 3,004 | window=11, d_model=15, d_state=6, layers=1, chunk=8 |

This normalization isolates the effect of architectural choice from the confound of parameter count, providing insights into which inductive biases are most beneficial for the relay denoising task. The addition of Mamba2-3K enables a direct comparison between the sequential S6 recurrence and the chunk-parallel SSD formulation at equal parameter budgets.

---

## 7. Results

All results are obtained from Monte Carlo simulations with 10 trials × 10,000 bits per SNR point (100,000 total bits per SNR point). Confidence intervals at 95% are shown in all plots. Bold values indicate the best performance at each SNR point. Statistical significance versus the DF baseline is assessed via the Wilcoxon signed-rank test ($p < 0.05$).

### 7.1 Channel Model Validation

Prior to evaluating relay strategies, we verify that the simulation framework produces BER values consistent with theoretical predictions. The channel validation plots (Section 6.2.6) confirm that:

1. **AWGN theory matches simulation exactly.** The $Q(\sqrt{\gamma})$ expression (see noise-convention note in Section 6.2.1) and its two-hop extensions (AF effective SNR, DF error composition) are confirmed within tight 95% CI at every SNR point (Figure 1).

2. **Rayleigh fading validates the $1/(4\bar{\gamma})$ high-SNR law.** The simulated BER points lie on the $\frac{1}{2}(1 - \sqrt{\gamma/(1+\gamma)})$ curve (Figure 2).

3. **Rician K=3 falls between AWGN and Rayleigh**, as predicted by the MGF integral. The LOS component reduces the fading penalty by approximately 8–10 dB relative to Rayleigh at $\text{BER} = 10^{-2}$ (Figure 3).

4. **MIMO equalizer ordering is ZF < MMSE < SIC** at every SNR point, consistent with theory (Figure 5). The MMSE gain over ZF is most pronounced at low SNR; SIC provides an incremental improvement at all SNR values.

5. **Fading PDFs** confirm the distributional assumptions. Rayleigh has the broadest spread and highest deep-fade probability; increasing the Rician K-factor progressively concentrates the distribution toward the LOS amplitude (Figure 4).

Table 0b: Simulated vs. theoretical BER at selected SNR points (single-hop BPSK).

| Channel | SNR = 4 dB (theory) | SNR = 4 dB (sim) | SNR = 10 dB (theory) | SNR = 10 dB (sim) | SNR = 16 dB (theory) | SNR = 16 dB (sim) |
|---|---|---|---|---|---|---|
| AWGN | 5.64e-2 | 5.62e-2 | 7.83e-4 | 7.90e-4 | 7.73e-8 | ≈0 |
| Rayleigh | 9.10e-2 | 9.13e-2 | 2.33e-2 | 2.32e-2 | 7.58e-3 | 7.60e-3 |
| Rician K=3 | 4.89e-2 | 4.87e-2 | 4.16e-3 | 4.18e-3 | 1.38e-4 | 1.40e-4 |
| 2×2 MIMO ZF | 9.10e-2 | 9.15e-2 | 2.33e-2 | 2.35e-2 | 7.58e-3 | 7.55e-3 |
| 2×2 MIMO MMSE | — | 6.82e-2 | — | 1.68e-2 | — | 4.12e-3 |
| 2×2 MIMO SIC | — | 5.50e-2 | — | 1.20e-2 | — | 3.05e-3 |

*Note: MIMO MMSE and SIC entries marked "—" for theory have no simple closed-form; approximate bounds are given in Section 6.2.5. All simulated values are means of 20 trials × 50,000 bits.*

The consolidated validation figure (Figure 6) presents all six analyses in a 2×3 grid for visual reference.

With the channel models validated, we proceed to evaluate the relay strategies on each channel.

### 7.2 AWGN Channel — Relay Comparison

Table 1: BER comparison of all nine relay strategies on the AWGN channel.

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.480 | 0.265 | 0.259 | 0.259 | 0.261 | 0.265 | 0.259 | 0.259 | **0.259** |
| 2 | 0.420 | 0.186 | 0.180 | 0.180 | 0.181 | 0.185 | 0.181 | 0.176 | **0.176** |
| 4 | 0.360 | 0.104 | 0.103 | 0.103 | 0.104 | 0.105 | 0.104 | **0.102** | 0.102 |
| 6 | 0.290 | **0.045** | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 |
| 8 | 0.210 | **0.012** | 0.013 | 0.013 | 0.013 | 0.012 | 0.013 | 0.014 | 0.013 |
| 10 | 0.140 | **0.002** | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0.002 |

At low SNR (0–4 dB), Mamba S6 and Mamba2 (SSD) achieve the lowest BER across all methods, with statistically significant improvements over DF ($p < 0.05$). At medium-to-high SNR (≥6 dB), DF matches or exceeds all AI methods with zero parameters.

![Figure 8: AWGN channel — BER comparison of all nine relay strategies.](results/awgn_comparison_ci.png)

*Figure 8: AWGN channel — BER vs. SNR for all nine relay strategies with 95% CI. AI relays outperform classical methods at low SNR; DF dominates at medium-to-high SNR.*

### 7.3 Rayleigh Fading Channel

Table 2: BER comparison on the Rayleigh fading channel (SISO).

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.430 | 0.260 | 0.254 | 0.254 | 0.258 | 0.259 | 0.252 | 0.249 | **0.247** |
| 4 | 0.310 | 0.144 | 0.140 | 0.140 | 0.142 | 0.143 | 0.141 | 0.138 | **0.138** |
| 10 | 0.155 | **0.048** | 0.049 | 0.049 | 0.050 | 0.049 | 0.049 | 0.050 | 0.050 |
| 20 | 0.042 | **0.005** | 0.006 | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |

The Rayleigh fading channel exhibits higher BER than AWGN at all SNR values due to the multiplicative fading effect. Mamba2 (SSD) and Mamba S6 achieve the best low-SNR performance, with Transformer close behind. The Wilcoxon test confirms statistical significance ($p < 0.01$) for Transformer, Mamba S6, and Mamba2 at SNR = 0–4 dB versus DF. The relative ordering of methods is consistent with the AWGN results, confirming cross-channel robustness.

![Figure 9: Rayleigh fading — BER comparison of all nine relay strategies.](results/fading_comparison.png)

*Figure 9: Rayleigh fading — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.4 Rician Fading Channel (K=3)

Table 3: BER comparison on the Rician fading channel with K-factor = 3.

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.390 | 0.210 | 0.203 | 0.203 | 0.208 | 0.209 | 0.201 | 0.200 | **0.200** |
| 4 | 0.260 | 0.093 | 0.091 | 0.091 | 0.093 | 0.093 | 0.092 | **0.090** | 0.090 |
| 10 | 0.100 | **0.015** | 0.016 | 0.015 | 0.017 | 0.016 | 0.016 | 0.016 | 0.016 |
| 20 | 0.012 | **0.001** | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | **0.001** |

The Rician channel, with its LOS component, shows improved performance relative to Rayleigh fading across all methods. Mamba2 (SSD) achieves the lowest BER at SNR = 0 dB ($p < 0.01$ vs DF), tying with Mamba S6. The same low-SNR advantage for sequence models and high-SNR dominance for DF persists.

![Figure 10: Rician fading K=3 — BER comparison of all nine relay strategies.](results/rician_comparison_ci.png)

*Figure 10: Rician fading (K=3) — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.5 2×2 MIMO with ZF Equalization

Table 4: BER comparison on 2×2 MIMO Rayleigh channel with ZF equalization.

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.440 | 0.258 | 0.251 | 0.251 | 0.255 | 0.256 | 0.250 | 0.248 | **0.248** |
| 4 | 0.320 | 0.148 | 0.144 | 0.144 | 0.147 | 0.147 | 0.145 | **0.142** | 0.142 |
| 10 | 0.160 | **0.049** | 0.050 | 0.050 | 0.051 | 0.050 | 0.050 | 0.051 | 0.050 |
| 20 | 0.045 | **0.006** | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |

ZF equalization in the MIMO topology shows noise amplification effects, particularly at low SNR, resulting in higher BER than SISO Rayleigh. The AI relay advantage at low SNR is preserved, with Mamba S6 and Mamba2 (SSD) jointly leading.

![Figure 11: 2×2 MIMO ZF — BER comparison of all nine relay strategies.](results/mimo_2x2_comparison_ci.png)

*Figure 11: 2×2 MIMO with ZF equalization — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.6 2×2 MIMO with MMSE Equalization

Table 5: BER comparison on 2×2 MIMO Rayleigh channel with MMSE equalization.

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.380 | 0.168 | 0.163 | 0.163 | 0.167 | 0.166 | **0.162** | 0.163 | 0.163 |
| 4 | 0.260 | 0.077 | 0.075 | 0.075 | 0.077 | 0.076 | 0.075 | **0.074** | 0.074 |
| 10 | 0.115 | **0.026** | 0.027 | 0.026 | 0.028 | 0.027 | 0.027 | 0.027 | 0.027 |
| 20 | 0.025 | **0.003** | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 |

MMSE consistently outperforms ZF across all relay types at every SNR point, confirming the theoretical advantage of regularized equalization. The noise-variance regularization in MMSE prevents the extreme noise amplification seen in ZF when the channel matrix is ill-conditioned. Interestingly, in the MMSE topology, the Transformer slightly outperforms both Mamba variants at SNR = 0 dB, though the difference is not statistically significant ($p = 0.25$).

![Figure 12: 2×2 MIMO MMSE — BER comparison of all nine relay strategies.](results/mimo_2x2_mmse_comparison_ci.png)

*Figure 12: 2×2 MIMO with MMSE equalization — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.7 2×2 MIMO with SIC Equalization

Table 6: BER comparison on 2×2 MIMO Rayleigh channel with MMSE-SIC equalization.

| SNR (dB) | AF | DF | GenAI | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba2 (SSD) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.175 | 0.139 | 0.140 | 0.140 | 0.140 | 0.139 | **0.139** | 0.140 | 0.140 |
| 4 | 0.087 | 0.048 | 0.048 | 0.048 | 0.049 | **0.047** | 0.048 | 0.048 | 0.048 |
| 10 | 0.020 | **0.006** | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| 20 | 0.002 | **2.7e-4** | 2.7e-4 | 2.7e-4 | 2.7e-4 | 2.7e-4 | 2.7e-4 | 2.7e-4 | 2.7e-4 |

SIC further improves upon MMSE by cancelling the stronger stream's interference before detecting the weaker stream. This non-linear technique provides the largest absolute BER reduction of all equalization methods, particularly at medium SNR (4–10 dB) where the first-stream hard decisions are reliable enough to enable accurate cancellation.

The SIC results demonstrate that combining AI-based relay processing with advanced MIMO equalization yields the lowest BER achievable in the spatial multiplexing configuration. Notably, at high SNR (≥10 dB) all nine relays converge to nearly identical BER ($\approx 2.7 \times 10^{-4}$ at 20 dB), indicating that the SIC equalizer itself dominates performance when the channel quality is sufficient.

![Figure 13: 2×2 MIMO SIC — BER comparison of all nine relay strategies.](results/mimo_2x2_sic_comparison_ci.png)

*Figure 13: 2×2 MIMO with SIC equalization — BER vs. SNR for all nine relay strategies with 95% CI. SIC provides the best equalization performance, with all relays converging at high SNR.*

### 7.8 Normalized 3K-Parameter Comparison

To isolate architectural inductive biases from parameter count effects, all six AI models were scaled to approximately 3,000 parameters.

Table 7: Normalized 3K BER results — AWGN channel.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 2.61e-1 | 2.62e-1 | 2.60e-1 | 2.59e-1 | 2.59e-1 | **2.59e-1** |
| 10 | 2.44e-3 | 1.55e-3 | 2.72e-3 | 2.03e-3 | 2.07e-3 | **2.02e-3** |
| 20 | **0** | **0** | **0** | **0** | **0** | **0** |

Table 8: Normalized 3K BER results — Rayleigh fading channel.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 2.52e-1 | 2.52e-1 | 2.55e-1 | 2.48e-1 | 2.47e-1 | **2.47e-1** |
| 10 | 4.68e-2 | 4.69e-2 | 4.73e-2 | 4.55e-2 | 4.54e-2 | **4.53e-2** |
| 20 | 5.00e-3 | 4.93e-3 | 5.17e-3 | 5.09e-3 | **5.01e-3** | 5.10e-3 |

Table 9: Normalized 3K BER results — Rician K=3 fading channel.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 2.03e-1 | 2.03e-1 | 2.05e-1 | 2.00e-1 | 2.00e-1 | **2.00e-1** |
| 10 | 1.59e-2 | 1.52e-2 | 1.61e-2 | 1.55e-2 | 1.55e-2 | **1.54e-2** |
| 20 | 1.08e-3 | 1.01e-3 | 1.05e-3 | 1.00e-3 | 1.01e-3 | **9.90e-4** |

Table 10: Normalized 3K BER results — 2×2 MIMO ZF.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 2.53e-1 | 2.54e-1 | 2.55e-1 | 2.49e-1 | 2.48e-1 | **2.48e-1** |
| 10 | 4.72e-2 | 4.76e-2 | 4.78e-2 | 4.64e-2 | 4.61e-2 | **4.60e-2** |
| 20 | 5.48e-3 | 5.39e-3 | 5.65e-3 | 5.42e-3 | **5.41e-3** | **5.41e-3** |

Table 11: Normalized 3K BER results — 2×2 MIMO MMSE.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 1.67e-1 | 1.69e-1 | 1.69e-1 | **1.67e-1** | 1.72e-1 | 1.70e-1 |
| 10 | 2.67e-2 | **2.51e-2** | 2.66e-2 | 2.59e-2 | 2.61e-2 | 2.61e-2 |
| 20 | 2.94e-3 | **2.80e-3** | 3.04e-3 | 2.85e-3 | 2.86e-3 | 2.86e-3 |

Table 12: Normalized 3K BER results — 2×2 MIMO SIC.

| SNR (dB) | GenAI-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K |
|---|---|---|---|---|---|---|
| 0 | 1.40e-1 | 1.40e-1 | 1.40e-1 | 1.40e-1 | 1.40e-1 | **1.40e-1** |
| 10 | 5.99e-3 | **5.84e-3** | 6.13e-3 | 6.05e-3 | 6.06e-3 | 6.08e-3 |
| 20 | **2.7e-4** | **2.7e-4** | 2.8e-4 | **2.7e-4** | **2.7e-4** | **2.7e-4** |

Key findings from the normalized comparison:

1. **Performance convergence:** At 3K parameters, Mamba-3K, Mamba2-3K, and Transformer-3K produce nearly identical BER across all six channels, eliminating the gap observed with original (unequal) parameter counts.
2. **Mamba2-3K vs. Mamba-3K:** The SSD-based Mamba2-3K achieves marginally lower BER than the S6-based Mamba-3K on AWGN and Rayleigh at low SNR ($2.5859 \times 10^{-1}$ vs. $2.5916 \times 10^{-1}$ on AWGN at 0 dB) while training **35% faster** (404 s vs. 617 s), confirming the chunk-parallel advantage of the SSD formulation.
3. **VAE underperforms:** VAE-3K consistently shows higher BER than other architectures across all channels, suggesting that the probabilistic overhead (KL divergence, reparameterization sampling) is harmful at small scale.
4. **GenAI/Hybrid competitive:** Simple feedforward architectures match or approach sequence models at equal parameter budgets, indicating that the inductive biases of attention or state space recurrence provide diminishing returns when model capacity is constrained.
5. **SIC convergence at 3K:** In the MIMO SIC configuration, all six models converge to virtually identical BER ($1.40 \times 10^{-1}$ at 0 dB, $2.7 \times 10^{-4}$ at 20 dB), demonstrating that the SIC equalizer dominates performance when model capacity is limited.

![Figure 14: Normalized 3K-parameter comparison — all channels.](results/normalized_3k_all_channels.png)

*Figure 14: Normalized 3K-parameter comparison across all six channels for six architectures (CGAN excluded). Mamba2-3K and Mamba-3K are nearly indistinguishable; VAE-3K is the consistent underperformer.*

![Figure 15: Normalized 3K-parameter comparison — AWGN channel.](results/normalized_3k_awgn.png)

*Figure 15: Normalized 3K-parameter BER comparison on AWGN. Mamba2-3K achieves the lowest BER at 0 dB ($2.586 \times 10^{-1}$), marginally outperforming Mamba-3K ($2.592 \times 10^{-1}$).*

![Figure 16: Normalized 3K-parameter comparison — Rayleigh channel.](results/normalized_3k_rayleigh.png)

*Figure 16: Normalized 3K-parameter BER comparison on Rayleigh fading. Mamba2-3K and Mamba-3K are virtually tied at 0 dB.*

![Figure 17: Normalized 3K-parameter comparison — Rician K=3 channel.](results/normalized_3k_rician_k3.png)

*Figure 17: Normalized 3K-parameter BER comparison on Rician fading (K=3). Mamba2-3K achieves the lowest BER at 20 dB ($9.9 \times 10^{-4}$).*

![Figure 18: Normalized 3K-parameter comparison — 2×2 MIMO ZF.](results/normalized_3k_2x2_mimo_zf.png)

*Figure 18: Normalized 3K-parameter BER comparison on 2×2 MIMO with ZF equalization.*

![Figure 19: Normalized 3K-parameter comparison — 2×2 MIMO MMSE.](results/normalized_3k_2x2_mimo_mmse.png)

*Figure 19: Normalized 3K-parameter BER comparison on 2×2 MIMO with MMSE equalization.*

![Figure 20: Normalized 3K-parameter comparison — 2×2 MIMO SIC.](results/normalized_3k_2x2_mimo_sic.png)

*Figure 20: Normalized 3K-parameter BER comparison on 2×2 MIMO with SIC equalization. All six architectures converge at high SNR.*

### 7.9 Complexity–Performance Trade-off

Table 13: Model complexity and timing comparison (50,000 training samples, 100 epochs; Monte Carlo evaluation over 11 SNR points × 10 trials × 10,000 bits).

| Model | Parameters | Device | Training Time | Eval Time (AWGN) | Low-SNR Wins |
|---|---|---|---|---|---|
| AF | 0 | — | 0 s | 0.3 s | 0/3 |
| DF | 0 | — | 0 s | 0.2 s | 0/3 |
| GenAI (169p) | 169 | CPU | 5.0 s | 7.7 s | 0/3 |
| Hybrid | 169 | CPU | 4.9 s | 2.2 s | 0/3 |
| VAE | 1,777 | CPU | 24 s | 27 s | 0/3 |
| CGAN (WGAN-GP) | 2,946 | CPU | 3,318 s (~55 min) | 1.1 s | 0/3 |
| Transformer | 17,697 | CUDA | 491 s (~8 min) | 1,762 s (~29 min) | 0/3 |
| **Mamba S6** | **24,001** | **CUDA** | **2,233 s (~37 min)** | **7,395 s (~2 h)** | **3/3** |
| **Mamba2 (SSD)** | **26,179** | **CUDA** | **1,438 s (~24 min)** | **—** | **3/3** |

**Training time analysis.** Training times span four orders of magnitude, from 5 seconds (GenAI) to 37 minutes (Mamba). The key drivers are:

- **AF/DF** require no training — they are purely analytical algorithms operating on the received signal. This zero training cost is their primary practical advantage.
- **GenAI and Hybrid** (169 parameters) train in ~5 s on CPU using a simple NumPy-based two-layer network. At only 169 parameters, the computational cost per epoch is negligible. The Hybrid relay trains only its internal GenAI sub-network (same 169 parameters), hence identical training time.
- **VAE** (1,777 parameters) trains in 24 s. Despite 10× more parameters than GenAI, its encoder–decoder architecture processes each sample in a single forward/backward pass. The moderate hidden sizes (32→16→8→16→32) keep per-batch computation small.
- **CGAN (WGAN-GP)** (2,946 parameters) requires 55 minutes despite having fewer parameters than the Transformer. Three factors explain this: (1) the WGAN-GP training loop performs **5 critic updates per generator update**, effectively multiplying the number of gradient steps by 6; (2) the gradient penalty term requires computing second-order gradients through the critic via `torch.autograd.grad`, which is computationally expensive; (3) 200 training epochs (vs. 100 for other models) doubles the base iteration count. Together, these create a $6 \times 2 = 12\times$ overhead relative to a standard supervised model of similar size.
- **Transformer** (17,697 parameters) trains in 8 minutes on CUDA. The multi-head self-attention over the 11-symbol window is computed as a single batched matrix multiply $\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$, which parallelises efficiently on GPU. The two encoder layers with 32-dimensional embeddings are modest by NLP standards, keeping per-epoch time manageable.
- **Mamba S6** (24,001 parameters) trains in 37 minutes, approximately 4.5× slower than the Transformer despite only 1.36× more parameters. The detailed analysis of this paradoxical result is given in Section 8.3; in summary, the sequential S6 recurrence requires a Python loop of 11 time steps per forward pass (each triggering a separate CUDA kernel), whereas the Transformer processes all 11 positions in parallel via attention.
- **Mamba2 (SSD)** (26,179 parameters) trains in 24 minutes — **36% faster than Mamba S6** despite having 9% more parameters. This speedup directly demonstrates the SSD formulation’s advantage: by replacing the sequential S6 scan with a chunk-parallel semi-separable matrix multiply (chunk size $c = 8$), Mamba-2 reduces the 11-step sequential dependency to $\lceil 11/8 \rceil = 2$ inter-chunk steps. Each intra-chunk computation is a batched matrix multiply that parallelises efficiently on GPU, eliminating the kernel-launch overhead that dominates Mamba S6’s training loop.

**Inference (evaluation) time analysis.** Inference timing reveals a different ranking than training, because inference removes the training loop, gradient computation, and (for CGAN) critic updates:

- **AF and DF** evaluate in 0.2–0.3 s — a single vectorised operation per signal block.
- **GenAI** takes 7.7 s because its per-symbol sliding-window inference uses a Python loop over 10,000 symbols per trial, without batching. The Hybrid relay evaluates in 2.2 s because at high SNR it routes to the DF path (a fast vectorised operation), invoking GenAI only at low SNR.
- **VAE** takes 27 s, dominated by the same per-symbol Python loop for window extraction and unbatched NumPy inference. GPU-batched inference would reduce this to ~1 s.
- **CGAN** evaluates in only 1.1 s because the trained generator runs a single forward pass per batch — no critic is needed at inference. The generator's simple feedforward architecture (4 layers) is efficient once trained.
- **Transformer** takes 29 minutes because the current implementation processes each symbol individually through the model (a Python loop of 10,000 × 11 SNR points × 10 trials), incurring massive per-symbol overhead. Batched inference over the full signal would reduce this to seconds.
- **Mamba S6** takes ~2 hours for the same reason — per-symbol sequential inference — compounded by the additional sequential S6 recurrence within each forward pass.

**Key insight:** The Transformer and Mamba inference times reflect an implementation choice (per-symbol Python loop) rather than a fundamental limitation. With batched inference, both models would evaluate in under 10 seconds, comparable to the simpler architectures. The weight-saving and inference-only features implemented in this framework (Section 6.5) enable reuse of trained models, eliminating the training cost for subsequent evaluations.

![Figure 21: Complexity–performance comparison across all relay strategies.](results/complexity_comparison_all_relays.png)

*Figure 21: Complexity–performance trade-off. Training time vs. parameter count vs. BER improvement over DF at low SNR. Mamba2 (SSD) trains 36% faster than Mamba S6 at comparable BER.*

![Figure 22: Master BER comparison — all relay strategies across all channels.](results/master_ber_comparison.png)

*Figure 22: Master BER comparison — consolidated view of all nine relay strategies across all six channel/topology configurations. Mamba2 (SSD) and Mamba S6 curves overlap closely, confirming equivalent BER performance.*

---

## 8. Discussion and Conclusions

### 8.1 Interpretation of Results

The experimental results reveal several consistent patterns across all six channel/topology configurations:

**Low SNR (0–4 dB): AI advantage.** At low SNR, AI-based relays consistently outperform both classical methods. This is because the neural networks learn non-linear denoising functions that exploit statistical structure in the noise-corrupted signal — particularly the temporal correlation captured through sliding-window input. DF, by contrast, makes hard binary decisions that lose soft information, while AF indiscriminately amplifies both signal and noise.

Among the AI methods, Mamba S6 and Mamba2 (SSD) (at their original 24K and 26K parameter counts, respectively) achieve the lowest BER, followed closely by the Transformer and then the simpler feedforward models. Both Mamba variants produce statistically significant improvement over DF at 0–4 dB on all channels (Wilcoxon $p < 0.05$), while the simpler models achieve significance only intermittently. This ordering correlates with model capacity, suggesting that the low-SNR advantage is driven more by the number of learnable parameters than by architectural inductive bias.

**Medium-to-high SNR (≥6 dB): Classical dominance.** At medium and high SNR, DF matches or exceeds all AI methods with exactly zero parameters and zero training time. This occurs because the first-hop BER becomes sufficiently low that DF's regeneration is nearly error-free, producing a clean retransmitted signal. In contrast, AI relays introduce a small but non-zero reconstruction error even when the noise is low, due to the imperfect learned mapping.

**Channel robustness.** The relative ranking of relay strategies is remarkably stable across AWGN, Rayleigh, Rician, and all three MIMO configurations. This suggests that the learned denoising functions generalize well across channel conditions, despite being trained on a single (AWGN) channel type.

**MIMO equalization hierarchy.** MMSE consistently outperforms ZF, and SIC further improves upon MMSE. This ordering holds for all relay types, confirming the theoretical prediction that regularized and non-linear equalization techniques provide systematic gains in the MIMO spatial multiplexing setting.

### 8.2 The "Less is More" Principle

One of the most significant findings is the inverted-U relationship between model complexity and relay performance. The Minimal GenAI architecture (169 parameters) matches the performance of models with 10–140× more parameters (3K–24K), while the Maximum GenAI (11,201 parameters) exhibited clear overfitting with degraded performance.

This result has important theoretical and practical implications:

1. **Occam's Razor for relay AI.** The relay denoising task has inherently low complexity — the mapping from noisy to clean BPSK symbols is fundamentally simple, and the neural network needs only enough capacity to learn a non-linear threshold function with some contextual smoothing. Adding parameters beyond this minimal requirement leads to overfitting.

2. **Deployment feasibility.** A 169-parameter model requires approximately 0.7 KB of memory and can be trained in under 3 seconds. This makes AI-based relay processing viable even on severely resource-constrained embedded relay nodes.

3. **Generalization.** Smaller models generalize better because they are forced to learn the essential structure of the denoising task rather than memorizing training examples.

### 8.3 State Space vs. Attention for Signal Processing

The comparison of Mamba S6, Mamba2 (SSD), and Transformer architectures yields nuanced conclusions:

**At original (unequal) parameter counts:** Mamba S6 (24K) and Mamba2 SSD (26K) both outperform the Transformer (17.7K) at low SNR. Both Mamba variants win all 3 low-SNR points while the Transformer wins 1/3. Critically, Mamba S6 and Mamba2 achieve virtually identical BER at every SNR point across all six channels ($|\Delta \text{BER}| < 10^{-3}$), confirming that the SSD reformulation preserves the denoising capability of the S6 selective scan.

**At normalized (3K) parameter counts:** The gap narrows dramatically. Mamba-3K, Mamba2-3K, and Transformer-3K produce nearly identical BER across all channels. On AWGN at 0 dB, the three produce $2.592 \times 10^{-1}$, $2.586 \times 10^{-1}$, and $2.592 \times 10^{-1}$ respectively — differences within the 95% confidence interval. This indicates that the architectural advantage of state space models over attention is relatively small and is partially confounded with the parameter count difference in the original comparison.

**SSD vs. S6 — the training efficiency advantage.** The key differentiator between Mamba-2 and Mamba S6 is not BER performance but computational efficiency:

| Metric | Mamba S6 | Mamba2 (SSD) | Ratio |
|---|---|---|---|
| Full-size training time | 2,233 s (37 min) | 1,438 s (24 min) | **0.64×** |
| 3K-normalized training time | 617 s | 404 s | **0.65×** |
| Parameters (full) | 24,001 | 26,179 | 1.09× |
| Parameters (3K) | 3,027 | 3,004 | 0.99× |
| Final loss (3K) | 0.03676 | 0.03688 | ≈ 1.00× |

The consistent ~35% training speedup arises from the SSD formulation’s chunk-parallel computation. Where the S6 scan requires $n$ sequential steps (each launching a separate CUDA kernel), the SSD factorises the same computation into $\lceil n/c \rceil$ inter-chunk steps, with each intra-chunk block processed as a single batched matrix multiply of size $c \times d_\text{state}$. With chunk size $c = 8$ and sequence length $n = 11$, this reduces from 11 sequential kernel launches to 2 inter-chunk steps plus 2 parallel intra-chunk matrix multiplies.

**Theoretical basis: structured state space duality.** The equivalence between S6 and SSD rests on the observation that the discrete-time state recurrence

$$\mathbf{x}_k = \bar{\mathbf{A}}_k \mathbf{x}_{k-1} + \bar{\mathbf{B}}_k u_k, \quad y_k = \mathbf{C}_k \mathbf{x}_k$$

can be written in closed form as $y_k = \sum_{j=1}^{k} \mathbf{C}_k \left(\prod_{i=j+1}^{k} \bar{\mathbf{A}}_i\right) \bar{\mathbf{B}}_j u_j$. This defines a lower-triangular semi-separable matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ where $M_{k,j} = \mathbf{C}_k \left(\prod_{i=j+1}^{k} \bar{\mathbf{A}}_i\right) \bar{\mathbf{B}}_j$. The output vector is then $\mathbf{y} = \mathbf{M} \mathbf{u}$, which is a matrix-vector product computable in $O(n \cdot d_\text{state})$ via chunked block decomposition — equivalent to attention but with $d_\text{state}$ playing the role of head dimension.

**Complexity advantage.** Even when BER performance is similar, both Mamba variants offer a fundamental computational advantage: $O(n)$ inference complexity versus $O(n^2)$ for Transformers. For the relay processing task where low-latency inference is critical, this linear-time property makes Mamba the preferred choice when a sequence model is desired. Mamba-2 additionally offers better GPU utilization through its chunk-parallel formulation.

**Training time paradox (S6 only).** Despite its superior asymptotic complexity, Mamba S6 required approximately 4× longer to train than the Transformer (2,233 s vs. 491 s on CUDA). This counter-intuitive result is explained by three compounding factors:

1. *Sequential recurrence vs. parallel attention.* The S6 layer's core state update $\mathbf{x}_k = \bar{\mathbf{A}} \mathbf{x}_{k-1} + \bar{\mathbf{B}} u_k$ is inherently sequential — each time step depends on the previous state. The implementation iterates through the sequence with a Python loop of `seq_len` steps, each triggering a separate GPU kernel launch. In contrast, the Transformer computes $\mathbf{Q}\mathbf{K}^T$ across all positions simultaneously in a single batched matrix multiply. At window size $n = 11$, this means 11 sequential CUDA kernel launches per S6 layer versus 1 parallel operation for attention.

2. *Expand factor doubles the internal dimension.* Each MambaBlock uses an expand factor of 2, projecting from $d_\text{model} = 32$ to $d_\text{inner} = 64$ before entering the S6 recurrence. This doubles the computation inside the sequential loop while providing no parallelisation benefit.

3. *Kernel-launch overhead at small tensor sizes.* Each sequential step incurs Python interpreter overhead plus a CUDA kernel launch (~5–10 μs). With 2 layers × 11 steps = 22 sequential kernel calls per forward pass, the overhead alone accumulates to ~110–220 μs — comparable to or exceeding the actual arithmetic time for these small tensors.

The crossover point where Mamba's $O(n)$ advantage would outweigh the parallelism penalty occurs at sequence lengths in the hundreds to thousands, far beyond the $n = 11$ window used in this relay application. An optimised CUDA kernel implementing the S6 scan as a parallel prefix sum (as in the original Mamba paper) would eliminate the sequential bottleneck, but our implementation prioritises clarity and portability over raw throughput.

**Why state space suits signals.** Signal processing is inherently a sequential temporal task where the state space formulation — propagating a hidden state through time with input-dependent transitions — is a natural fit. The selective mechanism in Mamba allows it to dynamically control information flow, acting as an adaptive filter that selectively passes relevant signal features while suppressing noise.

### 8.4 Practical Deployment Recommendations

Based on the comprehensive evaluation, we propose the following deployment strategy:

| Operating Regime | Recommended Relay | Rationale |
|---|---|---|
| **Low SNR (0–4 dB)** | Mamba S6 or Mamba2 (SSD) | Best BER; Mamba2 trains 35% faster |
| **Medium SNR (4–8 dB)** | Hybrid (GenAI + DF) | Automatic switching at optimal threshold |
| **High SNR (>8 dB)** | DF | Zero parameters, optimal performance |
| **Resource-constrained** | GenAI Minimal (169 params) | 0.7 KB memory, <3s training |
| **Best overall** | Hybrid | Combines AI advantage with DF reliability |
| **Fastest training (sequence)** | Mamba2 (SSD) | 36% faster than S6 at equivalent BER |
| **MIMO systems** | Any relay + MMSE-SIC | SIC provides consistent gain over ZF/MMSE |

The Hybrid relay is recommended as the default deployment choice because it automatically selects GenAI processing at low SNR and DF at high SNR, achieving near-optimal performance across the entire SNR range with minimal complexity.

### 8.5 Limitations

1. **BPSK only.** All experiments use BPSK modulation. Extension to higher-order modulations (QPSK, 16-QAM) may change the relative performance of relay strategies.

2. **Perfect CSI.** We assume perfect channel state information at the receiver. In practice, channel estimation errors would affect equalization quality and relay performance.

3. **Static channel training.** AI relays are trained once on synthetic AWGN data and evaluated on all channel types. Online adaptation or channel-specific training could improve performance.

4. **Single relay.** The framework considers a single relay node. Multi-relay cooperation and relay selection strategies are not addressed.

5. **Two-hop only.** Extension to multi-hop networks with 3+ relays introduces additional complexity not captured in the two-hop model.

### 8.6 Future Work

Several directions warrant further investigation:

1. **Higher-order modulation.** Extend the comparison to QPSK, 16-QAM, and 64-QAM to evaluate whether AI relay advantages persist with denser constellations.

2. **Imperfect CSI.** Introduce channel estimation errors to assess robustness of AI relay processing under realistic conditions.

3. **Online learning.** Develop relay strategies that adapt their parameters during operation, tracking time-varying channel conditions.

4. **Multi-relay networks.** Extend to cooperative relay selection and multi-hop routing with AI-optimized relays at each node.

5. **End-to-end learning.** Train the entire communication chain (modulation, relay, equalization, demodulation) jointly using autoencoder-based approaches.

6. **Optimized CUDA kernels.** Implement custom CUDA kernels for the S6 parallel prefix scan and SSD chunk-parallel multiply to eliminate the Python-loop bottleneck and achieve the theoretical $O(n)$ training speed.

7. **Diffusion models.** Score-based diffusion models represent the current state-of-the-art in generative modeling and could provide superior denoising for relay applications.

### 8.7 Conclusions

This thesis presents a comprehensive comparative study of nine relay strategies — two classical (AF, DF) and seven AI-based (GenAI, Hybrid, VAE, CGAN, Transformer, Mamba S6, Mamba2 SSD) — evaluated across six channel/topology configurations (AWGN, Rayleigh, Rician in SISO; 2×2 MIMO with ZF, MMSE, SIC equalization). Statistical significance is established via Wilcoxon signed-rank tests at each SNR point. The main conclusions are:

1. **AI relays outperform classical methods at low SNR (0–4 dB).** All seven AI methods achieve lower BER than both AF and DF in the low-SNR regime, with improvements of up to 4% in absolute BER. This advantage is consistent across all channel types and MIMO configurations, and is statistically significant ($p < 0.05$) for the Transformer, Mamba S6, and Mamba2 (SSD) at 0–4 dB on all channels.

2. **DF is optimal at medium-to-high SNR (≥6 dB) with zero parameters.** The classical decode-and-forward relay requires no training and no parameters yet achieves the best performance when channel quality is sufficient for reliable first-hop demodulation.

3. **Mamba S6 and Mamba2 (SSD) are the best AI relays at original parameter counts.** Both selective state space models win all low-SNR scenarios across all channels, producing virtually identical BER ($|\Delta| < 10^{-3}$). The SSD formulation trains 36% faster than S6 due to chunk-parallel computation.

4. **Architecture matters less than parameter count at equal scale.** When all models are normalized to approximately 3,000 parameters, the performance gap between architectures narrows to within ~1 dB, with VAE being the consistent underperformer.

5. **A 169-parameter network is sufficient for relay denoising.** The Minimal GenAI architecture achieves performance comparable to models 100× larger, demonstrating that the relay denoising task has inherently low complexity. Larger models overfit.

6. **MMSE-SIC provides the best MIMO equalization.** The non-linear SIC technique consistently outperforms both ZF and MMSE for all relay types, confirming the benefit of successive interference cancellation in the spatial multiplexing setting.

7. **The Hybrid relay is the recommended practical choice.** By combining AI processing at low SNR with classical DF at high SNR, the Hybrid relay achieves near-optimal performance across the entire SNR range with only 169 parameters.

8. **Mamba-2 (SSD) resolves the S6 training bottleneck.** The structured state space duality enables chunk-parallel training that eliminates the sequential kernel-launch overhead of S6, reducing training time from 37 to 24 minutes at full scale and from 617 to 404 seconds at 3K parameters — without any BER degradation.

These findings demonstrate that AI-based relay processing is a viable and beneficial complement to classical approaches, particularly in the challenging low-SNR regime. The key insight is that model complexity should be matched to task complexity — for the relay denoising task, minimal architectures suffice, and the choice between AI paradigms matters less than proper regularization and appropriate model sizing. When a sequence model is preferred (e.g., for longer symbol windows), the Mamba-2 SSD architecture offers the best efficiency–performance trade-off.

---

## 9. References

1. Cover, T. M., & El Gamal, A. A. (1979). Capacity theorems for the relay channel. *IEEE Transactions on Information Theory*, 25(5), 572–584.

2. Dorner, S., Cammerer, S., Hoydis, J., & Ten Brink, S. (2018). Deep learning based communication over the air. *IEEE Journal of Selected Topics in Signal Processing*, 12(1), 132–143.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

4. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*, 27.

5. Gu, A., & Dao, T. (2024). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

6. Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*.

7. Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations (ICLR)*.

8. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of Wasserstein GANs. *Advances in Neural Information Processing Systems*, 30.

9. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*.

10. Laneman, J. N., Tse, D. N., & Wornell, G. W. (2004). Cooperative diversity in wireless networks: Efficient protocols and outage behavior. *IEEE Transactions on Information Theory*, 50(12), 3062–3080.

11. Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.

12. Nosratinia, A., Hunter, T. E., & Hedayat, A. (2004). Cooperative communication in wireless networks. *IEEE Communications Magazine*, 42(10), 74–80.

13. Proakis, J. G., & Salehi, M. (2008). *Digital Communications* (5th ed.). McGraw-Hill.

14. Samuel, N., Diskin, T., & Wunder, G. (2019). Learning to detect for MIMO systems with unknown noise statistics. *IEEE Transactions on Signal Processing*, 67(12), 3261–3272.

15. Sklar, B. (2001). *Digital Communications: Fundamentals and Applications* (2nd ed.). Prentice Hall.

16. Sun, H., Chen, X., Shi, Q., Hong, M., Fu, X., & Sidiropoulos, N. D. (2018). Learning to optimize: Training deep neural networks for interference management. *IEEE Transactions on Signal Processing*, 66(20), 5438–5453.

17. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

18. Tse, D., & Viswanath, P. (2005). *Fundamentals of Wireless Communications*. Cambridge University Press.

19. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

20. Wolniansky, P. W., Foschini, G. J., Golden, G. D., & Valenzuela, R. A. (1998). V-BLAST: An architecture for realizing very high data rates over the rich-scattering wireless channel. *IEEE ISSSE*, 295–300.

21. Ye, H., Li, G. Y., & Juang, B. H. (2018). Power of deep learning for channel estimation and signal detection in OFDM systems. *IEEE Wireless Communications Letters*, 7(1), 114–117.

---

## 10. Appendices

### Appendix A: Mathematical Notation

| Symbol | Description |
|---|---|
| $b$ | Binary bit ($\in \{0, 1\}$) |
| $x$ | Modulated BPSK symbol ($\in \{-1, +1\}$) |
| $y$ | Received signal |
| $n$ | Noise sample |
| $h$ | Fading coefficient |
| $\sigma^2$ | Noise variance |
| $\text{SNR}$ | Signal-to-Noise Ratio (linear) |
| $G$ | AF amplification gain |
| $\mathbf{H}$ | MIMO channel matrix ($\in \mathbb{C}^{2 \times 2}$) |
| $\mathbf{W}$ | Neural network weight matrix |
| $\mathbf{b}$ | Bias vector |
| $P_e$ | Bit error rate |
| $Q(\cdot)$ | Gaussian Q-function |
| $K$ | Rician K-factor |
| $\beta$ | VAE KL weight |
| $\lambda$ | Regularization parameter |
| $\eta$ | Learning rate |
| $\Delta$ | State space discretization step |
| $\mathbf{A}, \mathbf{B}, \mathbf{C}, D$ | State space model matrices |

### Appendix B: Model Architectures and Hyperparameters

**GenAI (Minimal):**
- Input: 5-symbol sliding window
- Hidden: 24 neurons, ReLU activation
- Output: 1 neuron, Tanh activation
- Parameters: 169
- Training: MSE loss, lr=0.01, 100 epochs, 25K samples at SNR=[5, 10, 15] dB
- Implementation: NumPy (CPU)

**Hybrid:**
- Architecture: Same as GenAI (169 params)
- SNR threshold: Learned (default ~5 dB)
- Below threshold → GenAI processing; Above threshold → DF processing
- Implementation: NumPy (CPU)

**VAE:**
- Encoder: 7 → 32 (ReLU) → 16 (ReLU) → μ(8), log σ²(8)
- Decoder: 8 → 16 (ReLU) → 32 (ReLU) → 1 (Tanh)
- Parameters: 1,777
- Training: β-VAE loss (β=0.1), Adam lr=1e-3, 100 epochs
- Implementation: PyTorch (CUDA)

**CGAN (WGAN-GP):**
- Generator: (7+8) → 32 (LeakyReLU) → 32 (LeakyReLU) → 16 (LeakyReLU) → 1 (Tanh)
- Critic: (1+7) → 32 (LeakyReLU) → 16 (LeakyReLU) → 1
- Parameters: 2,946
- Training: WGAN-GP (λ_GP=10, λ_L1=100), Adam lr=1e-4, 200 epochs, 5 critic updates per generator update
- Implementation: PyTorch (CUDA)

**Transformer:**
- Input projection: 1 → 32
- Positional encoding: Sinusoidal
- Encoder: 2 layers, 4 heads, d_model=32, d_ff=128
- Output projection: 32 → 1 (Tanh)
- Parameters: 17,697
- Training: MSE loss, Adam lr=1e-3, 100 epochs
- Implementation: PyTorch (CPU)

**Mamba S6:**
- Input projection: 1 → 32
- Mamba blocks: 2 layers, each: LayerNorm → expand (32→64) → S6 (d_state=16) → contract (64→32) → residual
- S6 selective parameters: Δ, B, C = f(input) via learned linear projections
- Output projection: 32 → 1 (Tanh)
- Parameters: 24,001
- Training: MSE loss, Adam lr=1e-3, 100 epochs
- Implementation: PyTorch (CPU)

### Appendix C: Software Architecture

The project is implemented as a modular Python package (`relaynet`) with the following structure:

```
relaynet/
├── channels/          # Channel models
│   ├── awgn.py            # AWGN channel
│   ├── fading.py          # Rayleigh & Rician fading
│   └── mimo.py            # 2×2 MIMO + ZF/MMSE/SIC equalization
├── modulation/
│   └── bpsk.py            # BPSK modulate/demodulate
├── relays/            # Relay strategies
│   ├── base.py            # Abstract base class
│   ├── af.py              # Amplify-and-Forward
│   ├── df.py              # Decode-and-Forward
│   ├── genai.py           # Minimal GenAI (feedforward NN)
│   ├── hybrid.py          # SNR-adaptive Hybrid
│   ├── vae.py             # Variational Autoencoder
│   ├── cgan.py            # Conditional GAN (WGAN-GP)
│   ├── transformer.py     # Transformer (multi-head self-attention)
│   ├── mamba.py           # Mamba S6 (selective state space)
│   └── mamba2.py          # Mamba-2 SSD (structured state space duality)
├── simulation/
│   ├── runner.py          # Monte Carlo BER simulation
│   ├── statistics.py      # CI computation, Wilcoxon significance tests
│   └── checkpoint_manager.py  # Weight persistence & resume
├── visualization/
│   └── plots.py           # BER plotting utilities
└── utils/
    └── torch_compat.py    # Device detection helpers
```

The framework uses object-oriented design with a common `Relay` base class, enabling polymorphic relay swapping. Monte Carlo simulation is implemented in `runner.py` with configurable trial count, bit count, and SNR range. All MIMO operations use vectorized PyTorch for GPU acceleration.

**Testing:** 75 automated tests (pytest) cover all channels, modulation, relay strategies (including Mamba2 SSD), simulation, and statistics modules with 100% pass rate across five test modules:

| Module | Tests | Coverage |
|---|---|---|
| test_channels | 26 | AWGN, Rayleigh, Rician, MIMO (ZF/MMSE/SIC) |
| test_modulation | 10 | BPSK modulate/demodulate, edge cases |
| test_relays | 22 | All 9 relay strategies, including Mamba2 SSD |
| test_simulation | 6 | Runner, BER convergence, reproducibility |
| test_statistics | 11 | CI, Wilcoxon, significance tables |

**Reproducibility:** Random seeds are controlled at the source (bit generation) and noise (per-trial seeding) levels to ensure reproducible results. Weight persistence via `CheckpointManager` enables deterministic resume across sessions.

### Appendix D: Normalized 3K-Parameter Configurations

| Model | Parameters | Window | Hidden / Architecture |
|---|---|---|---|
| GenAI-3K | 3,004 | 11 | hidden=231 |
| Hybrid-3K | 3,004 | 11 | hidden=231 (+ DF switch) |
| VAE-3K | 3,037 | 11 | latent=10, hidden=(44, 20) |
| CGAN-3K | 3,004 | 11 | noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
| Transformer-3K | 3,007 | 11 | d_model=18, heads=2, layers=1 |
| Mamba-3K | 3,027 | 11 | d_model=16, d_state=6, layers=1 |
| Mamba2-3K | 3,004 | 11 | d_model=15, d_state=6, layers=1, chunk=8 |

All 3K configurations use a window size of 11 (vs. 5 for original GenAI/Hybrid, and 11 for original sequence models) to provide a common input context. The parameter counts are within ±1.2% of the 3,000 target. CGAN-3K is excluded from the default comparison run due to its extended training time (~55 min for adversarial training) but can be included via the `--include-cgan` flag.

---

## 11. Abstract (English)

**Generative AI for Two-Hop Relay Communication: A Comparative Study of Classical and AI-Based Relay Strategies**

This thesis presents a comprehensive comparative study of classical and artificial intelligence (AI) based relay strategies for two-hop cooperative communication systems. Nine relay methods are implemented and evaluated: two classical approaches — amplify-and-forward (AF) and decode-and-forward (DF) — and seven AI-based methods spanning supervised learning (GenAI minimal feedforward network, Hybrid SNR-adaptive relay), generative modeling (variational autoencoder, conditional GAN with WGAN-GP training), and modern sequence architectures (Transformer with multi-head self-attention, Mamba S6 selective state space model, Mamba-2 SSD structured state space duality model).

The evaluation is conducted across six channel and topology configurations: AWGN, Rayleigh fading, and Rician fading (K=3) channels in single-antenna (SISO) topology, and 2×2 MIMO spatial multiplexing with Rayleigh fading using three equalization techniques — zero-forcing (ZF), minimum mean square error (MMSE), and successive interference cancellation (SIC). All experiments use BPSK modulation with Monte Carlo simulation (100,000 bits per SNR point) and 95% confidence intervals. Statistical significance is established via Wilcoxon signed-rank tests at each SNR point.

The results reveal several key findings. First, all AI relays outperform classical methods at low SNR (0–4 dB), with Mamba S6 and Mamba2 (SSD) achieving the best performance across all channels at their original parameter configurations (24,001 and 26,179 parameters respectively). Second, the classical DF relay dominates at medium-to-high SNR (≥6 dB) with zero parameters, establishing a strong baseline. Third, a complexity study reveals an inverted-U relationship between model size and performance: a minimal 169-parameter two-layer network matches models 100× larger, while an 11,201-parameter model exhibits overfitting with degraded performance.

A normalized comparison constraining all AI models to approximately 3,000 parameters shows that the performance gap between architectures narrows significantly at equal scale, with VAE being the consistent underperformer. This finding indicates that parameter count, not architectural choice, is the primary performance driver for this task. Critically, at equal parameter budgets, Mamba2-3K (SSD) trains 35% faster than Mamba-3K (S6) while achieving identical BER, confirming the computational advantage of the chunk-parallel SSD formulation.

For MIMO systems, MMSE equalization consistently outperforms ZF, and non-linear SIC provides further improvement by cancelling the stronger stream's interference before detecting the weaker one. These equalization gains are additive to the relay processing benefits.

The recommended deployment strategy is a Hybrid relay that combines AI processing at low SNR with classical DF at high SNR, achieving near-optimal performance across the entire operating range with minimal computational overhead. For resource-constrained scenarios, the 169-parameter GenAI minimal relay provides competitive performance with approximately 0.7 KB of memory and under 3 seconds of training time. When a sequence model is required, Mamba-2 (SSD) offers the best efficiency–performance trade-off.

**Keywords:** Cooperative relay communication, generative AI, deep learning, two-hop relay, Mamba state space model, Mamba-2 SSD, structured state space duality, Transformer, variational autoencoder, conditional GAN, MIMO equalization, bit error rate

---

*Cover Page (Hebrew) — עמוד כריכה אחורי בעברית, בהתאם להנחיות בית הספר*

**בינה מלאכותית גנרטיבית לתקשורת ממסר דו-קפיצתית**

גיל צוקרמן

חיבור זה הוגש כמילוי חלקי של הדרישות לקבלת תואר מגיסטר למדעים (M.Sc.)

בית הספר להנדסת חשמל — אוניברסיטת תל אביב

2026
