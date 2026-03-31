import re

with open("thesis.md", "r", encoding="utf-8") as f:
    text = f.read()

def extract(pattern):
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""

front_matter = extract(r"(.*?)(?=## 1\. \[Introduction and Literature Review\])")
toc = extract(r"(## Table of Contents.*?)## Introduction and Literature Review")
intro = extract(r"## Introduction and Literature Review(.*?)(?=## Research Objectives)")
objectives = extract(r"## Research Objectives(.*?)(?=## Methods)")
methods = extract(r"## Methods(.*?)(?=## Results)")
results = extract(r"## Results(.*?)(?=## Discussion and Conclusions)")
discussion = extract(r"## Discussion and Conclusions(.*?)(?=## References)")
back_matter = extract(r"(## References.*)")

def extract_from(block, pattern):
    match = re.search(pattern, block, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""

methods_sys_model = extract_from(methods, r"(### System Model.*?)(?=### Channel Models)")
methods_channels = extract_from(methods, r"(### Channel Models.*?)(?=### Relay Strategies)")
methods_relays = extract_from(methods, r"(### Relay Strategies.*?)(?=### MIMO Equalization Techniques)")
methods_mimo = extract_from(methods, r"(### MIMO Equalization Techniques.*?)(?=### Simulation Framework)")
methods_sim = extract_from(methods, r"(### Simulation Framework.*?)(?=### Normalized Parameter Comparison)")
methods_norm = extract_from(methods, r"(### Normalized Parameter Comparison.*?)(?=### Modulation Schemes)")
methods_mod = extract_from(methods, r"(### Modulation Schemes.*)")

# Move theory to Introduction
new_intro = intro + "\n\n### Theoretical Foundations: Channel Models\n\n" + methods_channels
new_intro += "\n\n### Theoretical Foundations: MIMO Equalization\n\n" + methods_mimo

new_methods = methods_sys_model + "\n\n" + methods_relays + "\n\n" + methods_sim + "\n\n" + methods_norm + "\n\n" + methods_mod

new_results = """
The experiments chapter is structured to walk through the goals, trials, and conclusions from each experiment, following a systematic evaluation of relay strategies across diverse configurations.

### 4.1 Channel Model Validation
**Goal:** Validate the simulation framework against closed-form theoretical BER expressions to ensure baseline accuracy before evaluating AI relays.

**Trials:**
- **Topology:** SISO, MIMO 2x2
- **Modulation scheme:** BPSK
- **Channel:** AWGN, Rayleigh, Rician (K=3)
- **Equalizers (in MIMO only):** None (SISO), ZF, MMSE, SIC
- **Demod:** Hard decision only

**Conclusion:** 
Monte Carlo simulations match theoretical predictions within 95% confidence intervals across all channel models and topologies. AWGN follows the expected exponential decay, Rayleigh validates the $1/(4\\bar{\\gamma})$ high-SNR slope, Rician falls between the two, and MIMO equalization correctly exhibits the expected ZF < MMSE < SIC performance hierarchy.

### 4.2 SISO BPSK Performance (Baseline Relay Comparison)
**Goal:** Evaluate baseline classical and AI-based relay strategies on single-antenna configurations across different fading environments.

**Trials:**
- **Topology:** SISO
- **Modulation scheme:** BPSK
- **Channel:** AWGN, Rayleigh, Rician (K=3)
- **Equalizers:** None
- **NN architecture:** Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)
- **NN activation:** tanh

**Conclusion:** 
AI relays selectively outperform AF and, on selected channels (AWGN, Rician), DF at low SNR (0–4 dB). However, under Rayleigh fading, classical DF dominates even at low SNR. Across all channels, classical DF remains dominant at medium-to-high SNR ($\\geq 6$ dB), matching or exceeding all AI methods with zero parameters.

### 4.3 MIMO 2x2 BPSK Performance
**Goal:** Evaluate relay strategies under spatial multiplexing and various interference cancellation techniques.

**Trials:**
- **Topology:** MIMO 2x2
- **Modulation scheme:** BPSK
- **Channel:** Rayleigh
- **Equalizers (in MIMO only):** ZF, MMSE, SIC
- **NN architecture:** Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)
- **NN activation:** tanh

**Conclusion:** 
The MIMO equalization hierarchy (ZF < MMSE < SIC) holds for all relay types. The AI advantage at low SNR is preserved under ZF and MMSE (where Mamba S6 achieves the lowest BER), but under the superior SIC equalization, classical DF provides the lowest BER at all low-to-medium SNR points. Relay processing gains and MIMO equalization gains are additive.

### 4.4 Parameter Normalization & Complexity Trade-off
**Goal:** Isolate architectural inductive biases from parameter count effects and characterize the complexity-performance trade-off for relay denoising.

**Trials:**
- **Topology:** SISO, MIMO 2x2
- **Modulation scheme:** BPSK
- **Channel:** AWGN, Rayleigh, Rician, MIMO (ZF, MMSE, SIC)
- **Equalizers (in MIMO only):** None, ZF, MMSE, SIC
- **NN architecture:** All normalized to ~3,000 parameters, plus original sizes (169 to 26K)
- **NN activation:** tanh

**Conclusion:** 
The relay denoising task exhibits an inverted-U complexity relationship: a minimal 169-parameter MLP matches models 140x larger, while excessive parameters (11K+) lead to overfitting. At a normalized scale of 3,000 parameters, the performance gap between feedforward and sequence architectures narrows to ~1% BER, indicating that parameter count rather than architectural choice is the primary performance driver. Generative VAE is a consistent underperformer due to probabilistic overhead.

### 4.5 Higher-Order Modulation Scalability (Constellation-Aware Training)
**Goal:** Evaluate the generalizability of BPSK-trained relays to complex constellations and resolve the multi-level amplitude bottleneck.

**Trials:**
- **Topology:** SISO
- **Modulation scheme:** QPSK, 16-QAM
- **Channel:** AWGN, Rayleigh
- **Equalizers:** None
- **NN architecture:** Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)
- **NN activation:** tanh, linear, clipped tanh (hardtanh), scaled tanh, scaled sigmoid
- **Special case:** Constellation Aware training

**Conclusion:** 
QPSK performance mirrors BPSK perfectly due to I/Q independence. On 16-QAM, standard tanh compression causes a severe, irreducible BER floor (~0.22 at 16 dB). Replacing tanh with constellation-aware bounded activations (hardtanh, scaled tanh) bounded to the precise signal amplitude ($3/\\sqrt{10}$) and retraining eliminates this floor. Sequence models benefit most, reducing their BER floor by 5x, though a gap to classical DF persists due to per-axis error accumulation.

### 4.6 Input Normalization and CSI Injection
**Goal:** Determine the impact of structural input normalization and explicit channel state information (CSI) injection on higher-order modulations in fading channels.

**Trials:**
- **Topology:** SISO
- **Modulation scheme:** 16-QAM, 16-PSK
- **Channel:** Rayleigh
- **Equalizers:** None
- **NN architecture:** Transformer, Mamba S6, Mamba-2 SSD
- **NN activation:** tanh, hardtanh, scaled tanh, sigmoid
- **Special case:** CSI Injection, LayerNorm

**Conclusion:** 
Input LayerNorm universally benefits multi-level constellations like 16-QAM. Explicit CSI injection is highly modulation-dependent: it degrades performance for amplitude-carrying 16-QAM (creating redundant feature confusion) but significantly improves performance for constant-envelope 16-PSK, bringing the best neural models to within 2.5% of AF at high SNR. Across the 48 tested combinatorial variants, Mamba S6 proved the strongest architecture, but no neural relay surpassed classical DF.

### 4.7 16-Class 2D Classification for QAM16
**Goal:** Eliminate the structural BER floor imposed by per-axis I/Q splitting for 16-QAM by utilizing full 2D decision boundaries.

**Trials:**
- **Topology:** SISO
- **Modulation scheme:** 16-QAM
- **Channel:** AWGN
- **Equalizers:** None
- **NN architecture:** Supervised (MLP), Generative (VAE), Sequence (Transformer, Mamba S6, Mamba-2 SSD)
- **NN activation:** None (Softmax implicitly via Cross-Entropy loss)
- **Special case:** 16-class joint 2D classification

**Conclusion:** 
Treating the relay as a joint 16-point classifier over the full 2D constellation space completely eliminates the structural 4-class BER floor. For the first time, neural variants (VAE, Transformer, MLP) matched classical DF performance at high SNR, achieving near-zero BER at 20 dB. This proves the previous BER floor was an artifact of I/Q splitting, not a fundamental limitation of neural relays.

### 4.8 End-to-End Joint Optimization
**Goal:** Compare the modular neural relay approach against a fully joint transmitter-receiver autoencoder.

**Trials:**
- **Topology:** SISO
- **Modulation scheme:** Learned latent space (M=16, power-constrained)
- **Channel:** Rayleigh
- **Equalizers:** ZF explicitly at the receiver
- **NN architecture:** MLP Autoencoder (Encoder/Decoder)
- **Special case:** End-to-End (E2E) optimization

**Conclusion:** 
The E2E autoencoder underperforms both the theoretical limits of classical 16-QAM and the modular two-hop DF relay across all SNR points (67-141% higher BER). The network fails to discover a constellation geometry that surpasses the classical square grid under single-antenna Rayleigh fading, demonstrating that 'black-box' deep learning is inefficient compared to modular designs that leverage classical signal processing algorithms for modulation and equalization.
"""

new_text = text.replace(results, new_results)
new_text = new_text.replace(intro, new_intro)
new_text = new_text.replace(methods, new_methods)

toc_new = '''## Table of Contents {.unnumbered}

**Front Matter**

| | |
|---|---|
| I | List of Abbreviations |
| II | Abstract (Hebrew) |
| III | Keywords |

**Body**

1. [Introduction and Literature Review](#introduction-and-literature-review)
   - 1.1 Cooperative Relay Communication
   - 1.2 Classical Relay Strategies
   - 1.3 Machine Learning in Wireless Communication
   - 1.4 Generative Models for Signal Processing
   - 1.5 Sequence Models: Transformers and State Space Models
   - 1.6 MIMO Systems and Equalization
   - 1.7 Research Gap and Motivation
   - 1.8 Theoretical Foundations: Channel Models
   - 1.9 Theoretical Foundations: MIMO Equalization
2. [Research Objectives](#research-objectives)
   - 2.1 Main Objective
   - 2.2 Research Hypotheses
   - 2.3 Specific Objectives
   - 2.4 Scope and Delimitations
3. [Methods](#methods)
   - 3.1 System Model
   - 3.2 Relay Strategies
   - 3.3 Simulation Framework
   - 3.4 Normalized Parameter Comparison
   - 3.5 Modulation Schemes
4. [Experiments](#experiments)
   - 4.1 Channel Model Validation
   - 4.2 SISO BPSK Performance (Baseline Relay Comparison)
   - 4.3 MIMO 2x2 BPSK Performance
   - 4.4 Parameter Normalization & Complexity Trade-off
   - 4.5 Higher-Order Modulation Scalability (Constellation-Aware Training)
   - 4.6 Input Normalization and CSI Injection
   - 4.7 16-Class 2D Classification for QAM16
   - 4.8 End-to-End Joint Optimization
5. [Discussion and Conclusions](#discussion-and-conclusions)
   - 5.1 Interpretation of Results
   - 5.2 The Less is More Principle
   - 5.3 State Space vs. Attention for Signal Processing
   - 5.4 Practical Deployment Recommendations
   - 5.5 Limitations
   - 5.6 Future Work
   - 5.7 Conclusions
6. [References](#references)
7. [Appendices](#appendices)

**Back Matter**

| | |
|---|---|
| VIII | Abstract (English) |
'''

new_text = re.sub(r'## Results', '## Experiments', new_text)
new_text = re.sub(r'## Table of Contents \{\.unnumbered\}.*?## List of Figures', toc_new + '\n\n## List of Figures', new_text, flags=re.DOTALL)

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Successfully restructured and saved to thesis_restructured.md")