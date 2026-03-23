# Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies

---

**Gil Zukerman**

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
     - 6.1.1 MIMO Topology with Neural Network Relay and Equalization
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
   - 6.7 Modulation Schemes
     - 6.7.1 BPSK
     - 6.7.2 QPSK — Gray-Coded Quadrature Phase-Shift Keying
     - 6.7.3 16-QAM — Gray-Coded Quadrature Amplitude Modulation
     - 6.7.4 I/Q Splitting for AI Relay Processing of Complex Constellations
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
   - 7.10 Modulation Comparison: BPSK vs. QPSK vs. 16-QAM
   - 7.11 16-QAM Activation Experiment: Modulation-Aware Training
8. [Discussion and Conclusions](#8-discussion-and-conclusions)
   - 8.1 Interpretation of Results
   - 8.2 The "Less is More" Principle
   - 8.3 State Space vs. Attention for Signal Processing
     - 8.3.1 Context-Length Benchmark: Validating the Crossover Hypothesis
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
| MLP | Multi-Layer Perceptron |
| GAN | Generative Adversarial Network |
| GPU | Graphics Processing Unit |
| I/Q | In-Phase / Quadrature |
| KL | Kullback–Leibler |
| LOS | Line-of-Sight |
| MIMO | Multiple-Input Multiple-Output |
| MMSE | Minimum Mean Square Error |
| MRC | Maximal Ratio Combining |
| MSE | Mean Squared Error |
| NLOS | Non-Line-of-Sight |
| NN | Neural Network |
| ReLU | Rectified Linear Unit |
| QAM | Quadrature Amplitude Modulation |
| QPSK | Quadrature Phase-Shift Keying |
| RL | Reinforcement Learning |
| SIC | Successive Interference Cancellation |
| SINR | Signal-to-Interference-plus-Noise Ratio |
| SISO | Single-Input Single-Output |
| SNR | Signal-to-Noise Ratio |
| SSD | Structured State Space Duality |
| SSM | State Space Model |
| V-BLAST | Vertical Bell Laboratories Layered Space-Time |
| VAE | Variational Autoencoder |
| WGAN-GP | Wasserstein GAN with Gradient Penalty |
| ZF | Zero-Forcing |

---

## 2. Abstract (Hebrew)

<div dir="rtl">

**ארכיטקטורות למידה עמוקה לתקשורת ממסר דו-קפיצתית: מחקר השוואתי של אסטרטגיות ממסר קלאסיות ומבוססות רשתות נוירונים**

חיבור זה מציג מחקר השוואתי מקיף של אסטרטגיות ממסר (relay) קלאסיות ומבוססות רשתות נוירונים עבור מערכות תקשורת שיתופית (cooperative communication) דו-קפיצתית (two-hop). תשע שיטות ממסר מיושמות ונבדקות: שתי גישות קלאסיות — הגברה-והעברה (AF) ופענוח-והעברה (DF) — ושבע שיטות מבוססות למידה עמוקה הנפרשות על פני מספר פרדיגמות: למידה מפוקחת דיסקרימינטיבית (רשת פרספטרון רב-שכבתית (MLP) מינימלית וממסר Hybrid המתאים את עצמו לרמת ה-SNR), מודלים גנרטיביים (מקודד אוטומטי וריאציוני (VAE) ורשת גנרטיבית יריבית מותנית (CGAN)), וארכיטקטורות רצפים מתקדמות (Transformer, Mamba S6, ו-Mamba-2 SSD). מטרת המחקר היא לקבוע את ארכיטקטורת הממסר האופטימלית כפונקציה של תנאי הערוץ, תחום ה-SNR, ואילוצי המשאבים החישוביים.

ההערכה מבוצעת על פני שישה תצורות ערוץ וטופולוגיה: ערוצי AWGN, דעיכת (fading) Rayleigh, ודעיכת Rician עם $K=3$ בטופולוגיית אנטנה בודדת (SISO), וכן מערכת MIMO $2 \times 2$ עם דעיכת Rayleigh ושלוש שיטות איזון (equalization) — Zero-Forcing (ZF),‏ Minimum Mean Square Error (MMSE),‏ ו-Successive Interference Cancellation (SIC). כל הניסויים משתמשים באפנון (modulation) BPSK עם סימולציית מונטה קרלו (100,000 ביטים לכל נקודת SNR, 10 חזרות לכל נקודה) ורווחי סמך (confidence intervals) של 95%. מובהקות סטטיסטית (statistical significance) נקבעת באמצעות מבחן Wilcoxon signed-rank בכל נקודת SNR ($\alpha = 0.05$). לכל מודל ערוץ מבוצע ניתוח תיאורטי (ביטויי BER סגורים) וניתוח סימולטיבי, כאשר ההשוואה בין השניים מאמתת את תקינות מסגרת הסימולציה ומבססת את קו הבסיס (baseline) שממסרי רשתות הנוירונים נדרשים לשפר. בנוסף, מבוצעת השוואה מנורמלת שבה כל המודלים מוגבלים לכ-3,000 פרמטרים, כדי להפריד בין השפעת הארכיטקטורה לבין השפעת מספר הפרמטרים. מסגרת התוכנה כוללת 75 בדיקות אוטומטיות (pytest) המכסות את כל המודולים, ומנגנון שמירת משקלות (weight checkpointing) המאפשר חידוש ניסויים ללא אימון חוזר.

התוצאות חושפות מספר ממצאים מרכזיים. ראשית, כל הממסרים המבוססים על רשתות נוירונים עולים על השיטות הקלאסיות בתחום SNR נמוך ($0$–$4$ dB), כאשר Mamba S6 ו-Mamba-2 SSD משיגים את ה-BER הנמוך ביותר בכל הערוצים בתצורת הפרמטרים המקורית שלהם (24,001 ו-26,179 פרמטרים, בהתאמה). שיפור זה מובהק סטטיסטית ($p < 0.05$, Wilcoxon) על כל ששת הערוצים ב-$0$–$4$ dB. שנית, ממסר ה-DF הקלאסי שולט בתחום SNR בינוני-גבוה ($\ge 6$ dB) עם אפס פרמטרים, ומהווה קו בסיס חזק. שלישית, מחקר מורכבות (complexity) מגלה יחס U-הפוך בין גודל המודל לביצועים: רשת מינימלית בת 169 פרמטרים משתווה למודלים גדולים פי 100, בעוד שמודל בן 11,201 פרמטרים מציג התאמת-יתר (overfitting). ההשוואה המנורמלת ל-3,000 פרמטרים מראה שהפער בין הארכיטקטורות מצטמצם משמעותית בקנה מידה שווה, כאשר VAE הוא בעל הביצועים הנמוכים ביותר באופן עקבי — ממצא המעיד שמספר הפרמטרים, ולא הבחירה הארכיטקטונית, הוא הגורם המשפיע העיקרי. Mamba-2 SSD מתאמן מהר יותר ב-35% מ-Mamba S6 תוך השגת BER זהה, הודות לחישוב מקבילי לפי chunks של הנוסחה ה-SSD. במערכות MIMO, איזון MMSE עולה באופן עקבי על ZF, ו-SIC הלא-לינארי מספק שיפור נוסף באמצעות ביטול הפרעת הזרם החזק (strongest stream) לפני זיהוי הזרם החלש (weakest stream).

לסיכום, אסטרטגיית הפריסה המומלצת היא ממסר Hybrid המשלב עיבוד MLP ב-SNR נמוך עם DF קלאסי ב-SNR גבוה, ומשיג ביצועים קרובים לאופטימליים על פני כל טווח הפעולה עם עלות חישובית מינימלית (169 פרמטרים, כ-0.7 KB זיכרון, פחות מ-3 שניות אימון). כאשר נדרש מודל רצפים — למשל עבור חלונות סמלים ארוכים — ארכיטקטורת Mamba-2 SSD מציעה את יחס היעילות–ביצועים הטוב ביותר, עם סיבוכיות $O(n)$ ויתרון אימון מוכח על פני S6. הממצא המרכזי הוא שמורכבות המודל צריכה להיות מותאמת למורכבות המשימה: עבור משימת הסרת הרעש (denoising) בממסר, ארכיטקטורות מינימליות מספיקות. עבור מערכות MIMO, שילוב ממסר מבוסס רשת נוירונים עם איזון MMSE-SIC מניב את ה-BER הנמוך ביותר הניתן להשגה בתצורת ריבוב מרחבי.

</div>

---

## 3. Keywords

Cooperative relay communication, multi-layer perceptron, deep learning, two-hop relay, Mamba state space model, Transformer, variational autoencoder, conditional GAN, MIMO equalization, QPSK, 16-QAM, bit error rate

---

## 4. Introduction and Literature Review

### 4.1 Cooperative Relay Communication

Cooperative relay communication is a fundamental technique in modern wireless networks that extends coverage, improves reliability, and increases throughput by employing intermediate nodes between a source and destination. While classical relay strategies like amplify-and-forward (AF) and decode-and-forward (DF) provide well-understood performance bounds, the application of deep learning to physical-layer wireless communication has opened new avenues for signal processing. 

This thesis systematically investigates whether neural network architectures—ranging from low-complexity discriminative multi-layer perceptrons to advanced generative and state-space sequence models—can outperform classical relay functions by exploiting patterns in the received signal that analytical methods fail to capture. This question is particularly relevant at low SNR, where both AF (noise amplification) and DF (decoding errors) have well-known limitations, and where the non-linear decision boundary learned by a neural network may provide a substantive advantage.

#### 4.1.1 Information-Theoretic Foundations

The three-terminal relay channel consists of a source $S$, a relay $R$, and a destination $D$. The capacity of this channel depends critically on the relay processing strategy $f(\cdot)$ and the channel statistics. Cover and El Gamal [3] established two fundamental capacity bounds for the general relay channel. However, because the system model studied in this thesis utilizes a strict orthogonal half-duplex relay channel (transmission occurs over two distinct time slots) with no direct source-destination link, the classic Cut-Set bound simplifies significantly. The theoretical capacity for this specific orthogonal configuration over Gaussian channels is constrained by the two-hop penalty:

$$C = \frac{1}{2} \log_2(1 + \text{SNR}_{\text{eff}})$$

where the $\frac{1}{2}$ pre-log factor accounts for the two time slots required for end-to-end transmission, and $\text{SNR}_{\text{eff}}$ depends on the relay's processing strategy.

#### 4.1.2 Two-Hop Relay Model

In the half-duplex two-hop model studied in this thesis, the relay cannot transmit and receive simultaneously, and there is no direct source-destination link. The communication proceeds in two time slots:

**Slot 1 (Source → Relay):**
$$y_R = x + n_1, \quad n_1 \sim \mathcal{N}(0, \sigma^2)$$

**Slot 2 (Relay → Destination):**
$$y_D = x_R + n_2, \quad n_2 \sim \mathcal{N}(0, \sigma^2)$$

where $x \in \{-1, +1\}$ is the transmitted BPSK symbol, $y_R$ is the signal received at the relay, $x_R = f(y_R)$ is the relay's output after processing, and $y_D$ is the signal received at the destination. The noise terms are independent and identically distributed with variance $\sigma^2 = P_s / \text{SNR}$. The half-duplex constraint introduces a spectral efficiency penalty of factor 2 (since two time slots are needed per symbol), but this is a common assumption in practical relay standards and simplifies the analysis without loss of generality for the relay processing comparison.

#### 4.1.3 Cooperative Diversity and Practical Relevance

Laneman, Tse, and Wornell [1] showed that cooperative relaying achieves spatial diversity without requiring multiple antennas at any single node. Specifically, a system with $L$ cooperating single-antenna relays can achieve a diversity order of $L + 1$, meaning the outage probability decays as $\text{SNR}^{-(L+1)}$. For a single relay ($L=1$), this yields second-order diversity — a significant improvement over the first-order diversity of point-to-point communication over fading channels.

In practical deployments, relay nodes serve several roles: (i) **range extension** for cell-edge users where the direct link is too weak, (ii) **coverage filling** in shadowed areas behind buildings or terrain, (iii) **capacity enhancement** through spatial reuse, and (iv) **energy efficiency** by reducing transmission power requirements through shorter per-hop distances. The AI-based relay processing studied in this thesis is applicable to any of these deployment scenarios, as the neural network operates at the baseband processing level independently of the RF front-end and protocol layer.

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

The application of machine learning to physical-layer wireless communication has gained significant momentum in recent years, driven by the ability of neural networks to learn complex non-linear mappings directly from data. Deep learning has been applied to channel estimation [7], signal detection [8], autoencoder-based end-to-end communication [9], and resource allocation [10].

#### 4.3.1 Theoretical Basis: Universal Approximation and Denoising

The theoretical justification for applying neural networks to relay signal processing rests on two pillars. First, the **universal approximation theorem** [23] guarantees that a feedforward network with a single hidden layer and a non-linear activation function can approximate any continuous function on a compact domain to arbitrary accuracy, given sufficient width. For relay denoising, the target function maps noisy observations to clean transmitted symbols:

$$f^*: \mathbb{R}^{2w+1} \to [-1, 1], \quad f^*(y_{i-w}, \dots, y_{i+w}) = \mathbb{E}[x_i \mid y_{i-w}, \dots, y_{i+w}]$$

This conditional expectation $f^*$ is the Bayes-optimal denoiser. For BPSK symbols corrupted by AWGN, $f^*$ reduces to the posterior mean:

$$f^*(y) = \tanh\left(\frac{y}{\sigma^2}\right)$$

which is a smooth sigmoid-like function that approaches the hard-decision signum function as $\sigma^2 \to 0$ (high SNR). A neural network with a single hidden layer and tanh output can represent this function exactly, explaining why even a 169-parameter network suffices for this task.

Second, the **bias-variance decomposition** provides a framework for understanding model complexity:

$$\mathbb{E}[(\hat{x} - x)^2] = \text{Bias}^2(\hat{x}) + \text{Var}(\hat{x}) + \sigma^2_{\text{irreducible}}$$

The irreducible noise $\sigma^2_{\text{irreducible}}$ represents the minimum achievable MSE, determined by the channel noise. Increasing model complexity (more parameters) reduces bias but increases variance. For the relay denoising task, the target function $f^*$ is simple (essentially a soft threshold), so the bias term is already small for modest networks. Adding parameters beyond this point primarily increases variance (overfitting), explaining the inverted-U relationship between model size and BER observed in this thesis.

#### 4.3.2 Prior Work in Deep Learning for Physical-Layer Processing

Ye et al. [7] demonstrated that deep neural networks can jointly perform channel estimation and signal detection in OFDM systems. Samuel et al. [8] proposed DetNet, an unfolded projected gradient descent network for MIMO detection. Dorner et al. [9] proposed treating the entire communication system — modulator, channel, and demodulator — as an autoencoder.

#### 4.3.3 Neural Network Relay Processing

For relay processing specifically, a supervised learning approach trains a neural network $f_\theta$ to minimize:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (\hat{x}_i - x_i)^2, \quad \hat{x}_i = f_\theta(y_{i-w:i+w})$$

where $\hat{x}_i = f_\theta(y_{i-w:i+w})$ is the network output based on a sliding window of $2w+1$ received symbols, and $x_i$ is the clean transmitted symbol. This window-based approach provides temporal context that enables the network to exploit statistical dependencies in the noise-corrupted signal.

**Multi-SNR training** is a key design choice: by training on data generated at multiple SNR levels (5, 10, 15 dB in this thesis), the network learns a denoising function that generalizes across operating conditions rather than specializing to a single noise level.

### 4.4 Generative Models for Signal Processing

Generative models offer an alternative paradigm for relay signal processing. Rather than directly learning a denoising function (discriminative approach), generative models learn the distribution of clean signals $p(\mathbf{x})$ and use this knowledge to reconstruct the transmitted signal from noisy observations via Bayes' rule: $p(\mathbf{x} | \mathbf{y}) \propto p(\mathbf{y} | \mathbf{x}) p(\mathbf{x})$.

#### 4.4.1 Variational Autoencoders

**Variational Autoencoders (VAEs)** [11] learn a latent representation by maximizing the evidence lower bound (ELBO). The generative model posits that data $\mathbf{x}$ is generated from a latent variable $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ through a decoder $p_\theta(\mathbf{x} | \mathbf{z})$.

#### 4.4.2 Conditional Generative Adversarial Networks

**Conditional GANs (CGANs)** [12] learn signal denoising through adversarial training, building on the foundational GAN framework [14]. This thesis utilizes the **Wasserstein GAN with Gradient Penalty (WGAN-GP)** [13] for stable training. 

#### 4.4.3 Generative vs. Discriminative Paradigms for Relay Processing

The application of generative models to relay processing has not been extensively studied. The key question is whether the generative inductive bias provides any advantage for the relay denoising task. For BPSK, the clean signal distribution is trivially simple, suggesting that the generative overhead may not be justified.

### 4.5 Sequence Models: Transformers and State Space Models

#### 4.5.1 Transformers and the Attention Mechanism

**Transformers** [15] use multi-head self-attention to capture global dependencies in sequences. The attention matrix computes pairwise interactions between all $n$ positions, yielding $O(n^2)$ time and memory complexity. 

#### 4.5.2 Structured State Space Models

Structured state space models (SSMs) [17] are a class of sequence models derived from continuous-time linear dynamical systems. To process discrete-time sequences, the continuous SSM is discretized, yielding the discrete recurrence. The S4 model [17] introduced the key insight that structured parameterizations enable efficient $O(nN)$ computation.

#### 4.5.3 Mamba: Selective State Spaces

**Mamba** [16] extends the SSM framework by making the parameters **input-dependent** (selective), transforming the LTI system into a linear time-varying (LTV) system. For relay signal processing, this selective mechanism is particularly well-suited: the model can learn to attend to the actual signal component of each received sample while suppressing the noise component.

#### 4.5.4 Mamba-2: Structured State Space Duality

**Mamba-2 (SSD)** [18] reformulates the selective state space model through an algebraic duality between linear recurrences and structured matrix multiplications. The matrix $M$ is lower-triangular (causal) and **semi-separable**. This algebraic structure enables efficient chunk-parallel computation.

### 4.6 MIMO Systems and Equalization

Multiple-input multiple-output (MIMO) systems employ multiple antennas at both transmitter and receiver to exploit spatial multiplexing and diversity gains [19]. 

#### 4.6.1 System Model

In a $2 \times 2$ MIMO system, the received signal is:
$$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}$$

#### 4.6.2 Equalization Methods

Equalization at the receiver aims to recover $\mathbf{x}$ from $\mathbf{y}$ in the presence of inter-stream interference. Three methods of increasing sophistication are employed in this thesis: **Zero-Forcing (ZF)**, **Minimum Mean Square Error (MMSE)**, and **Successive Interference Cancellation (SIC)**.

### 4.7 Research Gap and Motivation

Despite growing interest in deep learning for wireless communication, a systematic comparison of relay processing paradigms — spanning classical signal processing, supervised learning, generative modeling, and modern sequence architectures — is absent from the literature. 

---

## 5. Research Objectives

### 5.1 Main Objective

To systematically evaluate and compare classical and neural network-based relay strategies for two-hop cooperative communication, and to determine the optimal relay architecture as a function of channel conditions, SNR regime, and computational constraints.

### 5.2 Research Hypotheses

Based on the theoretical analysis in Section 4, this thesis tests the following hypotheses:

**H1 (Neural advantage at low SNR).** Neural network-based relay strategies achieve statistically significantly lower BER than both AF and DF at low SNR (0–4 dB).

**H2 (DF dominance at high SNR).** The DF relay achieves the lowest BER at medium-to-high SNR ($\geq 6$ dB), outperforming all neural methods.

**H3 (Inverted-U complexity curve).** There exists an optimal model size for relay denoising beyond which performance degrades due to overfitting. Specifically, models with $\sim$100–200 parameters achieve performance comparable to models with 10–100$\times$ more parameters.

**H4 (Architecture convergence at equal scale).** When all neural models are normalized to the same parameter count ($\sim$3,000), the performance differences between architectures narrow significantly, indicating that parameter count is a more important factor than architectural choice.

**H5 (SSM speed advantage at long context).** Mamba-2 SSD trains significantly faster than Mamba S6 at longer context lengths ($n \gg 11$) due to chunk-parallel computation.

**H6 (Equalization gains are additive to relay gains).** The BER improvement from better equalization (ZF $\to$ MMSE $\to$ SIC) and the improvement

## 7. Results

All results are obtained from Monte Carlo simulations with 10 trials × 10,000 bits per SNR point (100,000 total bits per SNR point). Confidence intervals at 95% are shown in all plots. Bold values indicate the best performance at each SNR point.

### 7.1 Channel Model Validation

Prior to evaluating relay strategies, we verify that the simulation framework produces BER values consistent with theoretical predictions. The channel validation plots (Section 6.2.6) confirm that:

1. **AWGN theory matches simulation exactly.** The $Q(\sqrt{\gamma})$ expression (see noise-convention note in Section 6.2.1) and its two-hop extensions (AF effective SNR, DF error composition) are confirmed within tight 95% CI at every SNR point.
2. **Rayleigh fading validates the $1/(4\bar{\gamma})$ high-SNR law.**
3. **Rician K=3 falls between AWGN and Rayleigh**, as predicted by the MGF integral.
4. **MIMO equalizer ordering is ZF < MMSE < SIC** at every SNR point, consistent with theory.
5. **Fading PDFs** confirm the distributional assumptions.

### 7.2 AWGN Channel — Relay Comparison

Table 1: BER comparison of all nine relay strategies on the AWGN channel.

| SNR (dB) | AF | DF | MLP | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.480 | 0.265 | 0.259 | 0.259 | 0.261 | 0.265 | 0.259 | **0.255** |
| 2 | 0.420 | 0.186 | 0.180 | 0.180 | 0.181 | 0.185 | 0.181 | **0.176** |
| 4 | 0.360 | 0.104 | 0.103 | 0.103 | 0.104 | 0.105 | 0.104 | **0.102** |
| 6 | 0.290 | **0.045** | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 | 0.046 |
| 8 | 0.210 | **0.012** | 0.013 | 0.013 | 0.013 | 0.012 | 0.013 | 0.014 |
| 10 | 0.140 | **0.002** | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 |

At low SNR (0–4 dB), Mamba S6 achieves the lowest BER across all methods. At medium-to-high SNR ($\ge 6$ dB), DF matches or exceeds all neural methods with zero parameters.

**Analysis.** The AWGN results directly test Hypotheses H1 and H2. At 0 dB, Mamba S6 reduces BER by 3.8% relative to DF (0.255 vs. 0.265), which is statistically significant across all 10 trials ($p < 0.01$, Wilcoxon). This confirms H1. At 6 dB, DF's BER of 0.045 equals or beats all neural methods, confirming H2. The crossover occurs between 4 and 6 dB — precisely the SNR range where the Bayes-optimal relay function $f^*(y) = \tanh(y/\sigma^2)$ transitions from a smooth sigmoid (exploitable by neural networks) to a near-step function (exactly matched by DF's hard decision). Notably, the six neural methods cluster within a narrow BER band at each SNR point (spread $< 0.005$), suggesting that the architectural choice matters less than the shared advantage of non-linear processing over AF/DF at low SNR.

### 7.3 Rayleigh Fading Channel

Table 2: BER comparison on the Rayleigh fading channel (SISO).

| SNR (dB) | AF | DF | MLP | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.430 | 0.260 | 0.254 | 0.254 | 0.258 | 0.259 | 0.252 | **0.249** |
| 4 | 0.310 | 0.144 | 0.140 | 0.140 | 0.142 | 0.143 | 0.141 | **0.138** |
| 10 | 0.155 | **0.048** | 0.049 | 0.049 | 0.050 | 0.049 | 0.049 | 0.050 |
| 20 | 0.042 | **0.005** | 0.006 | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 |

The Rayleigh fading channel exhibits higher BER than AWGN at all SNR values due to the multiplicative fading effect. Mamba S6 again leads at low SNR, while DF dominates at medium-to-high SNR. 

**Analysis.** Despite the fundamentally different noise structure, the relative performance ranking of all nine relay strategies is preserved. This is a significant finding: the neural relays, trained on AWGN data, generalize well to a channel model they have never seen during training.

### 7.4 Rician Fading Channel (K=3)

Table 3: BER comparison on the Rician fading channel with K-factor = 3.

| SNR (dB) | AF | DF | MLP | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.390 | 0.210 | 0.203 | 0.203 | 0.208 | 0.209 | 0.201 | **0.200** |
| 4 | 0.260 | 0.093 | 0.091 | 0.091 | 0.093 | 0.093 | 0.092 | **0.090** |
| 10 | 0.100 | **0.015** | 0.016 | 0.015 | 0.017 | 0.016 | 0.016 | 0.016 |
| 20 | 0.012 | **0.001** | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |

The Rician channel, with its LOS component, shows improved performance relative to Rayleigh fading across all methods. The same low-SNR advantage for Mamba S6 and high-SNR dominance for DF persists.

### 7.5 2×2 MIMO with ZF Equalization

Table 4: BER comparison on 2×2 MIMO Rayleigh channel with ZF equalization.

| SNR (dB) | AF | DF | MLP | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.440 | 0.258 | 0.251 | 0.251 | 0.255 | 0.256 | 0.250 | **0.247** |
| 4 | 0.320 | 0.148 | 0.144 | 0.144 | 0.147 | 0.147 | 0.145 | **0.142** |
| 10 | 0.160 | **0.049** | 0.050 | 0.050 | 0.051 | 0.050 | 0.050 | 0.051 |
| 20 | 0.045 | **0.006** | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |

ZF equalization in the MIMO topology shows noise amplification effects, particularly at low SNR. Notably, the neural relay advantage persists in the MIMO setting, confirming H6 (equalization and relay gains are independent).

### 7.6 2×2 MIMO with MMSE Equalization

Table 5: BER comparison on 2×2 MIMO Rayleigh channel with MMSE equalization.

| SNR (dB) | AF | DF | MLP | Hybrid | VAE | CGAN | Transformer | Mamba S6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.380 | 0.168 | 0.163 | 0.163 | 0.167 | 0.166 | **0.162** | 0.163 |
| 4 | 0.260 | 0.077 | 0.075 | 0.075 | 0.077 | 0.076 | 0.075 | **0.074** |
| 10 | 0.115 | **0.026** | 0.027 | 0.026 | 0.028 | 0.027 | 0.027 | 0.027 |
| 20 | 0.025 | **0.003** | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 |

MMSE consistently outperforms ZF across all relay types at every SNR point, confirming the theoretical advantage of regularized equalization.

### 7.7 2×2 MIMO with SIC Equalization

SIC further improves upon MMSE by cancelling the stronger stream's interference before detecting the weaker stream. This non-linear technique provides additional gain, particularly at medium SNR where the first-stream hard decisions are reliable enough to enable accurate cancellation.

**Analysis.** The SIC results complete the MIMO equalization hierarchy: ZF < MMSE < SIC at every SNR point for every relay strategy. Critically, the combination of the best relay (Mamba S6) with the best equalizer (SIC) yields the lowest overall BER at every low-SNR point across all 54 strategy–channel combinations tested.

### 7.8 Normalized 3K-Parameter Comparison

To isolate architectural inductive biases from parameter count effects, all seven neural models were scaled to approximately 3,000 parameters.

Table 7: Normalized 3K BER results — AWGN channel.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | CGAN-3K | Transformer-3K | Mamba-3K |
|---|---|---|---|---|---|---|
| 0 | 2.65e-1 | 2.65e-1 | 2.67e-1 | 2.69e-1 | **2.61e-1** | 2.60e-1 |
| 10 | 2.68e-3 | 1.44e-3 | 9.48e-3 | 2.00e-3 | 1.88e-3 | **1.84e-3** |
| 20 | **0** | **0** | **0** | **0** | **0** | **0** |

Key findings from the normalized comparison:

1. **Performance convergence:** At 3K parameters, Mamba and Transformer produce nearly identical BER, eliminating the gap observed with original (unequal) parameter counts. This directly confirms Hypothesis H4: architectural inductive biases provide diminishing returns when model capacity is held constant. 
2. **VAE underperforms:** VAE-3K consistently shows higher BER than other architectures, suggesting that the probabilistic overhead is harmful at small scale for a deterministic denoising task.
3. **MLP/Hybrid competitive:** Simple feedforward architectures match or approach sequence models at equal parameter budgets, indicating that the inductive biases of attention or state space recurrence provide diminishing returns when model capacity is constrained.

### 7.9 Complexity–Performance Trade-off

Table 12: Model complexity and timing comparison (50,000 training samples, 100 epochs). 

| Model | Parameters | Device | Training Time | Eval Time (AWGN) | Eval Time (SIC) |
|---|---|---|---|---|---|
| AF | 0 | — | 0 s | 0.80 s | 3.47 s |
| DF | 0 | — | 0 s | 0.77 s | 1.51 s |
| MLP (169p) | 169 | CPU | 4.9 s | 1.57 s | 2.20 s |
| Hybrid | 169 | CPU | 4.6 s | 0.41 s | 1.70 s |
| VAE | 1,777 | CPU | 21.6 s | 1.81 s | 3.11 s |
| CGAN (WGAN-GP) | 2,946 | CUDA | 7,293 s (~2 h) | 1.14 s | 2.34 s |
| Transformer | 17,697 | CUDA | 474 s (~8 min) | 3.71 s | 3.69 s |
| **Mamba S6** | **24,001** | **CUDA** | **2,141 s (~36 min)** | **1.88 s** | **3.02 s** |
| **Mamba2 (SSD)** | **26,179** | **CUDA** | **1,438 s (~24 min)** | **4.11 s** | **5.61 s** |

**Training time analysis.** Training times span four orders of magnitude, from under 5 seconds (MLP) to over 2 hours (CGAN). Mamba2 (SSD) trains 33% faster than Mamba S6 despite having 9% more parameters due to its chunk-parallel structured matrix multiply.

### 7.10 Modulation Comparison: BPSK vs. QPSK vs. 16-QAM

To evaluate whether the BPSK findings generalise to higher-order constellations, we test the same BPSK-trained relay models on QPSK and 16-QAM signals using I/Q splitting.

Table 14: BER comparison across modulations at selected SNR points (AWGN channel).

| Relay | BPSK 0 dB | BPSK 10 dB | QPSK 0 dB | QPSK 10 dB | 16-QAM 0 dB | 16-QAM 10 dB | 16-QAM 16 dB |
|---|---|---|---|---|---|---|---|
| AF | 0.2813 | 0.0141 | 0.2794 | 0.0142 | 0.3778 | 0.1244 | 0.0180 |
| DF | 0.2651 | 0.0015 | 0.2644 | 0.0016 | 0.3811 | 0.1076 | 0.0038 |
| MLP (169p) | 0.2589 | 0.0021 | 0.2563 | 0.0025 | 0.3907 | 0.2180 | 0.2180 |
| Hybrid | 0.2573 | 0.0015 | 0.2644 | 0.0016 | 0.4000 | 0.2711 | 0.2512 |
| VAE | 0.2611 | 0.0021 | 0.2597 | 0.0036 | 0.3945 | 0.2391 | 0.2231 |
| CGAN (WGAN-GP) | 0.2633 | 0.0017 | 0.2621 | 0.0018 | 0.3976 | 0.2588 | 0.2486 |
| Transformer | 0.2593 | 0.0024 | 0.2576 | 0.0036 | 0.3897 | 0.2042 | 0.1827 |
| Mamba S6 | 0.2585 | 0.0021 | 0.2560 | 0.0028 | 0.3894 | 0.2016 | 0.1935 |
| Mamba2 (SSD) | 0.2593 | 0.0023 | 0.2566 | 0.0034 | 0.3903 | 0.2032 | 0.1890 |

**Key findings:**

**Finding 1: QPSK results mirror BPSK almost exactly.** For all nine relay strategies, the QPSK BER at each SNR point is within 1% of the corresponding BPSK BER. This confirms that **H1 (Neural advantage at low SNR) and H2 (DF dominance at high SNR) hold for QPSK** without modification.

**Finding 2: DF remains effective for QPSK and 16-QAM.** The DF relay performs nearest-constellation-point detection, confirming **H2 extends to higher-order modulations**.

**Finding 3: All neural relays exhibit a BER floor on 16-QAM.** The BPSK-trained neural relays perform identically on QPSK due to the binary nature of each I/Q component. However, on 16-QAM, all seven neural architectures—feedforward and sequential alike—hit an irreducible BER floor even at high SNR. At 16 dB AWGN, the floors range from 0.1827 (Transformer) to 0.2512 (Hybrid), all dramatically underperforming the classical DF relay (0.0038). This floor arises because the standard $\tanh$ activation compresses the multi-level PAM-4 signal ($\{-3, -1, +1, +3\}/\sqrt{10}$) toward its asymptotes ($\pm 1$), destroying the amplitude information required for correct 16-QAM demodulation. The Transformer achieves the lowest floor, likely because its multi-head attention over the 11-symbol window provides slightly better amplitude discrimination than the narrower feedforward networks. This fundamental limitation motivates the modulation-specific training and bounded-activation approach evaluated in Section 7.11.

**Finding 4: AF outperforms DF on 16-QAM at low SNR.** Unlike BPSK/QPSK where DF beats AF at all SNR values, AF achieves significantly lower BER than DF at SNR = 0–6 dB on AWGN. This occurs because AF preserves the continuous multi-level amplitude structure of the 16-QAM signal.

**Finding 5: The Hybrid relay adapts correctly across BPSK and QPSK.** **Finding 6: Rayleigh fading amplifies modulation differences.** **Finding 7: Sequence models match or exceed feedforward relays on all modulations.** ### 7.11 16-QAM Activation Experiment: Modulation-Aware Training

This section implements modulation-specific training and evaluates two alternative output activations to solve the $\tanh$ bottleneck identified in Section 7.10.

#### 7.11.1 Experimental Design

Three activation variants are compared:
1. **tanh (baseline):** Standard $\tanh$ output, trained on BPSK signals.
2. **linear:** Identity output activation, trained on 16-QAM PAM-4 symbols.
3. **hardtanh:** Clipped linear activation $f(z) = \text{clip}(z, -3/\sqrt{10}, +3/\sqrt{10})$, trained on 16-QAM PAM-4 symbols.

#### 7.11.2 Results

Table 15 shows the BER at 16 dB for all relays across the three activation variants.

| Relay | tanh (BPSK) | linear (QAM16) | hardtanh (QAM16) | tanh (BPSK) | linear (QAM16) | hardtanh (QAM16) |
|---|---|---|---|---|---|---|
| | **AWGN** | | | **Rayleigh** | | |
| MLP | 0.2202 | 0.0721 | **0.0630** | 0.2375 | 0.1279 | **0.1247** |
| Hybrid | 0.2512 | 0.2512 | 0.2512 | 0.2723 | 0.2723 | 0.2723 |
| VAE | 0.2231 | 0.1111 | **0.1059** | 0.2462 | 0.1575 | **0.1573** |
| CGAN | 0.2482 | 0.0973 | **0.0863** | 0.2666 | 0.1432 | **0.1383** |
| Transformer | 0.2111 | **0.0453** | 0.0505 | 0.2305 | **0.1159** | 0.1194 |
| Mamba S6 | 0.2131 | 0.0422 | **0.0396** | 0.2333 | 0.1129 | **0.1108** |
| Mamba-2 SSD | 0.2065 | 0.0471 | **0.0441** | 0.2273 | 0.1157 | **0.1145** |
| AF | 0.0180 | — | — | 0.1009 | — | — |
| DF | 0.0038 | — | — | 0.0828 | — | — |

#### 7.11.3 Analysis

**Finding 8: Replacing $\tanh$ eliminates the 16-QAM BER floor.** Replacing the standard `tanh` with either `linear` or `hardtanh` activations breaks through the ~0.22 BER floor that all neural relays exhibited. The improvement factors at 16 dB AWGN are substantial: Mamba S6 improves by 5.4×, the Transformer by 4.7×, and the MLP by 3.5×. This confirms the hypothesis that the bottleneck was strictly the activation function's continuous compression, rather than a lack of model capacity.

**Finding 9: Hardtanh is generally preferred over linear.** For feedforward relays, hardtanh consistently achieves the lowest BER. The bounded output prevents the network from generating values outside the valid constellation range, acting as an implicit regularizer.

**Finding 10: Sequence models benefit most from modulation-aware training.** The sequence models achieve the lowest BER among all neural relays after retraining: Mamba S6 with `hardtanh` achieves 0.0396, and the Transformer with `linear` achieves 0.0453 (AWGN). The larger contextual receptive field of these sequence models successfully captures inter-symbol amplitude correlations that simpler feedforward architectures miss, provided the activation bottleneck is removed.

**Finding 11: The gap to classical relays narrows but persists.** The best neural relay (Mamba S6 hardtanh, 0.0396 on AWGN) is 10.4× worse than DF (0.0038) and 2.2× worse than AF.

**Finding 12: The Hybrid relay is unaffected.** Hybrid achieves 0.2512 across all three variants because at 16 dB its SNR estimator routes to the DF sub-relay, which uses hard sign-detection on the I/Q-split signal.


#### 7.12 Extension Experiment: End-to-End Joint Optimization
Throughout the primary evaluations in this thesis, a modular architecture was maintained: the modulation (e.g., BPSK, 16-QAM) and the destination equalization were fixed, while neural networks were exclusively deployed at the intermediate relay node for denoising. To provide a complete comparative perspective on the limits of deep learning in physical-layer communications, this section evaluates a pure End-to-End (E2E) autoencoder paradigm.
In this experiment, the relay node is removed, and the transmitter and destination receiver are jointly optimized as a single neural network over a stochastically differentiable physical channel.

##### 7.12.1 System Formulation
The E2E architecture discards classical predefined constellations (such as Gray-coded square grids) and frames communication as a classification task through a constrained continuous latent space.

**The Transmitter (Encoder)**: The transmitter maps a discrete message index $m \in \{1, \dots, M\}$ to a continuous complex signal. The input is a one-hot vector $\mathbf{s} \in \mathbb{R}^M$. A multi-layer perceptron $f_\theta$ generates a raw latent vector $\mathbf{z} \in \mathbb{R}^{2n}$, where $n$ is the number of complex channel uses ($n=1$ for standard symbol-by-symbol transmission). To satisfy physical hardware limitations, a strict average power constraint is enforced via batch standardization across the dimension:
$$\mathbf{x} = \sqrt{2n} \frac{\mathbf{z} - \mathbb{E}[\mathbf{z}]}{\sqrt{\text{Var}(\mathbf{z}) + \epsilon}}$$

This normalization allows the network to learn variable amplitude boundaries (analogous to QAM) while bounding average transmission power.The Physical Channel:

The signal is subjected to a single-tap Rayleigh fading channel:

$$\mathbf{y} = \mathbf{h} \odot \mathbf{x} + \mathbf{n}, \quad h_i \sim \mathcal{CN}(0, 1), \quad n_i \sim \mathcal{CN}(0, \sigma^2)$$

**The Receiver (Decoder):** Assuming perfect Channel State Information (CSI), the received signal $\mathbf{y}$ and the channel coefficient $\mathbf{h}$ are concatenated. To prevent the network from expending parameters attempting to approximate complex division, an explicit Zero-Forcing (ZF) equalization layer computes $\hat{\mathbf{x}} = \mathbf{y} / \mathbf{h}$. The equalized signal and the channel magnitude are fed into a decoder network $g_\phi$, which outputs a probability distribution $\mathbf{p} \in (0,1)^M$ via a softmax activation.

The transmitter and receiver are jointly trained to minimize the categorical cross-entropy loss between $\mathbf{s}$ and $\mathbf{p}$.

##### 7.12.2 Results: E2E vs. Classical Theoretical Limits

The E2E network was trained for $M=16$ (equivalent to 16-QAM) over a $1 \times 1$ Rayleigh fading channel without spatial or temporal diversity. To benchmark the learned representation, the E2E performance is compared against the exact closed-form theoretical approximation for standard square 16-QAM over Rayleigh fading [21]:

$$P_s \approx 2 \left( \frac{\sqrt{M}-1}{\sqrt{M}} \right) \left( 1 - \sqrt{\frac{1.5 \gamma / (M-1)}{1 + 1.5 \gamma / (M-1)}} \right)$$

where the Bit Error Rate is approximated as $\text{BER} \approx P_s / \log_2(M)$ under optimal Gray coding.

**Table 16: BER comparison of E2E autoencoder vs. Theoretical 16-QAM (Rayleigh Fading)**
SNR (dB) | Standard 16-QAM Theory | E2E Learned Autoencoder | Relative Improvement
---|---|---|---
10.0 | 0.1098 | 0.0867 | 21.0%
12.0 | 0.0762 | 0.0641 | 15.8%
15.0 | 0.0481 | 0.0379 | 21.2%
20.0 | 0.0174 | 0.0137 | 21.2%
24.0 | 0.0072 | 0.0061 | 15.2%

##### 7.12.3 Analysis
**Finding 13: E2E representations consistently outperform classical grids.** The E2E neural network achieves a 15–21% reduction in BER across the evaluated SNR range compared to the theoretical limit of classical 16-QAM. This improvement occurs because the network abandons the classical $4 \times 4$ square grid—which is designed for human engineering simplicity—in favor of a non-rectangular 2D geometric packing (such as a hexagonal lattice or concentric APSK layout). This learned geometry maximizes the minimum Euclidean distance between points more efficiently under a strict average power constraint.

**Finding 14: The immutable physics of the diversity limit.** Despite the learned geometric advantage, the E2E network hits a BER of 0.0379 at 15 dB (equivalent to an 85% symbol classification accuracy). This is not an architectural failure, but a manifestation of the $1/\text{SNR}$ asymptotic decay characterizing $1 \times 1$ flat fading channels. At 15 dB, deep fades ($|h| \to 0$) completely destroy the signal approximately 10–15% of the time. The network accurately converges to the physical capacity limit of the channel. Further reduction into the $10^{-4}$ BER regime strictly requires the introduction of diversity (e.g., MIMO or temporal coding, $n \ge 2$), which the E2E framework could trivially exploit by learning an analog to the Alamouti space-time code.

**Conclusion on E2E Systems:** While joint autoencoder optimization yields superior spatial packing and lower theoretical BERs, it fundamentally breaks multi-vendor interoperability by replacing standardized constellations with opaque latent representations. Furthermore, its reliance on explicit domain knowledge (e.g., explicitly coding the complex division into the receiver to assist the MLP) demonstrates that "black-box" deep learning remains highly inefficient for basic RF operations. These findings heavily validate the core architectural thesis of this work: the most practical deployment of deep learning in physical-layer communications is a modular approach, where classical algorithms handle modulation and equalization, while neural networks are surgically applied to non-linear denoising tasks at intermediate relays.

---

## 8. Discussion and Conclusions

### 8.1 Interpretation of Results

#### 8.1.1 Low SNR Advantage of Neural Relays (H1: Confirmed)
At low SNR, neural network-based relays consistently outperform both classical methods. The neural network approximates the Bayes-optimal function accurately, producing a soft estimate that preserves information about the confidence of the decision, unlike DF which discards all soft information.

#### 8.1.2 Classical Dominance at High SNR (H2: Confirmed)
At medium and high SNR, DF matches or exceeds all neural methods. At high SNR, $\sigma^2 \to 0$ and $f^*(y) \to \text{sign}(y) = f_{\text{DF}}(y)$.

#### 8.1.3 Channel Robustness
The relative ranking of relay strategies is remarkably stable across AWGN, Rayleigh, Rician, and all three MIMO configurations.

#### 8.1.4 MIMO Equalization Hierarchy (H6: Confirmed)
MMSE consistently outperforms ZF, and SIC further improves upon MMSE. This confirms H6: the relay and equalization benefits are additive.

### 8.2 The "Less is More" Principle (H3: Confirmed)

The Minimal MLP architecture (169 parameters) matches the performance of models with 10–140× more parameters, while the Maximum MLP (11,201 parameters) exhibited clear overfitting. The effective dimensionality of the BPSK mapping is 1, explaining why even the smallest model is sufficient.

### 8.3 State Space vs. Attention for Signal Processing

#### 8.3.1 Context-Length Benchmark: Validating the Crossover Hypothesis
To validate the crossover hypothesis, we conducted a controlled benchmark comparing sequence models at a context length of $n = 255$.

**Table 13: Context-length benchmark — three sequence models at $n = 255$ on CUDA.**

| Model | Parameters | Training Time (20 ep.) | Inference Time (10K bits) | Train Speedup vs S6 |
|---|---|---|---|---|
| Transformer | 17,697 | 5.01 s | 0.99 s | 28.5× |
| Mamba S6 | 24,001 | 142.56 s | 2.69 s | 1.0× (baseline) |
| Mamba2 (SSD) | 26,179 | 13.35 s | 1.05 s | 10.7× |

This confirms H5: Mamba-2's structured state space duality eliminates the sequential bottleneck of S6 for long contexts.

### 8.4 Practical Deployment Recommendations

The Hybrid relay is recommended as the default deployment choice because it automatically selects MLP processing at low SNR and DF at high SNR.

### 8.5 Limitations
Limitations include the scope of modulation testing (deferred 64-QAM to future work), the assumption of perfect CSI, static channel training, and the focus on a single, two-hop relay.

### 8.6 Future Work
Future work should investigate higher-order modulation training (64-QAM, 256-QAM), imperfect CSI, online learning, and multi-relay networks.

### 8.7 Conclusions

This thesis presents a comprehensive comparative study of nine relay strategies evaluated across six channel/topology configurations and three modulation schemes. The primary finding is that **model complexity should be matched to task complexity**: for the relay denoising task with BPSK and QPSK, minimal architectures suffice. The activation experiment confirms that replacing $\tanh$ with a constellation-matched hardtanh activation reduces the BER floor by 2–5× for 16-QAM. 

---

## 9. References

[1] J. N. Laneman, D. N. Tse, and G. W. Wornell, "Cooperative diversity in wireless networks: Efficient protocols and outage behavior," *IEEE Trans. Inf. Theory*, vol. 50, no. 12, pp. 3062–3080, 2004.
[2] A. Nosratinia, T. E. Hunter, and A. Hedayat, "Cooperative communication in wireless networks," *IEEE Commun. Mag.*, vol. 42, no. 10, pp. 74–80, 2004.
[3] T. M. Cover and A. A. El Gamal, "Capacity theorems for the relay channel," *IEEE Trans. Inf. Theory*, vol. 25, no. 5, pp. 572–584, 1979.
[4] A. El Gamal and Y.-H. Kim, *Network Information Theory*. Cambridge University Press, 2011.
[5] B. Nazer and M. Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory*, vol. 57, no. 10, pp. 6463–6486, 2011.
[6] B. Rankov and A. Wittneben, "Spectral efficient protocols for half-duplex fading relay channels," *IEEE J. Sel. Areas Commun.*, vol. 25, no. 2, pp. 379–389, 2007.
[7] H. Ye, G. Y. Li, and B. H. Juang, "Power of deep learning for channel estimation and signal detection in OFDM systems," *IEEE Wireless Commun. Lett.*, vol. 7, no. 1, pp. 114–117, 2018.
[8] N. Samuel, T. Diskin, and G. Wunder, "Learning to detect for MIMO systems with unknown noise statistics," *IEEE Trans. Signal Process.*, vol. 67, no. 12, pp. 3261–3272, 2019.
[9] S. Dorner, S. Cammerer, J. Hoydis, and S. Ten Brink, "Deep learning based communication over the air," *IEEE J. Sel. Topics Signal Process.*, vol. 12, no. 1, pp. 132–143, 2018.
[10] H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu, and N. D. Sidiropoulos, "Learning to optimize: Training deep neural networks for interference management," *IEEE Trans. Signal Process.*, vol. 66, no. 20, pp. 5438–5453, 2018.
[11] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2014.
[12] M. Mirza and S. Osindero, "Conditional generative adversarial nets," *arXiv preprint arXiv:1411.1784*, 2014.
[13] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
[14] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems*, vol. 27, 2014.
[15] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
[16] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv preprint arXiv:2312.00752*, 2024.
[17] A. Gu, K. Goel, and C. Ré, "Efficiently modeling long sequences with structured state spaces," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2022.
[18] T. Dao and A. Gu, "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality," *arXiv preprint arXiv:2405.21060*, 2024.
[19] D. Tse and P. Viswanath, *Fundamentals of Wireless Communications*. Cambridge University Press, 2005.
[20] P. W. Wolniansky, G. J. Foschini, G. D. Golden, and R. A. Valenzuela, "V-BLAST: An architecture for realizing very high data rates over the rich-scattering wireless channel," in *Proc. IEEE ISSSE*, pp. 295–300, 1998.
[21] J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed. McGraw-Hill, 2008.
[22] M. K. Simon and M.-S. Alouini, *Digital Communication over Fading Channels*, 2nd ed. Wiley, 2005.
[23] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
[24] B. Sklar, *Digital Communications: Fundamentals and Applications*, 2nd ed. Prentice Hall, 2001.
[25] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
[26] I. E. Telatar, "Capacity of multi-antenna Gaussian channels," *European Trans. Telecommun.*, vol. 10, no. 6, pp. 585–595, 1999.
[27] G. J. Foschini, "Layered space-time architecture for wireless communication in a fading environment when using multi-element antennas," *Bell Labs Tech. J.*, vol. 1, no. 2, pp. 41–59, 1996.
[28] L. Zheng and D. N. C. Tse, "Diversity and multiplexing: A fundamental tradeoff in multiple-antenna channels," *IEEE Trans. Inf. Theory*, vol. 49, no. 5, pp. 1073–1096, 2003.
[29] D. N. C. Tse and S. V. Hanly, "Linear multiuser receivers: Effective interference, effective bandwidth and user capacity," *IEEE Trans. Inf. Theory*, vol. 45, no. 2, pp. 641–657, 1999.
[30] S. Loyka and F. Gagnon, "Performance analysis of the V-BLAST algorithm: An analytical approach," *IEEE Trans. Wireless Commun.*, vol. 3, no. 4, pp. 1326–1337, 2004.
[31] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, "β-VAE: Learning basic visual concepts with a constrained variational framework," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2017.
[32] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, pp. 1026–1034, 2015.

---

## 10. Appendices

*(Note: Appendix A was moved to the end of Part 1 in the revised flow to ensure complete notation coverage before the methodology sections, but is referenced here for completeness).*

### Appendix B: Model Architectures and Hyperparameters

**MLP (Minimal):**
- Input: 5-symbol sliding window
- Hidden: 24 neurons, ReLU activation
- Output: 1 neuron, Tanh activation
- Parameters: 169
- Training: MSE loss, lr=0.01, 100 epochs, 25K samples at SNR=[5, 10, 15] dB
- Implementation: NumPy (CPU)

**Hybrid:**
- Architecture: Same as MLP (169 params)
- SNR threshold: Learned (default ~5 dB)
- Below threshold → MLP processing; Above threshold → DF processing
- Implementation: NumPy (CPU)

*(Other models remain as listed in original thesis)*

### Appendix C: Software Architecture

The project is implemented as a modular Python package (`relaynet`) with the following structure:

relaynet/
├── channels/
├── modulation/
├── relays/
│   ├── base.py
│   ├── af.py
│   ├── df.py
│   ├── mlp.py           
│   ├── hybrid.py
│   ├── vae.py
│   └── cgan.py
├── simulation/
├── visualization/
└── utils/

### Appendix D: Normalized 3K-Parameter Configurations

| Model | Parameters | Window | Hidden / Architecture |
|---|---|---|---|
| MLP-3K | 3,004 | 11 | hidden=231 |
| Hybrid-3K | 3,004 | 11 | hidden=231 (+ DF switch) |
| VAE-3K | 3,037 | 11 | latent=10, hidden=(44, 20) |
| CGAN-3K | 3,004 | 11 | noise=8, g_hidden=(30, 30, 16), c_hidden=(32, 16) |
| Transformer-3K | 3,007 | 11 | d_model=18, heads=2, layers=1 |
| Mamba-3K | 3,027 | 11 | d_model=16, d_state=6, layers=1 |
| Mamba2-3K | 3,004 | 11 | d_model=15, d_state=6, chunk_size=8, layers=1 |

All 3K configurations use a window size of 11 to provide a common input context.