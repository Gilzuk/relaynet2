**To:** Gil Zukerman (M.Sc. Candidate) & University Senate
**Date:** April 2026
**Subject:** Review of M.Sc. Thesis: "Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies"

## 1. Overall Assessment
This is an outstanding piece of graduate-level research. The thesis bridges classical digital communication theory (information-theoretic bounds, fading channels, MIMO equalization) with cutting-edge artificial intelligence (Transformers, State Space Models like Mamba and Mamba-2 SSD, Generative Adversarial Networks). The scope of the work is highly ambitious, yet the execution is mathematically rigorous and methodologically sound. 

The systematic comparison of nine distinct relay strategies across six channel/topology configurations and four modulation schemes provides a definitive reference for the application of deep learning at the physical layer. I am particularly impressed by the empirical validation of the theoretical channel models before introducing the AI components, establishing a trustworthy baseline.

I fully endorse this thesis for approval by the University Senate, subject to a few minor clarifications detailed below.

---

## 2. Key Strengths & Contributions

### A. The "Less is More" Principle (Bias-Variance Analysis)
One of the most significant contributions of this work is the rigorous demonstration of the "Less is More" principle (Hypothesis 3). In an era where the AI community defaults to massively over-parameterized models, your theoretical grounding of the Bayes-optimal denoiser for BPSK in AWGN ($f^*(y) = \tanh(y/\sigma^2)$) perfectly explains why a minimal 169-parameter MLP is sufficient. Your demonstration that larger models (e.g., 11K+ parameter Transformers) suffer from variance-induced overfitting on this specific low-dimensional mapping task is a critical insight for practical, low-power, and low-latency telecom hardware deployments.

### B. Inclusion of State-of-the-Art Sequence Models
The application of Structured State Space Models (Mamba S6) and the novel Structured State Space Duality (Mamba-2 SSD) to physical-layer signal processing is highly original. By drawing the parallel between continuous-time LTI/LTV filters and the selective state transitions of Mamba, you have provided a compelling theoretical justification for why these architectures excel at time-series denoising compared to $O(N^2)$ attention mechanisms.

### C. Methodological Rigor
The simulation framework is exemplary. Evaluating 100,000 bits per SNR point across 10 independent trials and applying the Wilcoxon signed-rank test ensures that the reported gains (e.g., AI advantage at 0-4 dB SNR, and DF dominance at high SNR) are statistically robust. 

### D. Architectural Diversity & Negative Results
Including generative models (VAEs, CGANs) alongside discriminative models was a bold choice. The conclusion that generative models provide no significant advantage (and often underperform) due to the trivial discrete prior of the BPSK/QAM signal manifold is an excellent "negative result" that adds immense value to the literature.

---

## 3. Areas for Minor Improvement / Clarification (Prior to Final Submission)

Before final binding and submission, please consider addressing the following minor points to elevate the clarity of the manuscript:

1. **Generalization of Static Training:** 
   In Section 3.5, you mention that the AI relays are trained *offline* on synthetic AWGN data at a sparse SNR grid (5, 10, 15 dB) and then evaluated across Rayleigh, Rician, and MIMO channels. While you discuss the interpolation/extrapolation of the SNR grid, you should explicitly comment on *why* a model trained purely on AWGN noise generalizes so well to multiplicative fading environments. Briefly mention whether the sliding window implicitly allows the network to estimate local fading coefficients.

2. **Error Propagation in SIC vs. AI Relays:**
   In Section 1.9 (MIMO Equalization), you correctly identify error propagation as the main vulnerability of SIC. When combining an AI relay (Hop 1) with SIC equalization (Hop 2), does the "soft" non-linear output of the AI relay mitigate or exacerbate SIC error propagation compared to the hard-decision output of DF? A sentence or two in the discussion synthesizing this interaction would tie the two halves of the thesis together perfectly.

3. **Transition to End-to-End (E2E) Optimization:**
   Section 4.16 introduces E2E autoencoders. This is a paradigm shift from the modular approach (where the relay is just a denoiser) to jointly optimizing the transmitter and receiver. Ensure the narrative explicitly states that E2E is presented as an upper-bound/alternative paradigm to contrast with the modular constraints of the rest of the thesis.

4. **Future Work on Generative Models:**
   Since VAEs and CGANs struggled with simple AWGN/Fading BPSK tasks, you might add a note in "Future Work" (Section 5.6) suggesting that generative models could still hold promise in channels with highly non-Gaussian, structured interference (e.g., impulsive noise or non-linear amplifier clipping), where modeling the complex noise distribution is necessary.

---

## 4. Conclusion and Recommendation

**Recommendation: Approve for Submission (with minor typographical/narrative revisions).**

Gil, you have produced a thesis that is not only academically rigorous but also practically highly relevant to the future of 6G and AI-native air interfaces. The structured breakdown of the problem, the sheer volume of experimental validation, and the mature interpretation of the results reflect the qualities of an excellent researcher. 

Congratulations on an outstanding master's thesis.

Sincerely,

**[Professor's Name]**  
*Professor of Digital Communication and AI Science*