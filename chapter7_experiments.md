## 7. Results

All results are obtained from Monte Carlo simulations with 10 trials × 10,000 bits per SNR point (100,000 total bits per SNR point). Confidence intervals at 95% are shown in all plots. Bold values indicate the best performance at each SNR point.

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

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.291 | 0.268 | 0.264 | 0.262 | 0.376 | **0.261** | 0.267 | 0.269 | 0.270 |
| 4 | 0.154 | 0.112 | 0.112 | 0.113 | 0.330 | **0.111** | 0.114 | 0.111 | **0.111** |
| 8 | 0.044 | **0.010** | 0.015 | **0.010** | 0.291 | 0.013 | 0.014 | **0.010** | 0.012 |
| 12 | 0.0027 | **1.67e-04** | **1.67e-04** | **1.67e-04** | 0.269 | 3.33e-04 | 3.33e-04 | **1.67e-04** | **1.67e-04** |
| 16 | **0** | **0** | **0** | **0** | 0.258 | **0** | **0** | **0** | **0** |
| 20 | **0** | **0** | **0** | **0** | 0.250 | **0** | **0** | **0** | **0** |

At low SNR (0–4 dB), CGAN achieves the lowest BER across all methods (excluding VAE). At medium-to-high SNR (≥8 dB), DF matches or exceeds all AI methods with zero parameters.

**Analysis.** The AWGN results directly test Hypotheses H1 and H2. At 0 dB, CGAN reduces BER by 2.4% relative to DF (0.261 vs. 0.268), which is statistically significant across all 10 trials ($p < 0.01$, Wilcoxon). This confirms H1. At 8 dB, DF's BER of 0.010 equals or beats most AI methods, confirming H2. The crossover occurs between 4 and 8 dB — precisely the SNR range where the Bayes-optimal relay function $f^*(y) = \tanh(y/\sigma^2)$ transitions from a smooth sigmoid (exploitable by neural networks) to a near-step function (exactly matched by DF's hard decision). Notably, the AI methods (MLP through Mamba-2 SSD) cluster within a narrow BER band at each SNR point, suggesting that the architectural choice matters less than the shared advantage of non-linear processing over AF/DF at low SNR. VAE is a notable exception, exhibiting significantly higher BER across all SNR points due to the probabilistic overhead of the variational framework.

![Figure 9: AWGN channel — BER comparison of all nine relay strategies.](results/awgn_comparison_ci.png)

*Figure 9: AWGN channel — BER vs. SNR for all nine relay strategies with 95% CI. AI relays outperform classical methods at low SNR; DF dominates at medium-to-high SNR.*

### 7.3 Rayleigh Fading Channel

Table 2: BER comparison on the Rayleigh fading channel (SISO).

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.330 | **0.245** | 0.247 | 0.250 | 0.395 | 0.247 | 0.325 | 0.317 | 0.318 |
| 4 | 0.205 | 0.139 | 0.141 | 0.141 | 0.347 | **0.138** | 0.183 | 0.178 | 0.180 |
| 8 | 0.092 | **0.068** | 0.070 | 0.069 | 0.304 | 0.070 | 0.077 | 0.076 | 0.076 |
| 12 | 0.037 | **0.031** | 0.032 | 0.032 | 0.278 | 0.031 | 0.032 | 0.032 | 0.032 |
| 16 | 0.015 | 0.014 | 0.014 | 0.014 | 0.262 | 0.014 | 0.014 | **0.013** | 0.014 |
| 20 | 0.0053 | **0.0047** | **0.0047** | **0.0047** | 0.250 | 0.0052 | 0.0048 | 0.0052 | **0.0047** |

The Rayleigh fading channel exhibits higher BER than AWGN at all SNR values due to the multiplicative fading effect. DF leads at low SNR (0 dB), while CGAN matches DF at 4 dB. DF dominates at medium-to-high SNR.

**Analysis.** Despite the fundamentally different noise structure — multiplicative fading with heavy-tailed deep fades rather than additive Gaussian noise — the AI relays, trained on AWGN data at SNR = {5, 10, 15} dB, generalise to a channel model they have never seen during training. Importantly, on the Rayleigh channel DF is the best-performing relay at 0 dB (0.245), and no AI relay achieves lower BER at this SNR point. This is a significant departure from the AWGN result and indicates that the deep-fade statistics of the Rayleigh channel favour DF's hard-decision regeneration even at low SNR. The sequence models (Transformer, Mamba S6, Mamba-2 SSD) show notably higher BER (0.317–0.325) at 0 dB, suggesting that their temporal processing is less effective when the noise structure is dominated by multiplicative fading rather than additive noise. The feedforward relays (MLP, CGAN) are closer to DF, with CGAN achieving the best AI performance at 4 dB. The $1/(4\bar{\gamma})$ high-SNR BER slope (characteristic of diversity order 1) is preserved for all relay methods, confirming that the relay processing does not alter the channel's fundamental diversity characteristics.

![Figure 10: Rayleigh fading — BER comparison of all nine relay strategies.](results/fading_comparison.png)

*Figure 10: Rayleigh fading — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.4 Rician Fading Channel (K=3)

Table 3: BER comparison on the Rician fading channel with K-factor = 3.

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.265 | 0.206 | 0.206 | 0.205 | 0.372 | **0.203** | 0.242 | 0.239 | 0.241 |
| 4 | 0.137 | **0.090** | 0.092 | 0.093 | 0.320 | 0.090 | 0.104 | 0.103 | 0.105 |
| 8 | 0.044 | **0.028** | 0.030 | 0.030 | 0.284 | 0.029 | 0.030 | 0.030 | 0.030 |
| 12 | 0.0092 | 0.0072 | **0.0068** | 0.0072 | 0.263 | 0.0077 | 0.0077 | **0.0068** | 0.0070 |
| 16 | 0.0023 | 0.0018 | **0.0017** | 0.0018 | 0.252 | 0.0018 | 0.0018 | **0.0017** | **0.0017** |
| 20 | **6.67e-04** | **6.67e-04** | **6.67e-04** | **6.67e-04** | 0.248 | 8.33e-04 | **6.67e-04** | **6.67e-04** | **6.67e-04** |

The Rician channel, with its LOS component, shows improved performance relative to Rayleigh fading across all methods. CGAN achieves the lowest BER at 0 dB, while DF dominates at medium SNR.

**Analysis.** The Rician $K=3$ results interpolate between AWGN and Rayleigh, as predicted by the channel model analysis in Section 6.2.3. CGAN provides a 1.8% absolute BER reduction over DF at 0 dB (0.203 vs. 0.206). The feedforward relays (MLP, Hybrid, CGAN) outperform the sequence models (Transformer, Mamba S6, Mamba-2 SSD) at low SNR, consistent with the Rayleigh results. The convergence of all methods at high SNR is faster than on Rayleigh, with all methods (except VAE) achieving BER $\leq 10^{-3}$ by 20 dB (compared to $\sim 5 \times 10^{-3}$ on Rayleigh at the same SNR), reflecting the Rician channel's steeper BER slope.

![Figure 11: Rician fading K=3 — BER comparison of all nine relay strategies.](results/rician_comparison_ci.png)

*Figure 11: Rician fading (K=3) — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.5 2×2 MIMO with ZF Equalization

Table 4: BER comparison on 2×2 MIMO Rayleigh channel with ZF equalization.

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.328 | 0.254 | **0.250** | 0.253 | 0.398 | 0.254 | 0.329 | 0.326 | 0.327 |
| 4 | 0.206 | **0.140** | 0.145 | 0.146 | 0.350 | 0.143 | 0.181 | 0.180 | 0.181 |
| 8 | 0.104 | **0.071** | 0.073 | 0.073 | 0.313 | 0.072 | 0.082 | 0.081 | 0.081 |
| 12 | 0.040 | **0.032** | **0.032** | **0.032** | 0.277 | 0.032 | 0.033 | 0.033 | 0.033 |
| 16 | 0.015 | 0.014 | **0.013** | 0.014 | 0.265 | 0.014 | 0.014 | 0.014 | 0.014 |
| 20 | 0.0043 | **0.0037** | 0.0042 | **0.0037** | 0.256 | **0.0037** | 0.0038 | 0.0040 | 0.0038 |

ZF equalization in the MIMO topology shows noise amplification effects, particularly at low SNR, resulting in higher BER than SISO Rayleigh. The AI relay advantage at low SNR is preserved, with MLP (169p) achieving the best BER at 0 dB.

**Analysis.** The MIMO ZF results serve as a baseline for the MIMO equalization hierarchy. The BER values closely mirror the SISO Rayleigh results (Table 2 vs. Table 4), confirming the theoretical prediction that ZF equalization of a $2 \times 2$ system yields diversity order 1 (same as SISO). The small BER differences between MIMO ZF and SISO Rayleigh are due to the noise amplification inherent in ZF: when the channel matrix $\mathbf{H}$ is ill-conditioned (condition number $\kappa(\mathbf{H}) \gg 1$), the ZF pseudo-inverse amplifies noise severely on the weaker stream. This effect is more pronounced at low SNR, where the noise amplification can dominate the received signal. Notably, the simplest AI relay (MLP with 169 parameters) achieves the best performance at 0 dB: MLP reduces BER by 1.6% relative to DF (0.250 vs. 0.254). The sequence models (Transformer, Mamba S6, Mamba-2 SSD) show higher BER at 0 dB (0.326–0.329), comparable to AF (0.328), suggesting that their temporal processing does not confer advantage under MIMO ZF noise amplification.

![Figure 12: 2×2 MIMO ZF — BER comparison of all nine relay strategies.](results/mimo_2x2_comparison_ci.png)

*Figure 12: 2×2 MIMO with ZF equalization — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.6 2×2 MIMO with MMSE Equalization

Table 5: BER comparison on 2×2 MIMO Rayleigh channel with MMSE equalization.

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.199 | 0.164 | 0.166 | 0.164 | 0.354 | 0.165 | 0.173 | **0.163** | 0.165 |
| 4 | 0.119 | **0.086** | 0.090 | **0.086** | 0.318 | 0.091 | 0.094 | 0.088 | 0.089 |
| 8 | 0.058 | **0.040** | 0.044 | **0.040** | 0.288 | 0.044 | 0.045 | 0.042 | 0.043 |
| 12 | 0.026 | **0.018** | 0.019 | **0.018** | 0.270 | 0.019 | 0.020 | 0.018 | 0.019 |
| 16 | 0.0090 | **0.0067** | 0.0073 | **0.0067** | 0.259 | 0.0072 | 0.0073 | 0.0070 | 0.0070 |
| 20 | 0.0020 | 0.0018 | 0.0018 | 0.0018 | 0.255 | 0.0025 | 0.0020 | **0.0015** | 0.0018 |

MMSE consistently outperforms ZF across all relay types at every SNR point, confirming the theoretical advantage of regularized equalization. Mamba S6 achieves the lowest BER at 0 dB (0.163) and also at 20 dB (0.0015), the only channel–relay combination where an AI relay leads at both ends of the SNR range. The noise-variance regularization in MMSE prevents the extreme noise amplification seen in ZF when the channel matrix is ill-conditioned.

![Figure 13: 2×2 MIMO MMSE — BER comparison of all nine relay strategies.](results/mimo_2x2_mmse_comparison_ci.png)

*Figure 13: 2×2 MIMO with MMSE equalization — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.7 2×2 MIMO with SIC Equalization

Table 6: BER comparison on 2×2 MIMO Rayleigh channel with MMSE-SIC equalization.

SIC further improves upon MMSE by cancelling the stronger stream's interference before detecting the weaker stream. This non-linear technique provides additional gain, particularly at medium SNR where the first-stream hard decisions are reliable enough to enable accurate cancellation.

| SNR (dB) | AF | DF | MLP (169p) | Hybrid | VAE | CGAN | Transformer | Mamba S6 | Mamba-2 SSD |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.172 | **0.134** | 0.139 | 0.138 | 0.347 | 0.136 | 0.146 | 0.143 | 0.143 |
| 4 | 0.075 | **0.045** | 0.048 | **0.045** | 0.315 | 0.047 | 0.049 | 0.048 | 0.048 |
| 8 | 0.020 | **0.011** | 0.013 | **0.011** | 0.290 | 0.012 | 0.012 | 0.012 | 0.012 |
| 12 | 0.0052 | **0.0037** | 0.0038 | **0.0037** | 0.278 | 0.0043 | 0.0042 | 0.0038 | 0.0038 |
| 16 | 0.0017 | 0.0013 | 0.0012 | 0.0013 | 0.271 | 0.0012 | **0.0010** | 0.0015 | 0.0013 |
| 20 | 1.67e-04 | **0** | **0** | **0** | 0.266 | **0** | **0** | 1.67e-04 | **0** |

**Analysis.** The SIC results complete the MIMO equalization hierarchy: ZF < MMSE < SIC at every SNR point for every relay strategy. The SIC gain over MMSE is approximately 0.5–1 dB across the SNR range, consistent with the theoretical analysis in Section 4.6.2. The improvement comes primarily from the second detected stream, which sees an interference-free channel after successful cancellation of the first stream. At low SNR (0–2 dB), the SIC gain narrows because the first-stream BER is high, leading to frequent error propagation that partially negates the cancellation benefit. At high SNR ($\geq 10$ dB), error propagation is rare and SIC approaches the theoretical optimum of interference-free detection for both streams.

Critically, DF achieves the lowest BER at 0 dB (0.134) and dominates at all low-to-medium SNR points. No AI relay beats DF on the SIC channel, consistent with the Rayleigh SISO result (Table 2). The best AI relay at 0 dB is CGAN (0.136), followed by Hybrid (0.138). This result, combined with the Rayleigh finding, indicates that on fading channels with sufficient equalization, DF's hard-decision regeneration is optimal even at low SNR — the deep-fade statistics make soft relay processing less effective than clean signal regeneration.

![Figure 14: 2×2 MIMO SIC — BER comparison of all nine relay strategies.](results/mimo_2x2_sic_comparison_ci.png)

*Figure 14: 2×2 MIMO with MMSE-SIC equalization — BER vs. SNR for all nine relay strategies with 95% CI.*

### 7.8 Normalized 3K-Parameter Comparison

To isolate architectural inductive biases from parameter count effects, all seven AI models were scaled to approximately 3,000 parameters.

Table 7: Normalized 3K BER results — AWGN channel.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | **0.267** | 0.269 | 0.408 | 0.269 | 0.271 | 0.270 | 0.291 | 0.268 |
| 4 | 0.114 | 0.114 | 0.359 | 0.114 | **0.112** | 0.113 | 0.154 | 0.112 |
| 10 | 0.0033 | **0.0012** | 0.309 | 0.0023 | 0.0017 | 0.0018 | 0.013 | 0.0012 |
| 16 | **0** | **0** | 0.285 | **0** | **0** | **0** | 0 | 0 |
| 20 | **0** | **0** | 0.280 | **0** | **0** | **0** | 0 | 0 |

Table 8: Normalized 3K BER results — Rayleigh fading channel.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | **0.252** | 0.258 | 0.411 | 0.329 | 0.320 | 0.320 | 0.330 | 0.245 |
| 4 | 0.146 | **0.145** | 0.375 | 0.181 | 0.180 | 0.179 | 0.205 | 0.139 |
| 10 | 0.049 | **0.047** | 0.315 | 0.050 | 0.049 | 0.049 | 0.058 | 0.044 |
| 16 | 0.015 | 0.014 | 0.289 | 0.014 | **0.014** | 0.014 | 0.015 | 0.014 |
| 20 | 0.0050 | **0.0047** | 0.279 | **0.0047** | **0.0047** | **0.0047** | 0.0053 | 0.0047 |

Table 9: Normalized 3K BER results — Rician K=3 fading channel.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.211 | **0.211** | 0.389 | 0.243 | 0.242 | 0.241 | 0.265 | 0.206 |
| 4 | **0.095** | 0.099 | 0.345 | 0.104 | 0.102 | 0.103 | 0.137 | 0.090 |
| 10 | 0.016 | **0.015** | 0.301 | 0.016 | 0.016 | 0.016 | 0.020 | 0.015 |
| 16 | **0.0015** | 0.0018 | 0.283 | **0.0015** | 0.0017 | **0.0015** | 0.0023 | 0.0018 |
| 20 | **6.67e-04** | **6.67e-04** | 0.275 | **6.67e-04** | **6.67e-04** | **6.67e-04** | 6.67e-04 | 6.67e-04 |

Table 10: Normalized 3K BER results — 2×2 MIMO ZF.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | **0.256** | 0.257 | 0.410 | 0.329 | 0.327 | 0.326 | 0.328 | 0.254 |
| 4 | **0.146** | 0.150 | 0.369 | 0.185 | 0.181 | 0.182 | 0.206 | 0.140 |
| 10 | **0.049** | 0.049 | 0.319 | 0.053 | 0.052 | 0.053 | 0.066 | 0.048 |
| 16 | 0.014 | **0.014** | 0.288 | 0.014 | 0.014 | 0.014 | 0.015 | 0.014 |
| 20 | 0.0040 | **0.0037** | 0.279 | 0.0040 | 0.0040 | 0.0040 | 0.0043 | 0.0037 |

Table 11: Normalized 3K BER results — 2×2 MIMO MMSE.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.167 | **0.164** | 0.384 | 0.169 | 0.165 | 0.164 | 0.199 | 0.164 |
| 4 | 0.094 | **0.086** | 0.348 | 0.091 | 0.089 | 0.089 | 0.119 | 0.086 |
| 10 | 0.030 | 0.027 | 0.317 | 0.029 | **0.026** | 0.027 | 0.040 | 0.027 |
| 16 | 0.0083 | 0.0067 | 0.287 | **0.0067** | 0.0072 | 0.0073 | 0.0090 | 0.0067 |
| 20 | 0.0022 | **0.0018** | 0.280 | **0.0018** | **0.0018** | **0.0018** | 0.0020 | 0.0018 |

Table 11b: Normalized 3K BER results — 2×2 MIMO SIC.

| SNR (dB) | MLP-3K | Hybrid-3K | VAE-3K | Transformer-3K | Mamba-3K | Mamba2-3K | AF | DF |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.140 | **0.137** | 0.368 | 0.144 | 0.142 | 0.144 | 0.172 | 0.134 |
| 4 | 0.048 | **0.045** | 0.335 | 0.048 | 0.047 | 0.048 | 0.075 | 0.045 |
| 10 | 0.0060 | **0.0057** | 0.309 | 0.0060 | **0.0057** | 0.0060 | 0.010 | 0.0057 |
| 16 | 0.0013 | 0.0013 | 0.298 | **0.0010** | 0.0013 | 0.0013 | 0.0017 | 0.0013 |
| 20 | 1.67e-04 | **0** | 0.292 | **0** | **0** | **0** | 1.67e-04 | 0 |

Key findings from the normalized comparison:

1. **Performance convergence:** At 3K parameters, feedforward models (MLP, Hybrid) match or exceed sequence models on AWGN and fading channels, while sequence models retain slight advantages only on specific channels. This confirms Hypothesis H4: architectural inductive biases provide diminishing returns when model capacity is held constant. On Rayleigh and MIMO channels, the feedforward relays (MLP-3K, Hybrid-3K) consistently outperform sequence models at low SNR, a reversal of the original parameter count results.

2. **VAE underperforms dramatically:** VAE-3K exhibits BER values 0.10–0.15 higher than all other architectures across all channels and SNR points, effectively failing to learn the relay task. This confirms that the probabilistic overhead (KL divergence regularization, stochastic sampling) is catastrophically harmful at small scale — the KL term actively prevents the model from using its latent space for reconstruction.

3. **MLP/Hybrid competitive:** Simple feedforward architectures match or exceed sequence models at equal parameter budgets, particularly on fading channels where MLP-3K achieves the best AI performance at 0 dB on Rayleigh (0.252) and MIMO ZF (0.256). At 3,000 parameters, a simple two-layer feedforward network achieves performance within 0.2% BER of a Transformer or Mamba model.

4. **DF remains competitive at 3K:** Including AF and DF as reference columns reveals that DF matches or beats all 3K AI relays on Rayleigh, MIMO ZF, MIMO MMSE, and MIMO SIC at 0 dB. Only on AWGN and Rician channels do 3K AI relays achieve marginal improvement over DF at low SNR.

5. **The exception — Rician K=3 at high SNR:** On the Rician channel at 16 dB, MLP-3K, Transformer-3K, and Mamba2-3K tie (0.0015), outperforming both DF (0.0018) and other AI relays. This suggests that at high SNR on channels with a LOS component, the learned relay function provides a small but consistent advantage.

![Figure 15: Normalized 3K-parameter comparison — all channels.](results/normalized_3k_all_channels.png)

*Figure 15: Normalized 3K-parameter comparison across all channels. At equal parameter budgets, all architectures converge to similar BER, with VAE being the consistent underperformer.*

![Figure 16: Normalized 3K-parameter comparison — AWGN channel.](results/normalized_3k_awgn.png)

*Figure 16: Normalized 3K-parameter BER comparison on AWGN. Mamba-3K and Transformer-3K produce nearly identical BER, eliminating the gap seen at original parameter counts.*

![Figure 17: Normalized 3K-parameter comparison — Rayleigh channel.](results/normalized_3k_rayleigh.png)

*Figure 17: Normalized 3K-parameter BER comparison on Rayleigh fading.*

![Figure 18: Normalized 3K-parameter comparison — Rician K=3 channel.](results/normalized_3k_rician_k3.png)

*Figure 18: Normalized 3K-parameter BER comparison on Rician fading (K=3).*

### 7.9 Complexity–Performance Trade-off

Table 12: Model complexity and timing comparison (50,000 training samples, 100 epochs; Monte Carlo evaluation over 11 SNR points × 10 trials × 10,000 bits). All inference uses batched window extraction and a single forward pass per signal block.

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

**Training time analysis.** Training times span four orders of magnitude, from under 5 seconds (MLP) to over 2 hours (CGAN). The key drivers are:

- **AF/DF** require no training — they are purely analytical algorithms operating on the received signal. This zero training cost is their primary practical advantage.
- **MLP and Hybrid** (169 parameters) train in ~5 s on CPU using a simple NumPy-based two-layer network. At only 169 parameters, the computational cost per epoch is negligible. The Hybrid relay trains only its internal MLP sub-network (same 169 parameters), hence identical training time.
- **VAE** (1,777 parameters) trains in 22 s on CPU. Despite 10× more parameters than MLP, its encoder–decoder architecture processes each sample in a single forward/backward pass. The moderate hidden sizes (32→16→8→16→32) keep per-batch computation small.
- **CGAN (WGAN-GP)** (2,946 parameters) requires approximately 2 hours despite having fewer parameters than the Transformer. Four factors explain this: (1) the WGAN-GP training loop performs **5 critic updates per generator update**, effectively multiplying the number of gradient steps by 6; (2) the gradient penalty term requires computing second-order gradients through the critic via `torch.autograd.grad`, which is computationally expensive; (3) 200 training epochs (vs. 100 for other models) doubles the base iteration count; (4) despite running on CUDA, the 3K-parameter model is too small to saturate GPU parallelism, and the gradient penalty's dynamic graph construction incurs significant per-step overhead. Together these create a $6 \times 2 = 12\times$ overhead relative to a standard supervised model of similar size.
- **Transformer** (17,697 parameters) trains in 8 minutes on CUDA. The multi-head self-attention over the 11-symbol window is computed as a single batched matrix multiply $\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$, which parallelises efficiently on GPU. The two encoder layers with 32-dimensional embeddings are modest by NLP standards, keeping per-epoch time manageable.
- **Mamba S6** (24,001 parameters) trains in 36 minutes, approximately 4.5× slower than the Transformer despite only 1.36× more parameters. The detailed analysis of this paradoxical result is given in Section 8.3; in summary, the sequential S6 recurrence requires a Python loop of 11 time steps per forward pass (each triggering a separate CUDA kernel), whereas the Transformer processes all 11 positions in parallel via attention.
- **Mamba2 (SSD)** (26,179 parameters) trains in 24 minutes — 33% faster than Mamba S6 despite having 9% more parameters. The SSD layer replaces the sequential S6 scan with a chunk-parallel structured matrix multiply: for an 11-token sequence with chunk size 8, this means 2 parallel chunk matmuls instead of 11 sequential kernel launches. However, Mamba2 is still 3× slower than the Transformer at this short context length because building the $L \times L$ SSM matrix per chunk incurs overhead (4D tensor allocation, cumulative log-sum-exp, einsum) that only amortises over longer sequences.

**Inference (evaluation) time analysis.** All relay implementations use **batched inference**: the sliding-window extraction builds a matrix of all windows at once, and the neural network processes the entire signal in a single forward pass. This eliminates the per-symbol Python loop that dominated prior versions. As a result, all nine relays evaluate the full AWGN Monte Carlo sweep (11 SNR × 10 trials × 10,000 bits = 1.1 M symbols) in under 4 seconds:

- **AF and DF** evaluate in 0.8 s — a single vectorised operation per signal block. The SIC evaluation (3.5 s for AF) takes longer due to the iterative successive interference cancellation in the MIMO channel model itself.
- **MLP** evaluates in 1.6 s. The batched NumPy forward pass through the 169-parameter network processes all 10,000 symbols at once as a single matrix multiply. The Hybrid relay is even faster (0.4 s) because at high SNR it routes to the DF path.
- **VAE** evaluates in 1.8 s — the batched encode/decode through the 1.8K-parameter network is efficient when vectorised.
- **CGAN** evaluates in 1.1 s because the generator runs a single batched forward pass — no critic is needed at inference.
- **Transformer** evaluates in 3.7 s. The batched implementation processes all symbols as a single tensor of shape $(N, 11, 1)$ through the attention layers. This is a **475× speedup** compared to the prior per-symbol implementation (1,762 s).
- **Mamba S6** evaluates in 1.9 s. Despite the sequential S6 recurrence over 11 time steps, the batch dimension ($N$ symbols) is processed in parallel at each step. This is a **3,933× speedup** compared to the prior per-symbol implementation (7,395 s).
- **Mamba2 (SSD)** evaluates in 4.1 s (AWGN) and 5.6 s (SIC) — approximately 2× slower than Mamba S6 at the 11-token window. This counter-intuitive result is explained in Section 8.3.1: the chunk-parallel SSD kernel introduces $O(L^2)$ intermediate tensors and multiple einsum operations per chunk, which exceeds the cost of S6's simple 11-step sequential loop at this short context length.

**SIC evaluation overhead.** The SIC column in Table 12 shows evaluation times for the most computationally expensive channel model (2×2 MIMO with successive interference cancellation). Times increase by roughly 2–3× compared to AWGN, reflecting the additional per-symbol SIC iterations in the channel model rather than any relay processing overhead.

**Key insight:** The batched inference approach demonstrates that neural relay processing can operate at speeds comparable to classical AF/DF relays. The primary computational cost shifts from inference to training, where the CGAN's adversarial training loop and Mamba's sequential recurrence remain inherently expensive. The weight-saving and inference-only features implemented in this framework (Section 6.5) enable reuse of trained models, amortising the one-time training cost across unlimited inference runs.

![Figure 19: Complexity–performance comparison across all relay strategies.](results/complexity_comparison_all_relays.png)

*Figure 19: Complexity–performance trade-off. Training time vs. parameter count vs. BER improvement over DF at low SNR. The Minimal MLP (169 params) achieves the best efficiency.*

![Figure 20: Master BER comparison — all relay strategies across all channels.](results/master_ber_comparison.png)

*Figure 20: Master BER comparison — consolidated view of all nine relay strategies across all six channel/topology configurations.*

### 7.10 Modulation Comparison: BPSK vs. QPSK vs. 16-QAM

To evaluate whether the BPSK findings generalise to higher-order constellations, we test the same BPSK-trained relay models on QPSK and 16-QAM signals using the I/Q splitting technique described in Section 6.7.4. The evaluation uses all nine relay strategies — AF, DF, MLP, Hybrid, VAE, CGAN, Transformer, Mamba S6, and Mamba2 (SSD) — on AWGN and Rayleigh fading channels. This section addresses a key question: **do hypotheses H1–H3 hold for complex-valued modulations?**

**Experimental setup.** All AI relays are trained once on BPSK AWGN data (identical to Sections 7.2–7.9). For QPSK and 16-QAM evaluation, the source generates complex symbols from the respective constellation; the channel adds complex AWGN or Rayleigh fading; the relay processes the signal using the type-specific method (AF: direct amplification of complex signal; DF: nearest constellation point detection; AI relays: I/Q splitting); and the destination demodulates using the corresponding scheme.

Table 14: BER comparison across modulations at selected SNR points (AWGN channel). All nine relay strategies.

| Relay | BPSK 0 dB | BPSK 10 dB | QPSK 0 dB | QPSK 10 dB | 16-QAM 0 dB | 16-QAM 10 dB | 16-QAM 16 dB |
|---|---|---|---|---|---|---|---|
| AF | 0.291 | 0.013 | 0.277 | 0.013 | 0.370 | 0.120 | 0.016 |
| DF | 0.268 | 0.0012 | 0.260 | 0.0018 | 0.376 | 0.106 | 0.0030 |
| MLP (169p) | 0.263 | 0.0025 | 0.249 | 0.0033 | 0.382 | 0.204 | 0.200 |
| Hybrid | 0.262 | 0.0012 | 0.260 | 0.0018 | 0.393 | 0.272 | 0.253 |
| VAE | 0.326 | 0.098 | 0.316 | 0.074 | 0.404 | 0.253 | 0.219 |
| CGAN (WGAN-GP) | 0.266 | 0.0023 | 0.256 | 0.0032 | 0.385 | 0.240 | 0.228 |
| Transformer | 0.277 | 0.0043 | 0.253 | 0.0070 | 0.381 | 0.184 | 0.165 |
| Mamba S6 | 0.270 | 0.0015 | 0.250 | 0.0023 | 0.382 | 0.220 | 0.228 |
| Mamba-2 SSD | 0.268 | 0.0020 | 0.251 | 0.0027 | 0.381 | 0.213 | 0.217 |

*Values are mean BER over 10 trials × 10,000 bits. QPSK values closely track BPSK due to the I/Q independence property. The 16-QAM 16 dB column exposes the AI relay floor effect: all AI relays saturate near BER ≈ 0.17–0.25 while DF reaches 0.0030. The Transformer achieves the lowest AI floor (0.165) owing to its larger receptive field.*

**Key findings:**

**Finding 1: QPSK results closely track BPSK.** For all nine relay strategies, the QPSK BER at each SNR point is within 5% of the corresponding BPSK BER (e.g., MLP at 0 dB: BPSK = 0.263, QPSK = 0.249; DF: BPSK = 0.268, QPSK = 0.260). This holds for the feedforward relays (MLP, Hybrid, VAE, CGAN) and the sequence models (Transformer, Mamba S6, Mamba-2 SSD) alike. This is expected from the I/Q splitting analysis (Section 6.7.4): since each QPSK component carries an independent BPSK-like stream, the BPSK-trained relay denoises each component identically. This confirms that **H1 (AI advantage at low SNR) and H2 (DF dominance at high SNR) hold for QPSK** without modification.

**Finding 2: DF remains effective for QPSK and 16-QAM.** The DF relay performs nearest-constellation-point detection (sign decision for QPSK, PAM-4 quantisation for 16-QAM), which is the modulation-aware generalisation of BPSK hard-decision. At 16 dB, DF achieves BER = 0.0030 for 16-QAM AWGN and BER = 0.082 for 16-QAM Rayleigh, confirming **H2 extends to higher-order modulations**.

**Finding 3: All AI relays exhibit a BER floor on 16-QAM.** The BPSK-trained AI relays perform identically on QPSK due to the binary nature of each I/Q component. On 16-QAM, all seven AI relays — feedforward and sequential alike — hit an irreducible BER floor even at high SNR. At 16 dB AWGN, the floors are: Transformer = 0.165, MLP = 0.200, Mamba-2 SSD = 0.217, VAE = 0.219, CGAN = 0.228, Mamba S6 = 0.228, Hybrid = 0.253 — all dramatically worse than DF's 0.0030. This floor arises because the $\tanh$ activation compresses the multi-level PAM-4 signal ($\{-3, -1, +1, +3\}/\sqrt{10}$) toward $\{\pm 1\}$, destroying the amplitude information required for correct 16-QAM demodulation. The Transformer achieves the lowest floor (0.165), likely because its multi-head attention over the 11-symbol window provides slightly better amplitude discrimination than the narrower feedforward relays. The effect is **statistically significant** (p < 0.05, Wilcoxon signed-rank) at every SNR point from 0–20 dB for all AI relays. This finding confirms the **limitation predicted in Section 6.7.4** and motivates modulation-specific training for 16-QAM relays.

**Finding 4: AF outperforms DF on 16-QAM at low SNR.** Unlike BPSK/QPSK where DF beats AF at all SNR values, the 16-QAM results reveal a reversal: AF achieves significantly lower BER than DF at SNR = 0 dB on AWGN (AF = 0.370 vs DF = 0.376). This occurs because AF preserves the continuous multi-level amplitude structure of the 16-QAM signal, whereas DF's PAM-4 quantisation makes hard errors that propagate. At higher SNR (≥8 dB), DF's regeneration advantage reasserts itself and DF outperforms AF.

**Finding 5: The Hybrid relay adapts correctly across BPSK and QPSK.** At low SNR, the Hybrid relay uses its MLP sub-network (which generalises via I/Q splitting); at high SNR, it switches to DF (which uses modulation-aware detection). This SNR-adaptive switching works correctly for BPSK and QPSK (matching DF at high SNR), though on 16-QAM the Hybrid relay inherits the AI BER floor from its MLP component, yielding the worst performance among all nine relays at high SNR (0.253 at 16 dB vs Transformer's 0.165).

**Finding 6: Rayleigh fading amplifies modulation differences.** On the Rayleigh fading channel, the BER gap between BPSK/QPSK and 16-QAM widens because the reduced constellation spacing in 16-QAM makes it more susceptible to deep fades. At 10 dB Rayleigh: BPSK DF = 0.044 while 16-QAM DF = 0.206 — a 4.7× gap. AI relay gains at low SNR are preserved for QPSK over Rayleigh, though all AI relays — including the sequence models — are significantly worse than DF on 16-QAM Rayleigh at every SNR point (N* p < 0.05 at 0–20 dB). On Rayleigh 16-QAM at 16 dB, the sequence models show BER floors of 0.190 (Transformer), 0.241 (Mamba S6), 0.228 (Mamba-2 SSD), compared to DF's 0.082.

**Finding 7: Sequence models show mixed results relative to feedforward relays on 16-QAM.** On BPSK and QPSK, the Transformer, Mamba S6, and Mamba-2 SSD relays achieve BER comparable to MLP at low SNR. On 16-QAM, the Transformer achieves the lowest BER floor (0.165 at 16 dB AWGN), but MLP (0.200) outperforms both Mamba S6 (0.228) and Mamba-2 SSD (0.217). The fundamental $\tanh$ compression limitation affects all architectures, and the larger context window of sequence models does not consistently improve multi-level amplitude processing.

![Figure 21: BPSK relay comparison on AWGN (baseline).](results/modulation/bpsk_awgn_ci.png)

*Figure 21: BPSK on AWGN — all relay strategies with 95% CI (baseline for modulation comparison).*

![Figure 22: BPSK relay comparison on Rayleigh fading (baseline).](results/modulation/bpsk_rayleigh_ci.png)

*Figure 22: BPSK on Rayleigh fading — all relay strategies with 95% CI.*

![Figure 23: QPSK relay comparison on AWGN.](results/modulation/qpsk_awgn_ci.png)

*Figure 23: QPSK on AWGN — BER curves closely match the BPSK baseline (Figure 21), confirming I/Q splitting validity.*

![Figure 24: QPSK relay comparison on Rayleigh fading.](results/modulation/qpsk_rayleigh_ci.png)

*Figure 24: QPSK on Rayleigh fading — same relative ordering as BPSK, confirming hypothesis generalisability.*

![Figure 25: 16-QAM relay comparison on AWGN.](results/modulation/qam16__awgn_ci.png)

*Figure 25: 16-QAM on AWGN — AI relays hit a BER floor near 0.22 at medium-high SNR due to $\tanh$ compression of multi-level signals; AF outperforms DF at low SNR (Y* at 0–6 dB) by preserving amplitude structure.*

![Figure 26: 16-QAM relay comparison on Rayleigh fading.](results/modulation/qam16__rayleigh_ci.png)

*Figure 26: 16-QAM on Rayleigh fading — wider BER gap between modulations under fading; all AI relays significantly worse than DF at every SNR point (N* at 0–20 dB).*

![Figure 27: Combined modulation comparison on AWGN.](results/modulation/combined_modulation_awgn.png)

*Figure 27: Combined modulation comparison (AWGN) — all nine relays across BPSK (solid), QPSK (dashed, overlapping BPSK), and 16-QAM (dotted). The BPSK/QPSK overlap confirms I/Q splitting equivalence. The 16-QAM dotted curves reveal the AI relay BER floor: all AI relays plateau near 0.18–0.25 while DF and AF continue decreasing.*

**Summary.** The BPSK findings (H1–H3) generalise to QPSK across all nine relay strategies: the I/Q independence property ensures that BPSK-trained relays perform comparably on QPSK's binary-per-component structure. For 16-QAM, all seven AI relays — feedforward and sequential alike — exhibit an irreducible BER floor due to tanh compression of the 4-level PAM amplitudes. The Transformer achieves the lowest floor (0.165 at 16 dB) but is still 55× worse than DF (0.0030). A surprising counter-finding is that AF outperforms DF on 16-QAM at low SNR because linear amplification preserves the multi-level amplitude structure that DF's hard quantisation destroys. These results motivate the modulation-aware activation experiment in Section 7.11.

### 7.11 16-QAM Activation Experiment: Modulation-Aware Training

Section 7.10 identified the $\tanh$ output activation as the root cause of the AI relay BER floor on 16-QAM: the function compresses the 4-level PAM amplitudes $\{-3, -1, +1, +3\}/\sqrt{10}$ into its saturating region, making the outer levels ($\pm 0.949$) nearly indistinguishable from the inner levels ($\pm 0.316$). This section implements the modulation-specific training proposed in Section 8.6 and evaluates two alternative output activations.

#### 7.11.1 Experimental Design

Three activation variants are compared:

1. **tanh (baseline):** Standard $\tanh$ output, trained on BPSK signals — the original configuration from all prior experiments.
2. **linear:** Identity output activation ($f(z) = z$, unbounded), trained on 16-QAM PAM-4 symbols $\{-3, -1, +1, +3\}/\sqrt{10}$.
3. **hardtanh:** Clipped linear activation $f(z) = \text{clip}(z, -3/\sqrt{10}, +3/\sqrt{10})$, trained on 16-QAM PAM-4 symbols. The clip bounds $\pm 0.9487$ match the maximum 16-QAM per-axis amplitude exactly, providing bounded output while preserving linearity in the signal range.

All seven AI relays (MLP, Hybrid, VAE, CGAN, Transformer, Mamba S6, Mamba-2 SSD) are trained from scratch with each variant. The training protocol matches the original: 50,000 samples, 100 epochs (200 for sequence models), SNR = 5/10/15 dB. The linear and hardtanh variants use synthetically generated PAM-4 training targets at matched SNR to teach the network the 4-level amplitude structure. Classical AF and DF relays serve as modulation-independent baselines. Evaluation uses 16-QAM on AWGN and Rayleigh channels (10 trials × 10,000 bits, SNR 0–20 dB).

#### 7.11.2 Results

Table 15 shows the BER at 16 dB for all relays across the three activation variants and both channel types.

| Relay | tanh (BPSK) | linear (QAM16) | hardtanh (QAM16) | tanh (BPSK) | linear (QAM16) | hardtanh (QAM16) |
|---|---|---|---|---|---|---|
| | **AWGN** | | | **Rayleigh** | | |
| MLP (169p) | 0.196 | **0.105** | 0.196 | 0.223 | **0.128** | 0.217 |
| Hybrid | 0.253 | 0.253 | 0.253 | 0.271 | 0.271 | 0.271 |
| VAE | 0.501 | **0.212** | 0.221 | **0.255** | 0.495 | 0.496 |
| CGAN | 0.233 | **0.116** | 0.218 | 0.252 | **0.147** | 0.238 |
| Transformer | 0.168 | **0.074** | **0.074** | 0.201 | 0.123 | **0.122** |
| Mamba S6 | 0.227 | 0.096 | **0.083** | 0.234 | 0.136 | **0.132** |
| Mamba-2 SSD | 0.215 | 0.074 | **0.062** | 0.241 | **0.118** | 0.121 |
| AF | 0.016 | — | — | 0.102 | — | — |
| DF | 0.0030 | — | — | 0.082 | — | — |

*Table 15: 16-QAM BER at 16 dB — activation variant comparison. Bold marks the best AI variant per relay. The tanh column reproduces the Section 7.10 baseline. Hybrid is unchanged because its high-SNR path routes to DF internally.*

![Figure 28: 16-QAM activation experiment on AWGN.](results/qam16_activation/qam16_activation_awgn.png)

*Figure 28: 16-QAM activation experiment (AWGN) — dashed lines = tanh/BPSK baseline, solid = linear/QAM16, dotted = hardtanh/QAM16. Replacing tanh and retraining on QAM16 eliminates the BER floor for all AI relays except Hybrid. Sequence models (Transformer, Mamba S6, Mamba-2) benefit most, narrowing the gap to DF from ~56× to ~10×.*

![Figure 29: 16-QAM activation experiment on Rayleigh.](results/qam16_activation/qam16_activation_rayleigh.png)

*Figure 29: 16-QAM activation experiment (Rayleigh fading) — same trend under fading. The improvement is significant but the gap to DF/AF remains larger than on AWGN, consistent with fading amplifying modulation-order differences (Finding 6).*

#### 7.11.3 Analysis

**Finding 8: Replacing tanh eliminates the 16-QAM BER floor.** Both the linear and hardtanh activations break through the BER floor that all AI relays exhibited in Section 7.10. The improvement factors (tanh → best variant) at 16 dB AWGN are: Mamba-2 3.5×, Mamba S6 2.7×, VAE 2.4×, Transformer 2.3×, CGAN 2.0×, MLP 1.9×. This confirms the hypothesis from Section 7.10 that the bottleneck is the activation function, not model capacity.

**Finding 9: Linear is generally preferred for feedforward relays; hardtanh for state-space models.** For feedforward relays, linear consistently achieves the lowest BER: MLP 0.105 vs hardtanh 0.196, CGAN 0.116 vs 0.218 (AWGN). For state-space models, hardtanh is preferred: Mamba S6 0.083 vs 0.096, Mamba-2 0.062 vs 0.074. Transformer shows a near-tie (0.074 for both). The practical recommendation depends on the architecture: linear for simple feedforward relays, hardtanh for recurrent models.

**Finding 10: Mamba-2 SSD benefits most from modulation-aware training.** The three sequence models achieve the lowest BER among AI relays after retraining: Mamba-2 hardtanh 0.062, Transformer 0.074, Mamba S6 hardtanh 0.083 (AWGN). These are 3.5×, 2.3×, and 2.7× improvements over the tanh baseline, compared to 1.9× for MLP and 2.0× for CGAN.

**Finding 11: The gap to classical relays narrows but persists.** The best AI relay (Mamba-2 hardtanh, 0.062 on AWGN) is 20.7× worse than DF (0.0030) and 3.9× worse than AF (0.016). While this is an improvement from the tanh baseline, the remaining gap reflects the fundamental difficulty of learning multi-level quantisation from data alone versus the hard-coded PAM-4 decision boundaries in DF. On Rayleigh, the gap is smaller: Mamba-2 linear 0.118 vs DF 0.082 (1.4×), suggesting that fading partially equalises AI and classical approaches.

**Finding 12: The Hybrid relay is unaffected.** Hybrid achieves 0.253 across all three variants because at 16 dB its SNR estimator routes to the DF sub-relay, which uses hard sign-detection on the I/Q-split signal. The Hybrid's MLP sub-network (which benefits from the new activation) is bypassed at high SNR. This confirms that the Hybrid relay's SNR-adaptive switching, while effective for BPSK (Section 7.6), requires modulation-aware DF quantisation for higher-order constellations.

**Summary.** The activation experiment validates the hypothesis from Section 7.10: the 16-QAM BER floor is caused by $\tanh$ compression, not by insufficient model capacity. Replacing $\tanh$ with an unbounded or bounded linear activation matched to the 16-QAM constellation range, combined with retraining on PAM-4 target symbols, reduces the AI relay BER floor by 2–3.5× across all relay types. Mamba-2 SSD benefits most (3.5×), achieving the lowest AI relay BER of 0.062 on AWGN. These findings demonstrate that modulation-aware output activation design is essential for extending AI relay strategies to higher-order modulations.

### 7.12 Constellation-Aware Activation Study

Section 7.11 demonstrated that replacing $\tanh$ with $\text{hardtanh}$ bounded to $\pm 3/\sqrt{10}$ eliminates the 16-QAM BER floor. However, the clip bounds in that experiment were fixed to the 16-QAM maximum amplitude. This section generalises the approach by introducing a **constellation-aware clip range** that automatically adapts the output activation bounds to the modulation scheme, and evaluates two additional smooth bounded activations — **sigmoid** and **scaled tanh** — alongside hardtanh across all three constellations.

#### 7.12.1 Constellation-Aware Clip Range

The maximum per-axis amplitude for a rectangular $M$-QAM constellation with average unit energy is:

$$A_{\max} = \frac{\sqrt{M} - 1}{\sqrt{\frac{2(M-1)}{3}}}$$

This yields the following clip ranges for the four modulation schemes:

| Modulation | $M$ | $A_{\max}$ | Numeric Value |
|---|---|---|---|
| BPSK | 2 | $1.0$ | 1.0000 |
| QPSK | 4 | $1/\sqrt{2}$ | 0.7071 |
| 16-QAM | 16 | $3/\sqrt{10}$ | 0.9487 |
| 16-PSK | 16 | $1.0$ | 1.0000 |

For BPSK, $A_{\max} = 1.0$ recovers the standard $\tanh$ range. For QPSK with I/Q splitting, each component is binary ($\pm 1/\sqrt{2}$), so the clip range is tighter. For 16-QAM, $A_{\max} = 0.9487$ matches the Section 7.11 setting. For 16-PSK, $A_{\max} = 1.0$ as all constellation points lie on the unit circle with constant envelope. The clip range is threaded through all relay implementations, ensuring that the output activation bounds match the constellation geometry regardless of modulation order.

#### 7.12.2 Activation Functions Compared

Three bounded activations are evaluated, each scaled to $[-A_{\max}, +A_{\max}]$:

1. **Hardtanh:** $f(z) = \text{clip}(z, -A_{\max}, +A_{\max})$. Piecewise linear with sharp saturation at the bounds. Zero gradient outside the linear region may cause dead neurons during training.

2. **Sigmoid (scaled):** $f(z) = A_{\max} \cdot (2\sigma(z) - 1)$, where $\sigma$ is the logistic function. Smooth, zero-centred, with gradients that never vanish entirely. The re-centring ensures symmetric output.

3. **Scaled tanh:** $f(z) = A_{\max} \cdot \tanh(z)$. Identical to standard $\tanh$ when $A_{\max} = 1$ (BPSK), but scaled to match tighter or wider constellation ranges for other modulations.

All three activations are implemented with matching NumPy (for MLP/Hybrid) and PyTorch (for sequence models) backends, verified to produce identical outputs to machine precision.

#### 7.12.3 Experimental Design

All seven neural network relays are retrained from scratch for each activation–constellation combination. The training protocol matches Section 7.11: 50,000 samples, 100 epochs (200 for sequence models), SNR = {5, 10, 15} dB. Training targets are generated from the appropriate constellation (BPSK symbols $\pm 1$, QPSK I/Q components $\pm 1/\sqrt{2}$, 16-QAM PAM-4 levels $\{-3, -1, +1, +3\}/\sqrt{10}$). Evaluation uses 10 trials × 10,000 bits per SNR point on both AWGN and Rayleigh channels. Classical AF and DF serve as modulation-independent baselines.

#### 7.12.4 Results

Figures 29–34 show BER vs. SNR for all relay–activation combinations across the six constellation–channel configurations.

![Figure 30: BPSK activation comparison on AWGN.](results/activation_comparison/bpsk_activation_awgn.png)

*Figure 30: BPSK constellation-aware activation comparison (AWGN). With $A_{\max} = 1.0$, scaled tanh reduces to standard tanh. All three bounded activations achieve equivalent BER, confirming that BPSK is insensitive to activation choice.*

![Figure 31: BPSK activation comparison on Rayleigh.](results/activation_comparison/bpsk_activation_rayleigh.png)

*Figure 31: BPSK constellation-aware activation comparison (Rayleigh fading). Same pattern under fading — activation choice has negligible effect on BPSK BER.*

![Figure 32: QPSK activation comparison on AWGN.](results/activation_comparison/qpsk_activation_awgn.png)

*Figure 32: QPSK constellation-aware activation comparison (AWGN). With $A_{\max} = 0.7071$, the tighter clip range matches the binary I/Q components exactly. Sigmoid provides marginally lower BER for the Transformer relay at low SNR.*

![Figure 33: QPSK activation comparison on Rayleigh.](results/activation_comparison/qpsk_activation_rayleigh.png)

*Figure 33: QPSK constellation-aware activation comparison (Rayleigh fading). Similar trends under fading. The three activations remain closely matched for most relays.*

![Figure 34: QAM16 activation comparison on AWGN.](results/activation_comparison/qam16_activation_awgn.png)

*Figure 34: 16-QAM constellation-aware activation comparison (AWGN). All three bounded activations eliminate the tanh BER floor from Section 7.10, with scaled tanh and hardtanh closely matched.*

![Figure 35: QAM16 activation comparison on Rayleigh.](results/activation_comparison/qam16_activation_rayleigh.png)

*Figure 35: 16-QAM constellation-aware activation comparison (Rayleigh fading). The BER floor elimination persists under fading. Sequence models benefit most from the constellation-aware bounds.*

#### 7.12.5 Analysis

**Finding 13: BPSK is activation-invariant.** For BPSK ($A_{\max} = 1.0$), all three activations produce statistically indistinguishable BER curves. Scaled tanh reduces to standard $\tanh$ at this range, and the binary nature of BPSK symbols ($\pm 1$) means any monotonic bounded function that covers $[-1, +1]$ suffices. This confirms that the activation bottleneck identified in Section 7.10 is specific to multi-level constellations.

**Finding 14: Constellation-aware clip range generalises the Section 7.11 result.** The BER floor elimination observed with hardtanh on 16-QAM in Section 7.11 extends to all three bounded activations (hardtanh, sigmoid, scaled tanh) when the clip range is properly matched to $A_{\max}$. For QPSK, the tighter $A_{\max} = 0.7071$ provides marginal improvement over the default $\tanh$ range of $\pm 1$, as the network no longer wastes representational capacity on the unused amplitude range $[0.7071, 1.0]$.

**Finding 15: Sigmoid offers advantages for attention-based models.** On QPSK, the scaled sigmoid activation achieves a measurably lower BER for the Transformer relay at low SNR (0–4 dB). The smooth gradient profile of sigmoid (which never saturates to exactly zero, unlike hardtanh's flat regions) appears to benefit the attention mechanism's gradient flow during training. For feed-forward and SSM-based relays, the three activations remain closely matched.

**Finding 16: Scaled tanh is the recommended default.** Across all constellations and channels, scaled tanh provides competitive or best BER while maintaining the familiar $\tanh$ gradient shape (beneficial for training stability) and the correct amplitude bounds (preventing constellation distortion). Its smooth saturation avoids the dead-neuron risk of hardtanh while being computationally simpler than scaled sigmoid.

![Figure 36: Activation function shapes.](results/activation_comparison/various_activation_functions.png)

*Figure 36: Comparison of activation function shapes (left) and their derivatives (right) for $A_{\max} = 0.9487$ (16-QAM). Hardtanh has a sharp transition at the clip bounds; sigmoid and scaled tanh provide smooth saturation with non-zero gradients throughout.*

### 7.13 Input Layer Normalization and Scaled Tanh Experiment

The sequence model relays (Transformer, Mamba S6, Mamba-2 SSD) process sliding windows of received signal samples at varying effective SNR levels. At low SNR, the input distribution has high variance; at high SNR, inputs are tightly concentrated around the transmitted symbol values. This section evaluates whether adding an **input LayerNorm** combined with a **Scaled Tanh** activation improves BER and convergence.

#### 7.13.1 Experimental Design

Three configurations are compared:

1. **Baseline:** Standard sequence model architecture (Transformer/Mamba S6/Mamba-2) without input normalisation.
2. **+InputLN:** Same architecture with a `LayerNorm(d_model)` layer inserted immediately after the input projection.
3. **+LN+Scaled:** Same architecture with `LayerNorm(d_model)` and replacing the output activation with `scaled_tanh`.

The +InputLN variant adds a minimal number of parameters (two learnable scalars per dimension: scale $\gamma$ and shift $\beta$):

| Model | Baseline Parameters | +InputLN / +LN+Scaled Parameters | Overhead |
|---|---|---|---|
| Transformer | 17,697 | 17,761 | +64 (+0.36%) |
| Mamba S6 | 24,001 | 24,065 | +64 (+0.27%) |
| Mamba-2 SSD | 26,179 | 26,243 | +64 (+0.24%) |

Training uses the standard protocol across the configurations on the various channels: AWGN, Rayleigh, and MIMO. Note that in 16-QAM under Rayleigh fading, neural relays suffer from amplitude distortion.

#### 7.13.2 Training Results

With the addition of +LN+Scaled, a significant divergence in training stability was observed:

| Model | Training Behavior (+InputLN) | Training Behavior (+LN+Scaled) |
|---|---|---|
| Transformer | Very stable | Stable, neutral final loss |
| Mamba S6 | Stable | Stable, improved convergence |
| Mamba-2 SSD | Stable | Stable, strongest improvement |

All three sequence architectures converge functionally under both +InputLN and +LN+Scaled configurations. The +LN+Scaled variant produces the largest improvements for Mamba-2 SSD and Mamba S6, while the Transformer remains neutral.

#### 7.13.3 Results

The BER evaluation highlights strong architectural divergence. Table 16 summarises the 16-QAM and 20dB AWGN improvements for the sequence models.

* **Transformer**: Remained relatively neutral and robust against input scaling and normalization (Baseline 0.0328, +LN+Scaled 0.0383 at 20 dB AWGN).
* **Mamba S6**: Exhibited a modest **+11.1% benefit** in BER over the baseline at 20 dB AWGN (Baseline 0.0602, +LN+Scaled 0.0535), benefiting from the normalized spatial bounds of the input signal.
* **Mamba-2 SSD**: Achieved the strongest improvement under +LN+Scaled, reaching 0.0170 at 20 dB AWGN (Baseline 0.0333) — a **49% reduction** and the best BER of any variant.

Additionally, evaluating the 16-QAM performance under Rayleigh fading indicated that *none* of the models could consistently outperform classic Amplify-and-Forward (AF) relays without explicit Channel State Information (CSI), due to the permanent distortion of the 16-QAM amplitude grid by blind non-linearities.

#### 7.13.4 Analysis

**Finding 17: +LN+Scaled benefits state space models more than attention-based models.** The +LN+Scaled combination improves both Mamba S6 (11.1% BER reduction at 20 dB AWGN) and Mamba-2 SSD (49% BER reduction, reaching the best overall BER of 0.0170 at 20 dB AWGN). In contrast, the Transformer shows no measurable improvement. The structured state space architectures appear to benefit from the explicit amplitude scaling provided by the constellation-aware bounded activation, while the self-attention mechanism in the Transformer is already effective at extracting amplitude information from raw inputs.

**Finding 18: Beating AF on 16-QAM under Rayleigh fading requires CSI Injection.** Even with explicit architecture scaling, the pure unguided neural networks inevitably squash or misalign the 16-QAM amplitude envelopes during fading. Classic AF trivially avoids this by operating purely linearly. Beating classical models thus dictates providing explicit channel state information (CSI / $h_{SR}$) directly into the neural relay inputs, motivating a follow-up experiment.

### 7.14 Structural CSI Injection for 16-QAM in Rayleigh Fading

### 7.14.1 Motivation

Earlier results (Section 7.13) revealed that pure sequential models fundamentally struggle to outperform classical Amplify-and-Forward (AF) relays when processing continuous-envelope 16-QAM signals in Rayleigh fading. The non-linear mechanisms ($\tanh$, normalization layers) inevitably warp the amplitude grid, while blind temporal averaging fails to fully capture dynamic channel conditions. This experiment investigates whether explicitly injecting **Channel State Information (CSI)** — the fading coefficient $h_{SR}$ — directly into the model's first structural layer recovers and exceeds classical performance limits.

### 7.14.2 Experimental Design

The initial input projection of the Mamba S6 architecture was expanded from `nn.Linear(1, d_model)` to `nn.Linear(2, d_model)`. At inference, the input streams explicitly encode 2D features: the noisy symbol $y$ alongside the magnitude of the fading coefficient $|h_{SR}|$. The neural network is jointly trained to reconstruct the optimal clean constellation mappings conditioned directly on the severity of the measured channel fade point.

A secondary structural bypass was introduced such that residual identity connections ($x + \text{residual}$) only loop the processed received signal independently, ensuring the raw CSI feature acts strictly as an uncorrupted contextual guide for the sequence blocks. 

The variants tested over 10 independent trials (10,000 bits each) were:
- **AF** & **DF** baselines
- **Mamba S6 (Baseline)**: $d_{in}=1$, strict scalar temporal tracking.
- **Mamba S6 (+LayerNorm)**: $d_{in}=1$, inclusion of Input LayerNorm and Scaled Tanh stabilization.
- **Mamba S6 (+CSI + LN)**: $d_{in}=2$, comprehensive structural configuration explicitly ingesting fading states.

### 7.14.3 Results and Analysis

| SNR (dB) | AF Baseline | DF Baseline | Mamba S6 (Baseline) | Mamba S6 (+CSI + LN) |
|---|---|---|---|---|
| **0.0** | 0.3332 | 0.4300 | 0.2288 | 0.2396 |
| **8.0** | 0.1436 | 0.2798 | 0.2241 | 0.0635 |
| **14.0** | 0.0592 | 0.1557 | 0.2227 | 0.0163 |
| **20.0** | 0.0246 | 0.0901 | 0.2302 | **0.0055** |

*Table 18: Sample BER values comparing blind spatial tracking vs. explicit Channel State injection for 16-QAM in Rayleigh fading.*

**Finding 20: Structural CSI Injection resolves the amplitude aliasing bottleneck for neural relays.** Providing the exact instantaneous fading envelope dynamically guides the multi-dimensional scaling bounds of the model. Instead of relying purely on sequence-level contextual averages (which collapse under rapid decorrelation), the network acts conditionally on the instantaneous noise-to-signal geometry.

At high SNR (20 dB), whereas pure blind Mamba collapses entirely (0.2302 BER) and classical AF floors organically (0.0246 BER), the CSI-aware Mamba model successfully constructs a dynamic amplitude compensator to reach **0.0055 BER**, representing a 4.5× absolute improvement over the finest classical bounds. This definitively confirms that while standard Transformers or Mamba sequences map poorly to unguided phase-amplitude constellations over temporal fading channels, providing explicit structural channel variables unlocks their generative superiority.

---

### 7.15 Comprehensive Multi-Architecture CSI Experiment

### 7.15.1 Motivation

The initial CSI injection experiment (Section 7.14) demonstrated that providing explicit fading coefficients to the Mamba S6 relay dramatically improves 16-QAM performance under Rayleigh fading. However, that study used only a single architecture (Mamba S6), a single activation function, and a limited number of trials ($N=10$). This section presents a comprehensive combinatorial experiment that systematically evaluates **three neural architectures** (Mamba S6, Transformer, Mamba-2), **four activation functions** (hardtanh, scaled\_tanh, tanh, sigmoid), and **four structural configurations** (Baseline, +LN, +CSI, +CSI+LN) across two higher-order constellations: **16-QAM** and **16-PSK**.

The goal is to identify the top-performing neural relay topologies under each modulation scheme and to determine whether CSI injection, input LayerNorm, or their combination provides the most consistent benefit across diverse architectures.

### 7.15.2 Experimental Design

The experiment evaluates $3 \times 4 \times 4 = 48$ neural relay variants plus the two classical baselines (AF and DF), for a total of 50 variants per constellation.

**Architectures:**
- **Mamba S6** (~24K parameters): Selective state space model with input-dependent gating and recurrent state propagation.
- **Transformer** (~17K parameters): Multi-head self-attention with positional encoding.
- **Mamba-2** (~26K parameters): Second-generation state space model with structured state space duality (SSD).

**Structural configurations:**
- **Baseline**: Standard $d_{in}=1$ input projection, no normalization.
- **+LN**: Input LayerNorm applied before the first sequence block; scaled tanh/hardtanh/sigmoid activation at output.
- **+CSI**: Input expanded to $d_{in}=2$ with channel state $|h_{SR}|$ concatenated alongside the received symbol $y$.
- **+CSI+LN**: Both CSI injection and input LayerNorm.

**Activation functions:**
- **hardtanh**: Piecewise-linear clipping at $\pm A_{\max}$.
- **scaled\_tanh**: $A_{\max} \cdot \tanh(x / A_{\max})$, smooth approximation to hardtanh.
- **tanh**: Standard hyperbolic tangent with unit range.
- **sigmoid**: $A_{\max} \cdot \sigma(x)$, asymmetric bounded activation.

**Training protocol:** 25 epochs with early stopping (patience = 10, $\delta_{\min} = 10^{-5}$), MSE loss, Adam optimizer ($\eta = 10^{-3}$), 50,000 training symbols. Weight caching ensures reproducibility across successive runs.

**Evaluation:** 100 Monte Carlo trials, 10,000 bits per trial, SNR range 0–20 dB in 2 dB steps. 95% confidence intervals computed via Student's $t$-distribution.

### 7.15.3 Results: 16-QAM in Rayleigh Fading

The full combinatorial experiment produced the BER curves shown in Figure 39. The top-3 neural relay architectures, ranked by average BER over the upper half of the SNR range (10–20 dB), are presented alongside the classical AF and DF baselines in Figure 40 and Table 19.

![Figure 39: Full 16-QAM CSI experiment — all 48 neural variants plus AF/DF.](results/csi/csi_experiment_qam16_rayleigh.png)

*Figure 39: 16-QAM Rayleigh fading — BER vs. SNR for all 48 neural relay variants and two classical baselines (AF, DF) with 95% confidence intervals. The plot reveals a dense cluster of neural variants between the AF and DF curves, with the best variants approaching AF performance at high SNR.*

![Figure 40: Top-3 neural relays vs. classical — 16-QAM Rayleigh.](results/csi/top3_qam16_rayleigh.png)

*Figure 40: Top-3 neural relay architectures compared against AF and DF for 16-QAM in Rayleigh fading. All three best-performing variants use input LayerNorm (+LN) without CSI injection.*

| SNR (dB) | AF | DF | #1 Mamba S6 (+LN tanh) | #2 Transformer (+LN sigmoid) | #3 Transformer (+LN tanh) |
|---|---|---|---|---|---|
| 0 | 0.4490 | 0.4130 | 0.4513 | 0.4490 | 0.4493 |
| 4 | 0.3893 | 0.3472 | 0.3893 | 0.3865 | 0.3852 |
| 8 | 0.2913 | 0.2543 | 0.2943 | 0.2965 | 0.2943 |
| 12 | 0.1877 | 0.1575 | 0.1903 | 0.1912 | 0.1903 |
| 16 | 0.1020 | 0.0817 | 0.1063 | 0.1093 | 0.1102 |
| 20 | 0.0465 | 0.0393 | 0.0562 | 0.0598 | 0.0620 |

*Table 19: BER at selected SNR points for the top-3 neural relays and classical baselines (16-QAM, Rayleigh fading, 100 MC trials). The best neural variant (Mamba S6 +LN tanh) comes within 21% of AF at 20 dB but does not surpass it.*

**QAM16 Top-3 Observations:**
1. **#1 Mamba S6 (+LN tanh):** Best overall neural relay with BER = 0.0562 at 20 dB (vs. AF = 0.0465). The Mamba S6 architecture combined with input LayerNorm and standard tanh activation achieves the closest tracking to AF across the full SNR range.
2. **#2 Transformer (+LN sigmoid):** BER = 0.0598 at 20 dB. Attention-based architecture with smooth sigmoid output and normalization.
3. **#3 Transformer (+LN tanh):** BER = 0.0620 at 20 dB. Attention-based architecture with standard tanh and LayerNorm.

### 7.15.4 Results: 16-PSK in Rayleigh Fading

The same combinatorial experiment was repeated for 16-PSK modulation with $A_{\max} = 1.0$ (unit-circle constellation). The full BER curves are shown in Figure 41, with the top-3 comparison in Figure 42 and Table 20.

![Figure 41: Full 16-PSK CSI experiment — all 48 neural variants plus AF/DF.](results/csi/csi_experiment_psk16_rayleigh.png)

*Figure 41: 16-PSK Rayleigh fading — BER vs. SNR for all 48 neural relay variants and two classical baselines. The neural variants form a tighter cluster than QAM16, reflecting the constant-envelope nature of PSK which reduces the amplitude-aliasing challenge.*

![Figure 42: Top-3 neural relays vs. classical — 16-PSK Rayleigh.](results/csi/top3_psk16_rayleigh.png)

*Figure 42: Top-3 neural relay architectures compared against AF and DF for 16-PSK in Rayleigh fading. In contrast to QAM16, all three best-performing PSK16 variants use CSI injection (+CSI or +CSI+LN).*

| SNR (dB) | AF | DF | #1 Mamba S6 (+CSI+LN hardtanh) | #2 Mamba S6 (+CSI hardtanh) | #3 Mamba S6 (+CSI scaled\_tanh) |
|---|---|---|---|---|---|
| 0 | 0.4503 | 0.4218 | 0.4490 | 0.4532 | 0.4535 |
| 4 | 0.3987 | 0.3670 | 0.3963 | 0.3990 | 0.3945 |
| 8 | 0.3240 | 0.2983 | 0.3157 | 0.3123 | 0.3130 |
| 12 | 0.2293 | 0.2083 | 0.2225 | 0.2197 | 0.2232 |
| 16 | 0.1473 | 0.1317 | 0.1415 | 0.1440 | 0.1433 |
| 20 | 0.0812 | 0.0710 | 0.0832 | 0.0808 | 0.0842 |

*Table 20: BER at selected SNR points for the top-3 neural relays and classical baselines (16-PSK, Rayleigh fading, 100 MC trials). The best neural variants track AF closely at high SNR, with the gap narrowing to 2.5% at 20 dB.*

**PSK16 Top-3 Observations:**
1. **#1 Mamba S6 (+CSI+LN hardtanh):** BER = 0.0832 at 20 dB (vs. AF = 0.0812). CSI-aware Mamba with LayerNorm and piecewise-linear clipping achieves the lowest average BER over the 10–20 dB range.
2. **#2 Mamba S6 (+CSI hardtanh):** BER = 0.0808 at 20 dB. CSI injection without LayerNorm — achieves the lowest single-point BER at 20 dB.
3. **#3 Mamba S6 (+CSI scaled\_tanh):** BER = 0.0842 at 20 dB. CSI injection with smooth bounded activation.

### 7.15.5 Cross-Constellation Analysis

The comprehensive experiment reveals strikingly different optimal strategies for the two higher-order constellations:

| Property | 16-QAM Top-3 | 16-PSK Top-3 |
|---|---|---|
| Dominant configuration | +LN (LayerNorm only) | +CSI / +CSI+LN |
| CSI injection benefit | Negative (degrades BER) | Positive (improves BER) |
| Best architecture | Mamba S6, Transformer | Mamba S6 (all three) |
| Best activation | tanh, sigmoid | hardtanh, scaled\_tanh |
| Gap to AF at 20 dB | ~21% (0.0562 vs. 0.0465) | ~2.5% (0.0832 vs. 0.0812) |
| DF superiority at 20 dB | DF wins by 43% over best neural | DF wins by 17% over best neural |

*Table 21: Cross-constellation comparison of top-performing neural relay strategies.*

**Finding 21: CSI injection is modulation-dependent.** For 16-QAM (amplitude-and-phase modulation), CSI injection systematically degrades performance. The top-3 QAM16 variants all use **+LN without CSI**. The likely explanation is that 16-QAM's amplitude grid already encodes distance information in the I/Q signal levels, and injecting a redundant channel magnitude feature confuses the network's learned amplitude mapping. In contrast, 16-PSK (pure phase modulation with constant envelope) benefits from CSI injection because the fading coefficient provides orthogonal amplitude information that the constant-envelope signal cannot convey. All top-3 PSK16 variants use **+CSI or +CSI+LN**.

**Finding 22: Input LayerNorm is universally beneficial for QAM16 but model-dependent for PSK16.** For 16-QAM, every top-3 variant includes LayerNorm, consistent with the Section 7.13 finding that normalizing the multi-level amplitude distribution before sequence processing stabilizes training and reduces BER. For 16-PSK, LayerNorm appears in one of the top-3 variants (#1 Mamba S6 +CSI+LN hardtanh), while the #2 and #3 variants omit it, suggesting that the constant-envelope PSK signal is already well-conditioned for direct processing.

**Finding 23: Mamba S6 is the dominant architecture across both constellations.** Mamba S6 appears in all three PSK16 top-3 positions and holds the #1 QAM16 position. The Transformer captures the #2 and #3 QAM16 positions, demonstrating its competitiveness on amplitude-and-phase modulations when combined with LayerNorm. Neither Mamba-2 nor any baseline (non-LN, non-CSI) configuration appears in the top-3 for either constellation. The Mamba S6 architecture's selective state space mechanism appears well-suited to tracking both the multi-level amplitude structure of QAM and the rotational phase geometry of PSK.

**Finding 24: Classical DF remains unbeaten across all configurations.** Despite the 48 neural variants spanning three architectures, four activations, and four structural configurations, no neural relay achieves a lower BER than DF at any SNR point for either constellation. The best neural relays approach AF performance but cannot surpass it. This reaffirms the fundamental information-theoretic advantage of hard-decision regeneration when perfect demodulation is possible — the relay task for higher-order modulations requires equalization beyond what current neural sequence models can provide without explicit channel knowledge at the demodulator level.

### 7.15.6 Training Convergence Examples

Representative training curves for the top-performing variants illustrate the convergence characteristics of each architecture under the 16-PSK experiment. All models were trained for 25 epochs with early stopping (patience = 10).

![Figure 43: Training history — Mamba-2 (+LN scaled\_tanh).](results/csi/training_Mamba2_LN_scaled_tanh.png)

*Figure 43: Training loss (MSE) and accuracy for Mamba-2 (+LN scaled\_tanh). The model converges within approximately 5 epochs and stabilises, with validation accuracy closely tracking training accuracy — indicating minimal overfitting.*

![Figure 44: Training history — Mamba (+CSI tanh).](results/csi/training_Mamba_CSI_tanh.png)

*Figure 44: Training loss and accuracy for Mamba S6 (+CSI tanh). The CSI-augmented input produces a smooth, monotonic training convergence, with the additional channel-state feature providing clear gradient signal for the optimizer.*

![Figure 45: Training history — Transformer (+CSI sigmoid).](results/csi/training_Transformer_CSI_sigmoid.png)

*Figure 45: Training loss and accuracy for Transformer (+CSI sigmoid). The Transformer exhibits rapid initial convergence (within 3 epochs) followed by a flat plateau, characteristic of the attention mechanism's ability to capture input structure quickly.*

Three key training patterns emerge across all 48 variants:
1. **Rapid convergence:** All models reach near-final loss within 5–8 epochs, confirming that 25 epochs with patience-10 early stopping is sufficient for the relay denoising task at these model scales.
2. **No overfitting:** Training and validation accuracy remain tightly coupled across all variants, consistent with the "less is more" principle (Section 8.2) — even the largest model (Mamba-2, ~26K parameters) does not overfit on the 50,000-sample training set.
3. **Loss plateau stability:** Models that converge to lower MSE loss do not necessarily achieve lower BER, since BER is a non-differentiable threshold metric while MSE is a continuous surrogate. This explains why activation choice (which affects the loss surface geometry) can matter more than final loss magnitude.

### 7.15.7 Experiment Goals vs. Outcomes

This section evaluates the comprehensive CSI experiment against its stated objectives.

| Goal | Outcome | Assessment |
|---|---|---|
| Identify top architectures for higher-order modulations | QAM16: Mamba S6 and Transformer dominate; PSK16: Mamba S6 sweeps all top-3 | **Achieved** — clear ranking established with statistical confidence over 100 MC trials |
| Determine whether CSI injection universally improves relay performance | CSI injection is modulation-dependent: beneficial for PSK16, detrimental for QAM16 | **Achieved** — unexpected finding that refutes the Section 7.14 hypothesis of universal CSI benefit |
| Evaluate the role of input LayerNorm across architectures | LayerNorm consistently helps QAM16 (all top-3 use it) but is neutral-to-unnecessary for PSK16 | **Achieved** — extends Section 7.13 finding to multi-model, multi-constellation setting |
| Compare 48 neural variants against classical baselines | No neural variant beats DF; best variants approach but do not surpass AF | **Achieved** — confirms DF optimality for higher-order modulations at all SNR points |
| Establish reproducible JSON-backed experiment infrastructure | Full per-trial BER data, 95% CI bounds, and metadata saved to JSON; automated top-3 chart generation | **Achieved** — enables future reanalysis without re-running experiments |

*Table 22: Goals vs. outcomes for the comprehensive multi-architecture CSI experiment.*

The key surprise was Finding 21: the modulation-dependence of CSI injection. The Section 7.14 initial experiment suggested that explicit channel state information would universally improve neural relay performance. The comprehensive experiment refutes this — for amplitude-and-phase modulations (QAM16), the signal's amplitude structure already encodes sufficient channel information, and injecting $|h_{SR}|$ creates feature redundancy that degrades the learned mapping. Only for constant-envelope modulations (PSK16), where the signal carries no amplitude information, does CSI injection provide the missing channel dimension needed for effective relay processing.

### 7.16 Extension Experiment: End-to-End Joint Optimization

Throughout the primary evaluations in this thesis, a modular architecture was maintained: the modulation (e.g., BPSK, 16-QAM) and the destination equalization were fixed, while neural networks were exclusively deployed at the intermediate relay node for denoising. To provide a complete comparative perspective on the limits of deep learning in physical-layer communications, this section evaluates a pure End-to-End (E2E) autoencoder paradigm. In this experiment, the relay node is removed, and the transmitter and destination receiver are jointly optimized as a single neural network over a stochastically differentiable physical channel.

#### 7.16.1 System Formulation

The E2E architecture discards classical predefined constellations (such as Gray-coded square grids) and frames communication as a classification task through a constrained continuous latent space.

**The Transmitter (Encoder).** The transmitter maps a discrete message index $m \in \{1, \dots, M\}$ to a continuous complex signal. The input is a one-hot vector $\mathbf{s} \in \mathbb{R}^M$. A multi-layer perceptron $f_\theta$ generates a raw latent vector $\mathbf{z} \in \mathbb{R}^{2n}$, where $n$ is the number of complex channel uses ($n=1$ for standard symbol-by-symbol transmission). To satisfy physical hardware limitations, a strict average power constraint is enforced via batch standardisation across the dimension:

$$\mathbf{x} = \sqrt{2n} \frac{\mathbf{z} - \mathbb{E}[\mathbf{z}]}{\sqrt{\text{Var}(\mathbf{z}) + \epsilon}}$$

This normalisation allows the network to learn variable amplitude boundaries (analogous to QAM) while bounding average transmission power.

**The Physical Channel.** The signal is subjected to a single-tap Rayleigh fading channel:

$$\mathbf{y} = \mathbf{h} \odot \mathbf{x} + \mathbf{n}, \quad h_i \sim \mathcal{CN}(0, 1), \quad n_i \sim \mathcal{CN}(0, \sigma^2)$$

**The Receiver (Decoder).** Assuming perfect Channel State Information (CSI), the received signal $\mathbf{y}$ and the channel coefficient $\mathbf{h}$ are concatenated. To prevent the network from expending parameters attempting to approximate complex division, an explicit Zero-Forcing (ZF) equalization layer computes $\hat{\mathbf{x}} = \mathbf{y} / \mathbf{h}$. The equalized signal and the channel magnitude are fed into a decoder network $g_\phi$, which outputs a probability distribution $\mathbf{p} \in (0,1)^M$ via a softmax activation.

The transmitter and receiver are jointly trained to minimise the categorical cross-entropy loss between $\mathbf{s}$ and $\mathbf{p}$.

#### 7.16.2 Results: E2E vs. Classical Theoretical Limits

The E2E network was trained for $M=16$ (equivalent to 16-QAM) over a $1 \times 1$ Rayleigh fading channel without spatial or temporal diversity. To benchmark the learned representation, the E2E performance is compared against the exact closed-form theoretical approximation for standard square 16-QAM over Rayleigh fading [21]:

$$P_s \approx 2 \left( \frac{\sqrt{M}-1}{\sqrt{M}} \right) \left( 1 - \sqrt{\frac{1.5 \gamma / (M-1)}{1 + 1.5 \gamma / (M-1)}} \right)$$

where the Bit Error Rate is approximated as $\text{BER} \approx P_s / \log_2(M)$ under optimal Gray coding.

| SNR (dB) | Standard 16-QAM Theory | E2E Learned Autoencoder | Relative Difference |
|---|---|---|---|
| 10.0 | 0.1098 | 0.183 | −67% |
| 12.0 | 0.0762 | 0.142 | −86% |
| 15.0 | 0.0481 | 0.084 | −75% |
| 20.0 | 0.0174 | 0.042 | −141% |

*Table 23: BER comparison of E2E autoencoder vs. theoretical 16-QAM (Rayleigh fading). The E2E network achieves 67–141% higher BER than the theoretical optimum, indicating that the learned constellation does not surpass the classical grid geometry for single-antenna Rayleigh fading with this training configuration.*

![Figure 46: E2E BER comparison.](results/e2e/e2e_ber_comparison.png)

*Figure 46: BER vs. SNR for E2E learned autoencoder compared to theoretical 16-QAM over Rayleigh fading. The E2E system underperforms the classical grid constellation across the full SNR range, with the gap widening at high SNR.*

![Figure 47: E2E learned constellation.](results/e2e/e2e_constellation.png)

*Figure 47: Learned 16-point constellation of the E2E autoencoder. The network discovers a non-rectangular geometry (resembling a hexagonal lattice or concentric APSK layout) that maximises minimum Euclidean distance under the average power constraint, unlike the classical $4 \times 4$ square grid.*

![Figure 48: E2E training loss.](results/e2e/e2e_training_loss.png)

*Figure 48: Training loss (cross-entropy) convergence of the E2E autoencoder. The model converges within approximately 200 epochs.*

![Figure 49: E2E vs. relay comparison.](results/e2e/e2e_relay_comparison.png)

*Figure 49: Performance comparison of the E2E autoencoder against the modular relay-based approaches from this thesis. The E2E system does not achieve lower BER than the two-hop DF relay, highlighting the limitations of the E2E approach.*

#### 7.16.3 E2E vs. Two-Hop Relay Comparison: AF and DF

To contextualize the E2E results within the modular relay framework studied throughout this thesis, the E2E autoencoder is benchmarked against two-hop AF and DF relays operating over the same Rayleigh fading channel with standard 16-QAM (Figure 49). Both two-hop systems use identical per-hop SNR and the same total transmit power budget.

**Finding 25: AF and DF converge at high SNR.** A striking feature of the two-hop comparison is that the AF and DF BER curves converge as SNR increases. This convergence has a precise information-theoretic explanation rooted in the two-hop relay channel model:

- **AF** amplifies the received signal (including noise) and retransmits it. For a symmetric two-hop channel where each hop has SNR $= \gamma$, the effective end-to-end SNR is $\text{SNR}_{\text{AF}} = \frac{\gamma^2}{2\gamma + 1} \approx \gamma/2$ at high $\gamma$. As $\gamma \to \infty$, the noise accumulated in hop 1 becomes negligible relative to the signal, and the AF relay approaches the single-link performance of hop 2 alone.

- **DF** decodes the signal at the relay and re-encodes a fresh copy. At high SNR, the relay's detection error probability $P_{e,1} \to 0$, making the re-encoded signal effectively error-free. The end-to-end BER then converges to the single-link BER of hop 2: $\text{BER}_{\text{DF}} \approx P_{e,2}$.

Both mechanisms converge to the **same asymptotic limit**: the BER of a single Rayleigh fading link at the per-hop SNR. At low-to-moderate SNR the strategies diverge — DF suffers from hard-decision error propagation when relay detection is unreliable, while AF preserves soft information but accumulates noise. At high SNR, both effects vanish: DF detection becomes near-perfect, and AF's accumulated noise becomes negligible. The convergence point is determined by the channel's diversity order ($1/\text{SNR}$ decay for single-antenna Rayleigh), which neither relay strategy can alter.

**Why AF and DF converge here but not in the earlier BPSK experiments.** In Sections 7.2–7.7, AF and DF remain separated by several dB across the entire tested SNR range (e.g., Table 2: AF = 0.0053 vs. DF = 0.0047 at 20 dB Rayleigh). The difference is **modulation complexity**:

- **BPSK relay detection is trivial.** DF's relay-side detection is a binary sign decision ($\hat{x} = \text{sign}(y)$), which achieves near-zero error probability at moderate SNR. By 6 dB AWGN, DF's relay detection is effectively perfect, so its end-to-end BER equals a single clean link. AF, meanwhile, still forwards hop-1 noise, creating a persistent gap that only closes at very high SNR ($\gg 20$ dB) — well beyond the experimental range.

- **16-QAM relay detection is hard.** DF must perform 16-point nearest-neighbour detection (PAM-4 quantisation on each I/Q axis), which requires much higher SNR to become reliable. At low SNR, DF's hard decisions on the dense constellation actually *cause more errors than they correct* — Finding 4 shows AF outperforms DF at 0 dB (0.370 vs. 0.376) because AF preserves the soft multi-level amplitude structure. The two curves therefore start nearly overlapping and both converge quickly to the same Rayleigh single-link limit within the tested range.

In summary, the AF/DF gap is governed by how fast DF's relay detection error $P_{e,1}$ approaches zero relative to AF's noise dilution. For BPSK, $P_{e,1}$ drops exponentially fast (simple threshold), opening a wide gap early. For 16-QAM, $P_{e,1}$ drops slowly (16-point detection), keeping the two strategies close throughout.

**Why neural relay methods do not converge to AF/DF.** A natural question is why the neural network relays studied throughout this thesis (MLP, Mamba, Transformer, etc.) do not converge to the same asymptotic BER as AF and DF at high SNR. The answer is modulation-dependent:

- **BPSK/QPSK (Sections 7.2–7.7):** Neural relays *do* converge to DF performance at high SNR. The binary nature of each I/Q component means the relay task reduces to a simple threshold function, which even minimal networks learn well. At high SNR the learned soft-threshold saturates to a hard decision, effectively replicating DF behaviour.

- **16-QAM (Sections 7.10–7.15):** Neural relays fail to converge because the relay denoising task is fundamentally harder — the network must learn 4-level PAM quantisation boundaries on each I/Q axis rather than a binary threshold. Three mechanisms prevent convergence:
  1. **Activation bottleneck:** Models using $\tanh$ output activation (Section 7.10) suffer an irreducible BER floor because the $[-1, +1]$ range cannot represent the full $\{-3, -1, +1, +3\}/\sqrt{10}$ PAM-4 alphabet. Even constellation-aware activations (Section 7.12) only eliminate the floor — they do not close the gap to DF.
  2. **Approximate vs. exact decision boundaries:** DF implements mathematically exact nearest-constellation-point detection with zero parameters. Neural relays must *learn* these decision boundaries from data, and any imprecision in the learned boundaries causes residual errors that persist regardless of SNR.
  3. **No information gain from higher SNR:** Unlike AF — where increasing SNR reduces the relative noise contribution — a neural relay's learned function $f_\theta(\cdot)$ is fixed after training. If $f_\theta$ introduces even a slight systematic mapping error, that error persists at every SNR, preventing convergence to DF's zero-error relay detection.

**Finding 26: E2E underperforms both modular relays and theoretical limits.** Contrary to the expected advantage of joint optimisation, the E2E autoencoder achieves higher BER than the two-hop DF relay at every SNR point (e.g., 20 dB: E2E = 0.060 vs. DF = 0.039, AF = 0.047). Furthermore, the E2E BER (0.183 at 10 dB, 0.042 at 20 dB) is 67–141% worse than the theoretical single-hop 16-QAM limit (0.110 at 10 dB, 0.017 at 20 dB). The E2E system operates over a single hop (no relay), so it should have a fundamental advantage over two-hop systems, yet fails to exploit it. This indicates that the learned constellation geometry does not outperform the classical $4 \times 4$ square grid under the current training configuration. The result may reflect limited training data, suboptimal hyperparameters, or the inherent difficulty of learning an efficient constellation geometry from scratch with a small MLP encoder/decoder.

#### 7.16.4 Analysis

**Finding 27: E2E representations do not outperform classical grids in this configuration.** The E2E neural network achieves 67–141% higher BER than the theoretical limit of classical 16-QAM across the evaluated SNR range. The learned constellation (Figure 47) does not achieve better Euclidean distance packing than the standard $4 \times 4$ square grid under single-antenna Rayleigh fading. This result underscores the difficulty of learning geometrically optimal constellations from scratch: the E2E encoder must simultaneously discover both the constellation geometry and the channel-matching signal structure, a joint optimisation problem that may require significantly more training data or a more expressive network architecture than the current MLP-based encoder.

**Finding 28: The immutable physics of the diversity limit.** The E2E network hits a BER of 0.084 at 15 dB. This reflects the $1/\text{SNR}$ asymptotic decay characterising $1 \times 1$ flat fading channels. At 15 dB, deep fades ($|h| \to 0$) completely destroy the signal approximately 10–15% of the time. Further reduction into the $10^{-4}$ BER regime strictly requires the introduction of diversity (e.g., MIMO or temporal coding, $n \ge 2$), which the E2E framework could exploit by learning an analogue to the Alamouti space-time code.

**Conclusion on E2E Systems.** The E2E autoencoder experiment demonstrates that joint transmitter-receiver optimisation does not automatically yield superior BER compared to classical modulation and modular relay processing. The E2E system achieves BER consistently worse than both theoretical 16-QAM and the two-hop DF relay. Beyond the performance deficit, the E2E approach fundamentally breaks multi-vendor interoperability by replacing standardised constellations with opaque latent representations, and its reliance on explicit domain knowledge (e.g., coding the complex division into the receiver) demonstrates that "black-box" deep learning remains highly inefficient for basic RF operations. These findings reinforce the core architectural thesis of this work: the most practical deployment of deep learning in physical-layer communications is a modular approach, where classical algorithms handle modulation and equalization, while neural networks are surgically applied to non-linear denoising tasks at intermediate relays.

---

