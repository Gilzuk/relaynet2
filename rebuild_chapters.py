"""
Rebuild ch05_experiments.tex and ch09_appendices.tex per user requirements:
1. ch05: Add one representative figure per experiment
2. ch09: Use appendix structure, convert B to table, C to TikZ, consolidate chapters
"""
import os

# ============================================================
# ch05_experiments.tex - with one figure per experiment
# ============================================================
ch05 = r"""%% ============================================================
%% Chapter 5: Experiments
%% Structure: Master config table + one-figure-per-experiment sections
%% Full numerical tables and all figures are in Appendix
%% ============================================================
\chapter{Experiments}\label{sec:experiments}

This chapter presents the experimental evaluation of nine relay strategies across eight
systematic experiments. Each experiment is described by its goal, configuration,
conclusion, and one representative figure that supports the key finding. Complete
numerical tables and all supporting figures are consolidated in
Appendix~\ref{chap:app-experiments}.

%% ============================================================
%% Master Configuration Table
%% ============================================================
\section{Experiment Overview}\label{sec:experiment-overview}

Table~\ref{tbl:master-config} summarises the configuration space explored across all
eight experiments. Each row defines the topology, modulation, channel, equalizer, and
neural-network settings used; outcomes are reported in the per-experiment sections below
and detailed in Appendix~\ref{chap:app-experiments}.

\begin{table}[H]
\centering
\caption{Master experiment configuration table. All nine relay strategies
(AF, DF, MLP, Hybrid, VAE, CGAN, Transformer, Mamba~S6, Mamba-2~SSD) are evaluated
unless otherwise noted. ``CA'' = Constellation-Aware; ``E2E'' = End-to-End.}
\label{tbl:master-config}
\small
\begin{tabular}{@{}p{0.04\linewidth}p{0.18\linewidth}p{0.10\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.10\linewidth}p{0.22\linewidth}@{}}
\toprule
\textbf{E\#} & \textbf{Experiment} & \textbf{Topology} & \textbf{Modulation} & \textbf{Channel} & \textbf{Equalizer} & \textbf{Special Setting} \\
\midrule
E1 & Channel Model Validation & SISO, MIMO 2$\times$2 & BPSK & AWGN, Rayleigh, Rician & None / ZF / MMSE / SIC & Theory vs.\ simulation \\
\midrule
E2 & SISO BPSK Baseline & SISO & BPSK & AWGN, Rayleigh, Rician & None & Activation: \texttt{tanh} \\
\midrule
E3 & MIMO 2$\times$2 BPSK & MIMO 2$\times$2 & BPSK & Rayleigh & ZF, MMSE, SIC & Activation: \texttt{tanh} \\
\midrule
E4 & Parameter Normalisation & SISO, MIMO 2$\times$2 & BPSK & AWGN, Rayleigh, Rician, MIMO & None / ZF / MMSE / SIC & $\sim$3\,K normalised params \\
\midrule
E5 & Higher-Order Modulation & SISO & QPSK, 16-QAM & AWGN, Rayleigh & None & CA activations: \texttt{tanh}, \texttt{hardtanh}, scaled \texttt{tanh}, scaled sigmoid \\
\midrule
E6 & CSI Injection \& LayerNorm & SISO & 16-QAM, 16-PSK & Rayleigh & None & 48 variants: Baseline / CSI / LN / CSI+LN \\
\midrule
E7 & 16-Class 2D Classification & SISO & 16-QAM & AWGN & None & Joint 2D softmax (16-class) \\
\midrule
E8 & End-to-End Optimisation & SISO & Learned ($M{=}16$) & Rayleigh & ZF (receiver) & MLP autoencoder (E2E) \\
\bottomrule
\end{tabular}
\end{table}

%% ============================================================
%% E1: Channel Model Validation
%% ============================================================
\section{E1: Channel Model Validation}\label{sec:channel-model-validation}

The goal of this experiment is to validate the simulation framework against closed-form
theoretical BER expressions, ensuring baseline accuracy before evaluating AI relays.

The experiment covers both SISO and MIMO 2$\times$2 topologies using BPSK modulation.
Three channel models are evaluated: AWGN, Rayleigh fading, and Rician fading with
$K{=}3$. For the MIMO topology, three equalizers are applied independently: ZF, MMSE,
and SIC. All demodulation uses hard-decision decoding. No neural relay is involved in
this experiment; only the classical AF and DF strategies are compared against
single-hop theoretical bounds.

Monte Carlo simulations match theoretical predictions within the 95\,\% confidence
interval at all SNR points for every channel and equalizer combination.
The simulation framework is validated and used as the ground truth for all subsequent
experiments.
Full validation charts are provided in Appendix~\ref{sec:app-e1}
(Figures~\ref{fig:fig1}--\ref{fig:fig7}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/channel_analysis_summary.png}}
\caption{E1 --- Consolidated channel model validation: theory vs.\ simulation for
AWGN, Rayleigh, and Rician channels (top row) and MIMO equalizer comparison (bottom
row). All simulated curves fall within the 95\,\% confidence interval of the
theoretical predictions, confirming framework accuracy.}
\label{fig:fig6}
\end{figure}

%% ============================================================
%% E2: SISO BPSK Baseline
%% ============================================================
\section{E2: SISO BPSK Performance (Baseline Relay Comparison)}\label{sec:siso-bpsk-performance-baseline-relay-comparison}

The goal of this experiment is to evaluate baseline classical and AI-based relay
strategies on single-antenna configurations across different fading environments.

The experiment uses a SISO topology with BPSK modulation. Three channel conditions are
evaluated: AWGN, Rayleigh fading, and Rician fading with $K{=}3$. No equalizer is
applied. All seven neural relay architectures are evaluated --- MLP, Hybrid, VAE, CGAN,
Transformer, Mamba~S6, and Mamba-2~SSD --- alongside the classical AF and DF baselines.
All neural relays use a \texttt{tanh} output activation and are trained with a
multi-SNR protocol at 5, 10, and 15\,dB.

AI relays selectively outperform AF and, on AWGN and Rician channels, DF at low SNR
(0--4\,dB). Under Rayleigh fading, classical DF dominates even at low SNR. Across all
channels, DF remains dominant at medium-to-high SNR ($\geq 6$\,dB), matching or
exceeding all AI methods with zero parameters.
BER curves and numerical tables are in Appendix~\ref{sec:app-e2}
(Figures~\ref{fig:fig9}--\ref{fig:fig11}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/awgn_comparison_ci.png}}
\caption{E2 --- SISO BPSK on AWGN: BER vs.\ SNR for all nine relay strategies with
95\,\% CI. AI relays outperform AF and DF at low SNR (0--4\,dB); classical DF
dominates at medium-to-high SNR ($\geq 6$\,dB).}
\label{fig:fig9}
\end{figure}

%% ============================================================
%% E3: MIMO 2x2 BPSK
%% ============================================================
\section{E3: MIMO 2$\times$2 BPSK Performance}\label{sec:mimo-2x2-bpsk-performance}

The goal of this experiment is to evaluate relay strategies under spatial multiplexing
and various interference cancellation techniques.

The experiment uses a MIMO 2$\times$2 topology with BPSK modulation over a Rayleigh
fading channel. Three equalizers are applied at the receiver: Zero-Forcing (ZF), Minimum
Mean Square Error (MMSE), and MMSE with Successive Interference Cancellation (SIC).
All nine relay strategies are evaluated with a \texttt{tanh} output activation. The
experiment is designed to isolate the interaction between relay processing quality and
equalization quality, testing whether the AI advantage observed in SISO transfers to
the MIMO setting.

The MIMO equalization hierarchy (ZF $<$ MMSE $<$ SIC) holds for all relay types. The
AI advantage at low SNR is preserved under ZF and MMSE (Mamba~S6 achieves the lowest
BER), but under SIC equalization, classical DF provides the lowest BER at all
low-to-medium SNR points. Relay processing gains and MIMO equalization gains are
additive.
BER curves and numerical tables are in Appendix~\ref{sec:app-e3}
(Figures~\ref{fig:fig12}--\ref{fig:fig14}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/mimo_equalizer_comparison.png}}
\caption{E3 --- MIMO 2$\times$2 equalizer comparison: single-hop BER with ZF, MMSE,
and SIC equalization. MMSE provides $\sim$1--2\,dB gain over ZF; SIC provides an
additional $\sim$0.5--1\,dB gain. The equalization hierarchy holds for all relay types.}
\label{fig:fig5}
\end{figure}

%% ============================================================
%% E4: Parameter Normalisation
%% ============================================================
\section{E4: Parameter Normalisation \& Complexity Trade-off}\label{sec:parameter-normalization-complexity-trade-off}

The goal of this experiment is to isolate architectural inductive biases from
parameter-count effects and characterise the complexity--performance trade-off for relay
denoising.

The experiment covers both SISO and MIMO 2$\times$2 topologies with BPSK modulation
across all six channel configurations (AWGN, Rayleigh, Rician, and MIMO with ZF, MMSE,
and SIC). All nine relay architectures are evaluated at two scales: their original
parameter counts (ranging from 169 to 26\,179 parameters) and a normalised scale of
approximately 3\,000 parameters. The normalised configurations are designed to match
parameter counts within $\pm$1.2\,\% of the 3\,000-parameter target while preserving
each architecture's structural characteristics. All models use a \texttt{tanh} output
activation.

The relay denoising task exhibits an inverted-U complexity relationship: a minimal
169-parameter MLP matches models 140$\times$ larger, while excessive parameters
($>$11\,K) lead to overfitting. At a normalised scale of 3\,000 parameters, the
performance gap between feedforward and sequence architectures narrows to $\sim$1\,\%
BER, indicating that parameter count rather than architectural choice is the primary
performance driver. Generative VAE is a consistent underperformer due to probabilistic
overhead.
BER curves, complexity charts, and numerical tables are in Appendix~\ref{sec:app-e4}
(Figures~\ref{fig:fig15}--\ref{fig:fig20}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/normalized_3k_all_channels.png}}
\caption{E4 --- Normalised 3K-parameter comparison across all channels. At equal
parameter budgets, all architectures converge to similar BER, with VAE being the
consistent underperformer. The gap between feedforward and sequence models narrows
to $\sim$1\,\% BER.}
\label{fig:fig15}
\end{figure}

%% ============================================================
%% E5: Higher-Order Modulation
%% ============================================================
\section{E5: Higher-Order Modulation Scalability (Constellation-Aware Training)}\label{sec:higher-order-modulation-scalability-constellation-aware-training}

The goal of this experiment is to evaluate the generalisability of BPSK-trained relays
to complex constellations and resolve the multi-level amplitude bottleneck.

The experiment uses a SISO topology with two higher-order modulation schemes: QPSK and
16-QAM. Both AWGN and Rayleigh fading channels are evaluated. No equalizer is applied.
All nine relay architectures are tested with five activation functions: standard
\texttt{tanh}, linear (no activation), \texttt{hardtanh} (clipped tanh), scaled
\texttt{tanh}, and scaled sigmoid. The constellation-aware activations are bounded to
the precise signal amplitude of the target constellation: $1/\sqrt{2}$ for QPSK and
$3/\sqrt{10}$ for 16-QAM. Each activation variant is retrained from scratch on the
target modulation.

QPSK performance mirrors BPSK perfectly due to I/Q independence. On 16-QAM, standard
\texttt{tanh} compression causes a severe, irreducible BER floor ($\approx$0.22 at
16\,dB). Replacing \texttt{tanh} with constellation-aware bounded activations
(\texttt{hardtanh}, scaled \texttt{tanh}) bounded to the precise signal amplitude
($3/\sqrt{10}$) and retraining eliminates this floor. Sequence models benefit most,
reducing their BER floor by 5$\times$, though a gap to classical DF persists due to
per-axis error accumulation.
BER curves, activation comparisons, and numerical tables are in
Appendix~\ref{sec:app-e5}
(Figures~\ref{fig:fig21}--\ref{fig:fig36}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/modulation/combined_modulation_awgn.png}}
\caption{E5 --- Combined modulation comparison on AWGN: all nine relays across BPSK
(solid), QPSK (dashed, overlapping BPSK), and 16-QAM (dotted). The BPSK/QPSK overlap
confirms I/Q splitting equivalence. The 16-QAM dotted curves reveal the AI relay BER
floor ($\approx$0.18--0.25) that constellation-aware training eliminates.}
\label{fig:fig27}
\end{figure}

%% ============================================================
%% E6: CSI Injection & LayerNorm
%% ============================================================
\section{E6: Input Normalisation and CSI Injection}\label{sec:input-normalization-and-csi-injection}

The goal of this experiment is to evaluate the impact of explicit Channel State
Information (CSI) injection and Input Layer Normalisation across 48 distinct neural
relay configurations for 16-QAM and 16-PSK.

The experiment uses a SISO topology with two modulation schemes: 16-QAM and 16-PSK.
The channel is Rayleigh fading with no equalizer. Three sequence architectures are
evaluated: Transformer, Mamba~S6, and Mamba-2~SSD. For each architecture, four
structural variants are tested: Baseline (no modification), CSI injection only, Input
LayerNorm only, and CSI injection combined with LayerNorm. Each variant is further
evaluated with four activation functions (tanh, hardtanh, scaled tanh, sigmoid),
yielding $4 \times 4 \times 3 = 48$ configurations per modulation scheme. The CSI
injection augments the input window with the estimated channel magnitude $|h|$ as an
additional feature.

A clear dichotomy emerges: explicit CSI injection significantly improves performance
for phase-only modulations (16-PSK) by resolving phase ambiguity, but actively
degrades performance for amplitude-dependent modulations (16-QAM) by disrupting
amplitude-decision boundaries. Input Layer Normalisation is universally beneficial
across all architectures and modulations.
The full 48-variant sweep, BER charts, and training curves are in
Appendix~\ref{chap:app-csi}
(Figures~\ref{fig:fig39}--\ref{fig:fig45}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/csi/top3_qam16_rayleigh.png}}
\caption{E6 --- Top-3 neural relay architectures for 16-QAM in Rayleigh fading. All
three best-performing variants use Input LayerNorm (+LN) without CSI injection,
confirming that LayerNorm is universally beneficial while CSI injection degrades
amplitude-dependent modulations.}
\label{fig:fig40}
\end{figure}

%% ============================================================
%% E7: 16-Class 2D Classification
%% ============================================================
\section{E7: 16-Class 2D Classification for QAM16}\label{sec:class-2d-classification-for-qam16}

The goal of this experiment is to eliminate the structural BER floor imposed by
per-axis I/Q splitting for 16-QAM by utilising full 2D decision boundaries.

The experiment uses a SISO topology with 16-QAM modulation over an AWGN channel. No
equalizer is applied. Five relay architectures are evaluated: MLP, VAE, Transformer,
Mamba~S6, and Mamba-2~SSD. Instead of the standard I/Q splitting approach (which
treats the relay as two independent 4-class classifiers), the relay is reformulated as
a single 16-class joint classifier over the full 2D constellation space. The output
layer uses a softmax activation with cross-entropy loss, replacing the per-axis
regression formulation. The 16-class models are trained from scratch on 16-QAM symbols
and compared against the 4-class I/Q split variants from Experiment~E5.

Treating the relay as a joint 16-point classifier over the full 2D constellation space
completely eliminates the structural 4-class BER floor. For the first time, neural
variants (VAE, Transformer, MLP) matched classical DF performance at high SNR,
achieving near-zero BER at 20\,dB. This proves the previous BER floor was an artefact
of I/Q splitting, not a fundamental limitation of neural relays.
BER curves, bar charts, and numerical tables are in Appendix~\ref{sec:app-e7}
(Figures~\ref{fig:fig50}--\ref{fig:fig53}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/all_relays_16class/ber_all_relays_16class.png}}
\caption{E7 --- 16-QAM BER vs.\ SNR for all relay variants (4-class and 16-class) on
AWGN. The 4-class variants (dashed) plateau at BER $\approx 0.008$ while the
16-class variants (solid) continue decreasing, approaching and matching the DF
baseline at 20\,dB.}
\label{fig:fig50}
\end{figure}

%% ============================================================
%% E8: End-to-End Optimisation
%% ============================================================
\section{E8: End-to-End Joint Optimisation}\label{sec:end-to-end-joint-optimization}

The goal of this experiment is to compare the modular neural relay approach against a
fully joint transmitter--receiver autoencoder.

The experiment uses a SISO topology with a learned latent space of $M{=}16$ symbols
subject to an average power constraint. The channel is Rayleigh fading. A ZF equalizer
is applied at the receiver side. The relay is implemented as an MLP autoencoder
consisting of an encoder (transmitter) and a decoder (receiver), trained jointly
end-to-end to minimise cross-entropy loss. The learned constellation is unconstrained
in geometry, allowing the network to discover any 16-point arrangement that minimises
BER under the power constraint. The E2E system is compared against the theoretical BER
of standard 16-QAM and against the modular two-hop DF relay from Experiment~E5.

The E2E autoencoder underperforms both the theoretical limits of classical 16-QAM and
the modular two-hop DF relay across all SNR points (67--141\,\% higher BER). The
network fails to discover a constellation geometry that surpasses the classical square
grid under single-antenna Rayleigh fading, demonstrating that black-box deep learning
is inefficient compared to modular designs that leverage classical signal processing for
modulation and equalization.
BER curves, learned constellation, and training loss are in Appendix~\ref{sec:app-e8}
(Figures~\ref{fig:fig46}--\ref{fig:fig49}).

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{results/e2e/e2e_relay_comparison.png}}
\caption{E8 --- Performance comparison of the E2E autoencoder against the modular
relay-based approaches. The E2E system does not achieve lower BER than the two-hop
DF relay at any SNR point, demonstrating the superiority of modular design over
black-box end-to-end optimisation.}
\label{fig:fig49}
\end{figure}
"""

with open('chapters/ch05_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(ch05)

print(f'ch05_experiments.tex written: {len(ch05.splitlines())} lines')

import re
figs = re.findall(r'\\begin\{figure\}', ch05)
print(f'Figures: {len(figs)}')