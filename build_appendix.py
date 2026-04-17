"""
Append the Experimental Results appendix chapter to ch09_appendices.tex.
This chapter contains all figures organized by experiment (E1-E8).
"""

new_chapter = r"""

%% ============================================================
%% Appendix: Experimental Results (Figures per Experiment)
%% ============================================================
\chapter{Experimental Results}\label{chap:app-experiments}

This appendix contains all BER curves, charts, and supporting figures for the eight
experiments described in Chapter~\ref{sec:experiments}.
Each section corresponds to one experiment and is cross-referenced from the main text.

%% ── E1 ──────────────────────────────────────────────────────
\section{E1: Channel Model Validation}\label{sec:app-e1}

Figures~\ref{fig:fig1}--\ref{fig:fig7} are located in
Appendix~\ref{chap:app-validation} (Channel Model Validation Charts).

%% ── E2 ──────────────────────────────────────────────────────
\section{E2: SISO BPSK Performance (Baseline Relay Comparison)}\label{sec:app-e2}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/awgn_comparison_ci.png}}
\caption{AWGN channel --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.
AI relays outperform classical methods at low SNR; DF dominates at medium-to-high SNR.}
\label{fig:fig9}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/fading_comparison.png}}
\caption{Rayleigh fading --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.}
\label{fig:fig10}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/rician_comparison_ci.png}}
\caption{Rician fading ($K{=}3$) --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.}
\label{fig:fig11}
\end{figure}

Numerical BER tables for E2 are in Appendix~\ref{chap:app-tables}
(Tables~\ref{tbl:table1}--\ref{tbl:table3}).

%% ── E3 ──────────────────────────────────────────────────────
\section{E3: MIMO 2$\times$2 BPSK Performance}\label{sec:app-e3}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/mimo_2x2_comparison_ci.png}}
\caption{2$\times$2 MIMO with ZF equalization --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.}
\label{fig:fig12}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/mimo_2x2_mmse_comparison_ci.png}}
\caption{2$\times$2 MIMO with MMSE equalization --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.}
\label{fig:fig13}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/mimo_2x2_sic_comparison_ci.png}}
\caption{2$\times$2 MIMO with MMSE-SIC equalization --- BER vs.\ SNR for all nine relay strategies with 95\,\% CI.}
\label{fig:fig14}
\end{figure}

Numerical BER tables for E3 are in Appendix~\ref{chap:app-tables}
(Tables~\ref{tbl:table4}--\ref{tbl:table6}).

%% ── E4 ──────────────────────────────────────────────────────
\section{E4: Parameter Normalisation \& Complexity Trade-off}\label{sec:app-e4}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_all_channels.png}}
\caption{Normalised 3K-parameter comparison across all channels.
At equal parameter budgets, all architectures converge to similar BER, with VAE being the consistent underperformer.}
\label{fig:fig15}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_awgn.png}}
\caption{Normalised 3K-parameter BER comparison on AWGN.
Mamba-3K and Transformer-3K produce nearly identical BER, eliminating the gap seen at original parameter counts.}
\label{fig:fig16}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_rayleigh.png}}
\caption{Normalised 3K-parameter BER comparison on Rayleigh fading.}
\label{fig:fig17}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_rician_k3.png}}
\caption{Normalised 3K-parameter BER comparison on Rician fading ($K{=}3$).}
\label{fig:fig18}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/complexity_comparison_all_relays.png}}
\caption{Complexity--performance trade-off.
Training time vs.\ parameter count vs.\ BER improvement over DF at low SNR.
The Minimal MLP (169 params) achieves the best efficiency.}
\label{fig:fig19}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/master_ber_comparison.png}}
\caption{Master BER comparison --- consolidated view of all nine relay strategies across all six channel/topology configurations.}
\label{fig:fig20}
\end{figure}

Numerical BER tables for E4 are in Appendix~\ref{chap:app-tables}
(Tables~\ref{tbl:table7}--\ref{tbl:table13}).

%% ── E5 ──────────────────────────────────────────────────────
\section{E5: Higher-Order Modulation Scalability (Constellation-Aware Training)}\label{sec:app-e5}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/bpsk_awgn_ci.png}}
\caption{BPSK on AWGN --- all relay strategies with 95\,\% CI (baseline for modulation comparison).}
\label{fig:fig21}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/bpsk_rayleigh_ci.png}}
\caption{BPSK on Rayleigh fading --- all relay strategies with 95\,\% CI.}
\label{fig:fig22}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qpsk_awgn_ci.png}}
\caption{QPSK on AWGN --- BER curves closely match the BPSK baseline (Figure~\ref{fig:fig21}), confirming I/Q splitting validity.}
\label{fig:fig23}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qpsk_rayleigh_ci.png}}
\caption{QPSK on Rayleigh fading --- same relative ordering as BPSK, confirming hypothesis generalisability.}
\label{fig:fig24}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qam16__awgn_ci.png}}
\caption{16-QAM on AWGN --- AI relays hit a BER floor near 0.22 at medium-high SNR due to \(\tanh\) compression of multi-level signals.}
\label{fig:fig25}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qam16__rayleigh_ci.png}}
\caption{16-QAM on Rayleigh fading --- wider BER gap between modulations under fading; all AI relays significantly worse than DF at every SNR point.}
\label{fig:fig26}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/combined_modulation_awgn.png}}
\caption{Combined modulation comparison (AWGN) --- all nine relays across BPSK (solid), QPSK (dashed, overlapping BPSK), and 16-QAM (dotted).
The BPSK/QPSK overlap confirms I/Q splitting equivalence.
The 16-QAM dotted curves reveal the AI relay BER floor.}
\label{fig:fig27}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/qam16_activation/qam16_activation_awgn.png}}
\caption{16-QAM activation experiment (AWGN) --- dashed lines = \texttt{tanh}/BPSK baseline, solid = linear/QAM16, dotted = \texttt{hardtanh}/QAM16.
Replacing \texttt{tanh} and retraining on QAM16 eliminates the BER floor for all AI relays except Hybrid.}
\label{fig:fig28}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/qam16_activation/qam16_activation_rayleigh.png}}
\caption{16-QAM activation experiment (Rayleigh fading) --- same trend under fading.}
\label{fig:fig29}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/bpsk_activation_awgn.png}}
\caption{BPSK constellation-aware activation comparison (AWGN).
All three bounded activations achieve equivalent BER, confirming that BPSK is insensitive to activation choice.}
\label{fig:fig30}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/bpsk_activation_rayleigh.png}}
\caption{BPSK constellation-aware activation comparison (Rayleigh fading).}
\label{fig:fig31}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/qpsk_activation_awgn.png}}
\caption{QPSK constellation-aware activation comparison (AWGN).}
\label{fig:fig32}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/qpsk_activation_rayleigh.png}}
\caption{QPSK constellation-aware activation comparison (Rayleigh fading).}
\label{fig:fig33}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/qam16_activation_awgn.png}}
\caption{16-QAM constellation-aware activation comparison (AWGN).
All three bounded activations eliminate the \texttt{tanh} BER floor.}
\label{fig:fig34}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/qam16_activation_rayleigh.png}}
\caption{16-QAM constellation-aware activation comparison (Rayleigh fading).}
\label{fig:fig35}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/activation_comparison/various_activation_functions.png}}
\caption{Comparison of activation function shapes (left) and their derivatives (right) for $A_{\max} = 0.9487$ (16-QAM).}
\label{fig:fig36}
\end{figure}

Numerical BER tables for E5 are in Appendix~\ref{chap:app-tables}
(Tables~\ref{tbl:table14}--\ref{tbl:table16}).

%% ── E6 ──────────────────────────────────────────────────────
\section{E6: Input Normalisation and CSI Injection}\label{sec:app-e6}

Figures~\ref{fig:fig39}--\ref{fig:fig45} and the associated tables are located in
Appendix~\ref{chap:app-csi} (48-Variant Neural Relay Experiments).

%% ── E7 ──────────────────────────────────────────────────────
\section{E7: 16-Class 2D Classification for QAM16}\label{sec:app-e7}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/ber_all_relays_16class.png}}
\caption{16-QAM BER vs.\ SNR for all relay variants (4-class and 16-class) on AWGN.
The 4-class variants (dashed) plateau at BER $\approx 0.008$ while the 16-class variants (solid) continue decreasing, approaching and matching the DF baseline.}
\label{fig:fig50}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/grouped_bar_16class.png}}
\caption{Grouped bar comparison of 4-class vs.\ 16-class BER at 20\,dB for each architecture.}
\label{fig:fig51}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/heatmap_all_relays_16class.png}}
\caption{Heatmap of BER across all relay variants and SNR points.
The 16-class variants show a clear colour transition from high BER (warm) at low SNR to near-zero BER (cool) at high SNR.}
\label{fig:fig52}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/top3_16class.png}}
\caption{Top-3 16-class relay variants compared against AF and DF.
VAE 16-cls, Transformer 16-cls, and MLP 16-cls all match or approach DF performance across the full SNR range.}
\label{fig:fig53}
\end{figure}

Numerical BER table for E7 is in Appendix~\ref{chap:app-tables}
(Table~\ref{tbl:table24}).

%% ── E8 ──────────────────────────────────────────────────────
\section{E8: End-to-End Joint Optimisation}\label{sec:app-e8}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_ber_comparison.png}}
\caption{BER vs.\ SNR for E2E learned autoencoder compared to theoretical 16-QAM over Rayleigh fading.
The E2E system underperforms the classical grid constellation across the full SNR range.}
\label{fig:fig46}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_constellation.png}}
\caption{Learned 16-point constellation of the E2E autoencoder.
The network discovers a non-rectangular geometry that maximises minimum Euclidean distance under the average power constraint.}
\label{fig:fig47}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_training_loss.png}}
\caption{Training loss (cross-entropy) convergence of the E2E autoencoder.
The model converges within approximately 200 epochs.}
\label{fig:fig48}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_relay_comparison.png}}
\caption{Performance comparison of the E2E autoencoder against the modular relay-based approaches.
The E2E system does not achieve lower BER than the two-hop DF relay.}
\label{fig:fig49}
\end{figure}
"""

with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Append the new chapter before the end
content += new_chapter

with open('chapters/ch09_appendices.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Appendix updated with Experimental Results chapter.")