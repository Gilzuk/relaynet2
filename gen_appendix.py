"""
Generate the new ch09_appendices.tex with:
1. \appendix structure (sections labeled A, B, C...)
2. Appendix B as summary table
3. Appendix C as TikZ SW architecture diagram
4. All experimental chapters consolidated as sections
"""

# Read the current appendix to extract the BER tables and figures
with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    current = f.read()

# Extract the Mathematical Notation table (between \section{Appendix A} and \section{Appendix B})
import re

# Extract Appendix D table (Normalized 3K)
d_start = current.find(r'\section{Appendix D: Normalized 3K-Parameter Configurations}')
d_end = current.find(r'\begin{center}\rule', d_start)
appendix_d_content = current[d_start:d_end].strip() if d_start != -1 else ''

# Extract all BER tables (longtable environments with \caption{BER comparison...})
# These are in the chap:app-tables section
tables_start = current.find(r'\chapter{Detailed Experimental Tables}')
tables_end = current.find(r'\chapter{48-Variant Neural Relay Experiments')
if tables_start == -1:
    tables_start = current.find(r'This appendix contains the exhaustive numerical BER results')
    tables_end = current.find(r'\chapter{48-Variant Neural Relay Experiments')

ber_tables = current[tables_start:tables_end].strip() if tables_start != -1 and tables_end != -1 else ''

# Extract CSI section content
csi_start = current.find(r'\chapter{48-Variant Neural Relay Experiments')
csi_end = current.find(r'\chapter{Experimental Results}')
csi_content = current[csi_start:csi_end].strip() if csi_start != -1 and csi_end != -1 else ''

# Extract Experimental Results section
exp_start = current.find(r'\chapter{Experimental Results}')
exp_content = current[exp_start:].strip() if exp_start != -1 else ''

print(f'Appendix D content: {len(appendix_d_content)} chars')
print(f'BER tables content: {len(ber_tables)} chars')
print(f'CSI content: {len(csi_content)} chars')
print(f'Experimental Results content: {len(exp_content)} chars')

# Now build the new appendix
new_appendix = r"""\chapter{Appendices}\label{chap:appendices}

%% ── Appendix A: Mathematical Notation ───────────────────────
\section{Mathematical Notation}\label{sec:appendix-a-mathematical-notation}

{\def\LTcaptype{table} % do not increment counter
\begin{longtable}[]{@{}ll@{}}
\toprule\noalign{}
Symbol & Description \\
\midrule\noalign{}
\endhead
\bottomrule\noalign{}
\endlastfoot
\(b\) & Binary bit (\(\in \{0, 1\}\)) \\
\(x\) & Modulated BPSK symbol (\(\in \{-1, +1\}\)) \\
\(y\) & Received signal \\
\(n\) & Noise sample \\
\(h\) & Fading coefficient \\
\(\sigma^2\) & Noise variance \\
\(\text{SNR}\) & Signal-to-Noise Ratio (linear) \\
\(G\) & AF amplification gain \\
\(\mathbf{H}\) & MIMO channel matrix (\(\in \mathbb{C}^{2 \times 2}\)) \\
\(\mathbf{W}\) & Neural network weight matrix \\
\(\mathbf{b}\) & Bias vector \\
\(P_e\) & Bit error rate \\
\(Q(\cdot)\) & Gaussian Q-function \\
\(K\) & Rician K-factor \\
\(\beta\) & VAE KL weight \\
\(\lambda\) & Regularization parameter \\
\(\eta\) & Learning rate \\
\(\Delta\) & State space discretization step \\
\(\mathbf{A}, \mathbf{B}, \mathbf{C}, D\) & State space model matrices \\
\end{longtable}
}

%% ── Appendix B: Model Architectures Summary Table ────────────
\section{Model Architectures and Hyperparameters}\label{sec:appendix-b-model-architectures-and-hyperparameters}

Table~\ref{tbl:arch-summary} summarises the key architectural parameters and training
settings for all nine relay strategies evaluated in this thesis.

\begin{table}[H]
\centering
\caption{Summary of relay architecture configurations and training hyperparameters.
All models trained with MSE loss (regression) or cross-entropy (16-class), Adam
optimiser, 100 epochs, 25\,000 training samples.}
\label{tbl:arch-summary}
\small
\begin{tabular}{@{}p{0.12\linewidth}p{0.08\linewidth}p{0.10\linewidth}p{0.28\linewidth}p{0.10\linewidth}p{0.12\linewidth}@{}}
\toprule
\textbf{Architecture} & \textbf{Params} & \textbf{Device} & \textbf{Key Structure} & \textbf{Output Act.} & \textbf{Train Time} \\
\midrule
AF & 0 & --- & Scale received signal by gain $G$ & --- & 0\,s \\
DF & 0 & --- & Hard-decision demodulate and re-modulate & --- & 0\,s \\
MLP (Minimal) & 169 & CPU & Window(5)$\to$24(ReLU)$\to$1 & tanh & 4.9\,s \\
Hybrid & 169 & CPU & MLP below SNR threshold; DF above & tanh & 4.6\,s \\
VAE & 1\,777 & CPU & Enc: 7$\to$32$\to$16$\to$\(\mu,\sigma^2\)(8); Dec: 8$\to$32$\to$1 & tanh & 21.6\,s \\
CGAN (WGAN-GP) & 2\,946 & CUDA & Gen: (7+8)$\to$32$\to$32$\to$16$\to$1; Critic: (1+7)$\to$32$\to$16$\to$1 & tanh & $\sim$2\,h \\
Transformer & 17\,697 & CUDA & 2 layers, 4 heads, $d_\text{model}$=32, $d_\text{ff}$=128 & tanh & $\sim$8\,min \\
Mamba S6 & 24\,001 & CUDA & 2 layers, expand 32$\to$64, $d_\text{state}$=16, selective $\Delta,B,C$ & tanh & $\sim$36\,min \\
Mamba-2 SSD & 26\,179 & CUDA & 2 layers, SSD kernel, chunk\_size=8, $d_\text{state}$=6 & tanh & $\sim$24\,min \\
\bottomrule
\end{tabular}
\end{table}

%% ── Appendix C: Software Architecture ───────────────────────
\section{Software Architecture}\label{sec:appendix-c-software-architecture}

The project is implemented as a modular Python package (\texttt{relaynet}) with the
architecture shown in Figure~\ref{fig:sw-architecture}. The design follows an
object-oriented pattern with a common \texttt{Relay} base class enabling polymorphic
relay swapping. Monte Carlo simulation is implemented in \texttt{runner.py} with
configurable trial count, bit count, and SNR range.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  pkg/.style={rectangle, draw=tauBlue, fill=tauBlue!10, rounded corners=4pt,
              minimum width=3.2cm, minimum height=0.7cm, font=\small\ttfamily,
              align=center},
  mod/.style={rectangle, draw=tauGray, fill=gray!8, rounded corners=2pt,
              minimum width=2.8cm, minimum height=0.55cm, font=\footnotesize\ttfamily,
              align=center},
  arr/.style={-Stealth, thick, tauBlue},
  grp/.style={rectangle, draw=tauGray!60, dashed, rounded corners=6pt,
              inner sep=6pt}
]

%% Top-level package
\node[pkg] (relaynet) at (0,0) {\textbf{relaynet}};

%% Sub-packages
\node[pkg] (channels)    at (-5.5,-1.8) {channels/};
\node[pkg] (modulation)  at (-1.8,-1.8) {modulation/};
\node[pkg] (relays)      at (1.8,-1.8)  {relays/};
\node[pkg] (simulation)  at (5.5,-1.8)  {simulation/};

%% Arrows from top to sub-packages
\draw[arr] (relaynet) -- (channels);
\draw[arr] (relaynet) -- (modulation);
\draw[arr] (relaynet) -- (relays);
\draw[arr] (relaynet) -- (simulation);

%% channels modules
\node[mod] (awgn)    at (-6.5,-3.2) {awgn.py};
\node[mod] (fading)  at (-5.5,-3.2) {fading.py};
\node[mod] (mimo)    at (-4.5,-3.2) {mimo.py};
\draw[arr] (channels) -- (awgn);
\draw[arr] (channels) -- (fading);
\draw[arr] (channels) -- (mimo);

%% modulation modules
\node[mod] (bpsk)  at (-2.6,-3.2) {bpsk.py};
\node[mod] (qpsk)  at (-1.8,-3.2) {qpsk.py};
\node[mod] (qam)   at (-1.0,-3.2) {qam.py};
\draw[arr] (modulation) -- (bpsk);
\draw[arr] (modulation) -- (qpsk);
\draw[arr] (modulation) -- (qam);

%% relays modules
\node[mod] (af)    at (0.6,-3.2)  {af.py};
\node[mod] (df)    at (1.4,-3.2)  {df.py};
\node[mod] (mlp)   at (2.2,-3.2)  {genai.py};
\node[mod] (vae)   at (3.0,-3.2)  {vae.py};
\draw[arr] (relays) -- (af);
\draw[arr] (relays) -- (df);
\draw[arr] (relays) -- (mlp);
\draw[arr] (relays) -- (vae);

%% simulation modules
\node[mod] (runner) at (5.0,-3.2) {runner.py};
\node[mod] (stats)  at (6.0,-3.2) {statistics.py};
\draw[arr] (simulation) -- (runner);
\draw[arr] (simulation) -- (stats);

%% Dependency arrows
\draw[arr, dashed, gray] (runner) to[bend right=20] node[below, font=\tiny] {uses} (relays);
\draw[arr, dashed, gray] (runner) to[bend right=30] node[below, font=\tiny] {uses} (channels);
\draw[arr, dashed, gray] (runner) to[bend right=40] node[below, font=\tiny] {uses} (modulation);

\end{tikzpicture}
\caption{Software architecture of the \texttt{relaynet} package. Solid arrows indicate
containment (sub-package or module); dashed arrows indicate runtime dependencies.
The \texttt{simulation/runner.py} orchestrates Monte Carlo BER evaluation by
composing channel, modulation, and relay components.}
\label{fig:sw-architecture}
\end{figure}

\textbf{Testing:} 126 automated tests (pytest) cover all channels, modulation schemes,
relay strategies, simulation, and statistics modules with 100\,\% pass rate.

\textbf{Reproducibility:} Random seeds are controlled at the bit-generation and
per-trial noise levels to ensure reproducible results.

%% ── Appendix D: Normalized 3K-Parameter Configurations ──────
\section{Normalised 3K-Parameter Configurations}\label{sec:appendix-d-normalized-3k-parameter-configurations}

{\def\LTcaptype{table} % do not increment counter
\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.2500}}@{}}
\toprule\noalign{}
\begin{minipage}[b]{\linewidth}\raggedright
Model
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
Parameters
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
Window
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
Hidden / Architecture
\end{minipage} \\
\midrule\noalign{}
\endhead
\bottomrule\noalign{}
\endlastfoot
MLP-3K & 3,004 & 11 & hidden=231 \\
Hybrid-3K & 3,004 & 11 & hidden=231 (+ DF switch) \\
VAE-3K & 3,037 & 11 & latent=10, hidden=(44, 20) \\
CGAN-3K & 3,004 & 11 & noise=8, g\_hidden=(30, 30, 16), c\_hidden=(32, 16) \\
Transformer-3K & 3,007 & 11 & d\_model=18, heads=2, layers=1 \\
Mamba-3K & 3,027 & 11 & d\_model=16, d\_state=6, layers=1 \\
Mamba2-3K & 3,004 & 11 & d\_model=15, d\_state=6, chunk\_size=8, layers=1 \\
\end{longtable}
}

All 3K configurations use a window size of 11 to provide a common input context.
Parameter counts are within $\pm$1.2\,\% of the 3\,000-parameter target.

"""

# Now add the experimental results sections
# Extract figures from the current appendix

# E1 figures (fig1-fig7)
e1_section = r"""
%% ── E1: Channel Model Validation ────────────────────────────
\section{E1: Channel Model Validation}\label{sec:app-e1}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/channel_theoretical_awgn.png}}
\caption{AWGN Channel --- Theoretical vs.\ Simulative BER. Theory (solid lines) and
Monte Carlo simulation (markers with 95\,\% CI) match within the confidence interval
at all SNR points.}
\label{fig:fig1}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/channel_theoretical_rayleigh.png}}
\caption{Rayleigh fading --- theory vs.\ simulation for single-hop and two-hop DF.
The characteristic $1/\text{SNR}$ slope is clearly visible.}
\label{fig:fig2}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/channel_theoretical_rician.png}}
\caption{Rician fading ($K{=}3$) --- theory vs.\ simulation for single-hop and two-hop DF.}
\label{fig:fig3}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/channel_fading_pdf.png}}
\caption{Left --- PDF of $|h|$ for Rayleigh and Rician ($K{=}1, 3, 10$). Right --- CDF
(outage probability). Rayleigh has the highest deep-fade probability at any threshold.}
\label{fig:fig4}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/mimo_equalizer_comparison.png}}
\caption{2$\times$2 MIMO Rayleigh --- single-hop BER with ZF, MMSE, and SIC equalization.
MMSE provides $\sim$1--2\,dB gain over ZF; SIC provides an additional $\sim$0.5--1\,dB gain.}
\label{fig:fig5}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/channel_comparison_all.png}}
\caption{Single-hop BPSK BER for all three SISO channel models. The SNR penalty for
Rayleigh relative to AWGN exceeds 15\,dB at BER $= 10^{-3}$.}
\label{fig:fig7}
\end{figure}

"""

# E2 figures (fig9-fig11)
e2_section = r"""
%% ── E2: SISO BPSK Baseline ───────────────────────────────────
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

"""

# E3 figures (fig12-fig14)
e3_section = r"""
%% ── E3: MIMO 2x2 BPSK ───────────────────────────────────────
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

"""

# E4 figures (fig15-fig20)
e4_section = r"""
%% ── E4: Parameter Normalisation ─────────────────────────────
\section{E4: Parameter Normalisation \& Complexity Trade-off}\label{sec:app-e4}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_all_channels.png}}
\caption{Normalised 3K-parameter comparison across all channels. At equal parameter
budgets, all architectures converge to similar BER, with VAE being the consistent
underperformer.}
\label{fig:fig15}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/normalized_3k_awgn.png}}
\caption{Normalised 3K-parameter BER comparison on AWGN. Mamba-3K and Transformer-3K
produce nearly identical BER, eliminating the gap seen at original parameter counts.}
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
\caption{Complexity--performance trade-off. Training time vs.\ parameter count vs.\
BER improvement over DF at low SNR. The Minimal MLP (169 params) achieves the best
efficiency.}
\label{fig:fig19}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/master_ber_comparison.png}}
\caption{Master BER comparison --- consolidated view of all nine relay strategies
across all six channel/topology configurations.}
\label{fig:fig20}
\end{figure}

"""

# E5 figures (fig21-fig36)
e5_section = r"""
%% ── E5: Higher-Order Modulation ─────────────────────────────
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
\caption{QPSK on AWGN --- BER curves closely match the BPSK baseline, confirming I/Q splitting validity.}
\label{fig:fig23}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qpsk_rayleigh_ci.png}}
\caption{QPSK on Rayleigh fading --- same relative ordering as BPSK.}
\label{fig:fig24}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qam16__awgn_ci.png}}
\caption{16-QAM on AWGN --- AI relays hit a BER floor near 0.22 at medium-high SNR
due to \texttt{tanh} compression of multi-level signals.}
\label{fig:fig25}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/qam16__rayleigh_ci.png}}
\caption{16-QAM on Rayleigh fading --- wider BER gap; all AI relays significantly
worse than DF at every SNR point.}
\label{fig:fig26}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/modulation/combined_modulation_awgn.png}}
\caption{Combined modulation comparison (AWGN) --- all nine relays across BPSK (solid),
QPSK (dashed), and 16-QAM (dotted). The BPSK/QPSK overlap confirms I/Q splitting
equivalence.}
\label{fig:fig27}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/qam16_activation/qam16_activation_awgn.png}}
\caption{16-QAM activation experiment (AWGN) --- replacing \texttt{tanh} and retraining
on QAM16 eliminates the BER floor for all AI relays except Hybrid.}
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
\caption{BPSK constellation-aware activation comparison (AWGN). All three bounded
activations achieve equivalent BER, confirming BPSK is insensitive to activation choice.}
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
\caption{16-QAM constellation-aware activation comparison (AWGN). All three bounded
activations eliminate the \texttt{tanh} BER floor.}
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
\caption{Comparison of activation function shapes (left) and their derivatives (right)
for $A_{\max} = 0.9487$ (16-QAM).}
\label{fig:fig36}
\end{figure}

"""

# E6 CSI section
e6_section = r"""
%% ── E6: CSI Injection & LayerNorm ───────────────────────────
\section{E6: Input Normalisation and CSI Injection}\label{chap:app-csi}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/csi_experiment_qam16_rayleigh.png}}
\caption{16-QAM Rayleigh fading --- BER vs.\ SNR for all 48 neural relay variants and
two classical baselines (AF, DF) with 95\,\% confidence intervals.}
\label{fig:fig39}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/top3_qam16_rayleigh.png}}
\caption{Top-3 neural relay architectures for 16-QAM in Rayleigh fading. All three
best-performing variants use input LayerNorm (+LN) without CSI injection.}
\label{fig:fig40}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/csi_experiment_psk16_rayleigh.png}}
\caption{16-PSK Rayleigh fading --- BER vs.\ SNR for all 48 neural relay variants.}
\label{fig:fig41}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/top3_psk16_rayleigh.png}}
\caption{Top-3 neural relay architectures for 16-PSK in Rayleigh fading. In contrast
to QAM16, all three best-performing PSK16 variants use CSI injection (+CSI or +CSI+LN).}
\label{fig:fig42}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/training_Mamba2_LN_scaled_tanh.png}}
\caption{Training loss and accuracy for Mamba-2 (+LN scaled\_tanh). The model converges
within approximately 5 epochs with minimal overfitting.}
\label{fig:fig43}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/training_Mamba_CSI_tanh.png}}
\caption{Training loss and accuracy for Mamba S6 (+CSI tanh). The CSI-augmented input
produces smooth, monotonic training convergence.}
\label{fig:fig44}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/csi/training_Transformer_CSI_sigmoid.png}}
\caption{Training loss and accuracy for Transformer (+CSI sigmoid). Rapid initial
convergence within 3 epochs followed by a flat plateau.}
\label{fig:fig45}
\end{figure}

"""

# E7 figures (fig50-fig53)
e7_section = r"""
%% ── E7: 16-Class 2D Classification ──────────────────────────
\section{E7: 16-Class 2D Classification for QAM16}\label{sec:app-e7}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/ber_all_relays_16class.png}}
\caption{16-QAM BER vs.\ SNR for all relay variants (4-class and 16-class) on AWGN.
The 4-class variants (dashed) plateau at BER $\approx 0.008$ while the 16-class
variants (solid) continue decreasing, approaching and matching the DF baseline.}
\label{fig:fig50}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/grouped_bar_16class.png}}
\caption{Grouped bar comparison of 4-class vs.\ 16-class BER at 20\,dB for each
architecture.}
\label{fig:fig51}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/heatmap_all_relays_16class.png}}
\caption{Heatmap of BER across all relay variants and SNR points. The 16-class variants
show a clear colour transition from high BER (warm) to near-zero BER (cool) at high SNR.}
\label{fig:fig52}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/all_relays_16class/top3_16class.png}}
\caption{Top-3 16-class relay variants compared against AF and DF. VAE 16-cls,
Transformer 16-cls, and MLP 16-cls all match or approach DF performance.}
\label{fig:fig53}
\end{figure}

"""

# E8 figures (fig46-fig49)
e8_section = r"""
%% ── E8: End-to-End Optimisation ─────────────────────────────
\section{E8: End-to-End Joint Optimisation}\label{sec:app-e8}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_ber_comparison.png}}
\caption{BER vs.\ SNR for E2E learned autoencoder compared to theoretical 16-QAM over
Rayleigh fading. The E2E system underperforms the classical grid constellation.}
\label{fig:fig46}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_constellation.png}}
\caption{Learned 16-point constellation of the E2E autoencoder. The network discovers
a non-rectangular geometry that maximises minimum Euclidean distance.}
\label{fig:fig47}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_training_loss.png}}
\caption{Training loss (cross-entropy) convergence of the E2E autoencoder. The model
converges within approximately 200 epochs.}
\label{fig:fig48}
\end{figure}

\begin{figure}[H]
\centering
\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/e2e/e2e_relay_comparison.png}}
\caption{Performance comparison of the E2E autoencoder against the modular relay-based
approaches. The E2E system does not achieve lower BER than the two-hop DF relay.}
\label{fig:fig49}
\end{figure}

"""

# Now extract the BER tables from the current appendix
# Find the tables section
tables_marker = 'This appendix contains the exhaustive numerical BER results'
tables_start_idx = current.find(tables_marker)
if tables_start_idx == -1:
    # Try alternative
    tables_start_idx = current.find(r'\chapter{Detailed Experimental Tables}')
    if tables_start_idx != -1:
        tables_start_idx = current.find('\n', tables_start_idx) + 1

csi_marker = r'\chapter{48-Variant Neural Relay Experiments'
csi_idx = current.find(csi_marker)

if tables_start_idx != -1 and csi_idx != -1:
    ber_tables_content = current[tables_start_idx:csi_idx].strip()
else:
    ber_tables_content = '% BER tables not found'

print(f'BER tables extracted: {len(ber_tables_content)} chars')

# Extract CSI tables
exp_results_marker = r'\chapter{Experimental Results}'
exp_results_idx = current.find(exp_results_marker)

if csi_idx != -1 and exp_results_idx != -1:
    csi_tables_content = current[csi_idx:exp_results_idx].strip()
    # Remove the \chapter{} line and replace with section
    csi_tables_content = re.sub(
        r'\\chapter\{48-Variant Neural Relay Experiments.*?\}',
        '',
        csi_tables_content
    )
    # Remove the \label{chap:app-csi} if present (we'll add it in the section)
    csi_tables_content = csi_tables_content.replace(r'\label{chap:app-csi}', '')
else:
    csi_tables_content = ''

# Build the tables section
tables_section = r"""
%% ── Detailed Experimental Tables ────────────────────────────
\section{Detailed Experimental Tables}\label{chap:app-tables}

This section contains the exhaustive numerical BER results for all experiments
described in Chapter~\ref{sec:experiments}.

"""
tables_section += ber_tables_content

# Build the complete new appendix
full_appendix = (
    new_appendix +
    e1_section +
    e2_section +
    e3_section +
    e4_section +
    e5_section +
    e6_section +
    e7_section +
    e8_section +
    tables_section
)

with open('chapters/ch09_appendices.tex', 'w', encoding='utf-8') as f:
    f.write(full_appendix)

print(f'ch09_appendices.tex written: {len(full_appendix.splitlines())} lines')

import re
figs = re.findall(r'\\begin\{figure\}', full_appendix)
tables = re.findall(r'\\begin\{longtable\}', full_appendix)
print(f'Figures: {len(figs)}, Longtables: {len(tables)}')

# Check for duplicate labels
labels = re.findall(r'\\label\{([^}]+)\}', full_appendix)
from collections import Counter
dups = {k: v for k, v in Counter(labels).items() if v > 1}
print(f'Duplicate labels: {len(dups)}')
if dups:
    for k, v in dups.items():
        print(f'  {k}: {v} times')