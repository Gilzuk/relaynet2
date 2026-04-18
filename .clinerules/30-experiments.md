# Experiments Chapter Rules

## Chapter Structure
- Ch5 (Experiments) contains exactly two things:
  1. One master configuration table (`tbl:master-config`) summarising all 8 experiments.
  2. Eight experiment sections (E1–E8), each with Goal, Configuration, and Conclusion only.
- No figures, plots, or numerical tables appear in the Ch5 body.

## Master Configuration Table
- One row per experiment.
- Columns: Experiment #, Name, Topology, Modulation, Channel, Equalizer, Special Setting.
- Label: `\label{tbl:master-config}`.

## Experiment Subsections
Each subsection must contain exactly:
1. **Goal** — one or two sentences stating the research question.
2. **Configuration** — written as prose paragraphs (not bullets or bold text), elaborating the master table entry. Must describe topology, modulation, channel, equalizer, NN architecture, activation, and any special case.
3. **Conclusion** — one paragraph summarising the key finding.
4. **Appendix cross-reference** — explicit reference to the relevant appendix section and figure/table labels.

## Experiment Flow (canonical order)
Every experiment follows this configuration axis:
- **Topology**: SISO → MIMO 2×2
- **Modulation**: BPSK → QPSK → 16-QAM → 16-PSK
- **Channel**: AWGN → Rayleigh → Rician (K=3)
- **Equalizer** (MIMO only): ZF → MMSE → SIC
- **NN Architecture**: MLP, Hybrid, VAE, CGAN, Transformer, Mamba S6, Mamba-2 SSD
- **Activation**: tanh, hardtanh, scaled tanh, scaled sigmoid
- **Special cases**: Constellation-Aware training, CSI Injection, LayerNorm, E2E

## The 8 Experiments
| E# | Label | Name |
|----|-------|------|
| E1 | `sec:channel-model-validation` | Channel Model Validation |
| E2 | `sec:siso-bpsk-performance-baseline-relay-comparison` | SISO BPSK Baseline |
| E3 | `sec:mimo-2x2-bpsk-performance` | MIMO 2×2 BPSK |
| E4 | `sec:parameter-normalization-complexity-trade-off` | Parameter Normalisation |
| E5 | `sec:higher-order-modulation-scalability-constellation-aware-training` | Higher-Order Modulation |
| E6 | `sec:input-normalization-and-csi-injection` | CSI Injection & LayerNorm |
| E7 | `sec:class-2d-classification-for-qam16` | 16-Class 2D Classification |
| E8 | `sec:end-to-end-joint-optimization` | End-to-End Optimisation |

## Appendix Cross-Reference Format
Each experiment section must end with a sentence of the form:
> "BER curves and numerical tables are in Appendix~\ref{sec:app-eX} (Figures~\ref{fig:figA}--\ref{fig:figB})."