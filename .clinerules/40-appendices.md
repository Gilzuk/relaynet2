# Appendix Rules

## Structure
- Ch9 (`ch09_appendices.tex`) contains all appendix content in this order:
  1. **Appendix A** — Mathematical Notation (`sec:appendix-a-mathematical-notation`)
  2. **Appendix B** — Model Architectures and Hyperparameters (`sec:appendix-b-model-architectures-and-hyperparameters`)
  3. **Appendix C** — Software Architecture (`sec:appendix-c-software-architecture`)
  4. **Appendix D** — Normalized 3K-Parameter Configurations (`sec:appendix-d-normalized-3k-parameter-configurations`)
  5. **Channel Model Validation Charts** (`chap:app-validation`) — figures fig:fig1–fig:fig7
  6. **Detailed Experimental Tables** (`chap:app-tables`) — all BER longtables (tbl:table1–tbl:table24)
  7. **48-Variant CSI Experiment** (`chap:app-csi`) — figures fig:fig39–fig:fig45
  8. **Experimental Results** (`chap:app-experiments`) — all BER figures organized by experiment (E1–E8)

## Experimental Results Chapter (`chap:app-experiments`)
- Contains one section per experiment: `sec:app-e1` through `sec:app-e8`.
- Each section contains the figures for that experiment only.
- E6 section (`sec:app-e6`) redirects to `chap:app-csi` (no duplicate figures).

## Figure Requirements
- Every figure must have `\caption{...}` followed immediately by `\label{fig:figXX}`.
- Use `\begin{figure}[H]` for all figures.
- Figure labels follow the sequence: `fig:fig1`, `fig:fig2`, ..., `fig:fig53`.

## Table Requirements
- Every table must have `\caption{...}` with `\label{tbl:tableXX}`.
- BER tables use `longtable` environment.
- Table labels: `tbl:table1` through `tbl:table24`, plus `tbl:master-config`.

## Content Rules
- Appendices may contain: figures, tables, and concise explanations.
- Appendix content must **not** introduce new conclusions.
- Appendix content must **not** repeat text from the main chapters.
- Every appendix figure and table must be explicitly referenced from Ch5 (Experiments).

## Allowed Appendix Labels
| Label | Content |
|-------|---------|
| `chap:app-validation` | Channel model validation charts |
| `chap:app-tables` | All numerical BER tables |
| `chap:app-csi` | 48-variant CSI & LayerNorm experiment |
| `chap:app-experiments` | All BER figures by experiment |
| `sec:app-e1` – `sec:app-e8` | Per-experiment figure sections |