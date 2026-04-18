# LaTeX Structure Rules

## Output Format
- Output LaTeX only unless explicitly asked to explain.
- Preserve existing labels (`\label{}`), references (`\ref{}`, `\cite{}`), and cross-links.
- Do not rename figures, tables, or sections unless instructed.

## Label Conventions
- Sections/chapters: `\label{sec:kebab-case-name}` or `\label{chap:kebab-case-name}`
- Figures: `\label{fig:figXX}` (e.g., `fig:fig9`, `fig:fig21`)
- Tables: `\label{tbl:tableXX}` (e.g., `tbl:table1`, `tbl:master-config`)
- Equations: `\label{eq:kebab-case-name}` (e.g., `eq:awgn-ber`, `eq:zf-equalizer`)
- Appendix chapters: `\label{chap:app-experiments}`, `\label{chap:app-tables}`, etc.

## Figure Rules
- Use `\begin{figure}[H]` for all figures.
- Place `\label{fig:figXX}` immediately after `\caption{...}` (never before).
- Use `\pandocbounded{\includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/...}}` for result figures.
- Never place `\begin{figure}` in Ch5 (Experiments) body — all figures go in Ch9 (Appendices).

## Table Rules
- Use `longtable` for BER result tables (they span multiple pages).
- Use `table` + `tabular` for short summary tables (e.g., master config table).
- Use `tabularx` only when explicitly requested.
- Never place `\begin{longtable}` in Ch5 (Experiments) body — all tables go in Ch9 (Appendices).
- Exception: the master configuration table (`tbl:master-config`) stays in Ch5.

## Chapter Structure
- Ch5 (Experiments): master config table + 8 experiment sections (E1–E8), each with Goal/Configuration/Conclusion + appendix cross-refs only.
- Ch9 (Appendices): all figures and BER tables organized by experiment.

## Compilation
- Engine: XeLaTeX (required for Hebrew abstract and fontspec).
- Full build sequence: `xelatex → bibtex → xelatex → xelatex`.
- After compile: run `python check_log.py` — must show `Undefined References: None`.