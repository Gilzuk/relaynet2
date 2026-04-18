# Research Writing Rules

## Academic Tone
- Use formal academic tone suitable for an M.Sc. thesis or IEEE journal.
- Avoid subjective or promotional language.
- Prefer neutral phrasing (e.g., "results indicate" rather than "results prove").

## Claims and Evidence
- Every claim must be either:
  - Explicitly supported by existing results, OR
  - Clearly marked as interpretation or discussion.
- Do not introduce new claims without references.

## Chapter Content Boundaries
- **Theoretical content** (channel models, BER derivations, NN theory) belongs **only** in Ch1 (Introduction) and Ch2 (Literature Review).
- **Methods** (system model, relay architectures, training protocol) belong in Ch4.
- **Experiment sections** (Ch5) contain Goal → Configuration → Conclusion only — no inline results, no figures, no tables.
- **Discussion** (Ch6) interprets results; it does not repeat them.

## No Repetition
- No repeated equations, figures, or tables across chapters.
- If a concept is defined in Ch1/Ch2, subsequent chapters reference it with `\ref{}` — never redefine it.

## Cross-References
- Always use `\ref{}` for sections, figures, tables, and equations.
- Never hardcode "Section 4.10", "Figure 1", "Table 13" — these must be `\ref{sec:...}`, `\ref{fig:...}`, `\ref{tbl:...}`.

## Equations
- All equations must be numbered using `\begin{equation}...\end{equation}`.
- Every equation must cite its canonical source (textbook, paper, or journal) using `\cite{}`.
- A List of Equations chapter (`ch07_equation_ref.tex`) must be maintained.

## Figures and Tables
- Every figure caption must describe what the reader should observe.
- Every table must have a descriptive caption above the table.