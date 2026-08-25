# Safety & Integrity Rules

## Scientific Integrity
- **DO NOT**:
  - Modify experimental results (BER values, tables, figures)
  - Alter conclusions or findings
  - Regenerate figures from different data
  - Fabricate data or performance numbers
- Never change numerical results unless explicitly instructed with new data.
- If a request risks scientific integrity, ask for clarification before proceeding.

## LaTeX Safety
- Never delete `\label{}` commands — they break cross-references throughout the document.
- Never rename section/figure/table labels without updating all `\ref{}` calls.
- Never remove `\cite{}` commands without replacing them with an equivalent citation.
- Preserve `\phantomsection` and `\protect` commands in equation environments.

## Compilation Safety
- After any edit to `chapters/*.tex`, always run the full compile sequence:
  `xelatex → bibtex → xelatex → xelatex`
- After compile, run `python check_log.py` — must show `Undefined References: None`.
- If compilation fails, fix errors before committing to git.
- **Commit the rebuilt `thesis/main.pdf` in the same commit as the source change.**
  The PDF is the deliverable; a commit that updates the LaTeX and leaves the PDF
  behind ships a document missing the change, and nothing in CI catches it. This
  has happened: PR #25 merged five chapters' worth of edits while the committed
  PDF stayed at its pre-session 120-page build. Applies to any change that
  re-renders — `chapters/**`, `main.tex`, `references.bib`, or an included figure
  — and not to Python-, results- or notes-only commits.
- Validate with `latexmk -xelatex`, not a hand-rolled pass sequence, and check the
  *final* pass (see `memory-bank/techContext.md` on the bidi failure the manual
  sequence hides). Report the page count whenever it moves.

## File Safety
- Never edit `overleaf_thesis/` files directly — always edit `chapters/*.tex` and run `python sync_overleaf.py`.
- Never delete figures from `results/` that are referenced in `chapters/*.tex`.
- Run `python find_referenced_figs.py` before removing any figure file.

## Git Safety
- Always work on a feature branch off `main`; open a pull request back into `main`. Do not push to `clean-thesis` — it has been stalled since 2026-07-18 and `main` is the sole source of truth for `chapters/**`.
- Never force-push (`git push --force`) without explicit user approval.
- Commit message must describe what changed (e.g., "fix: resolve undefined references in ch04").