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

## File Safety
- Never edit `overleaf_thesis/` files directly — always edit `chapters/*.tex` and run `python sync_overleaf.py`.
- Never delete figures from `results/` that are referenced in `chapters/*.tex`.
- Run `python find_referenced_figs.py` before removing any figure file.

## Git Safety
- Always push to `clean-thesis` branch only.
- Never force-push (`git push --force`) without explicit user approval.
- Commit message must describe what changed (e.g., "fix: resolve undefined references in ch04").