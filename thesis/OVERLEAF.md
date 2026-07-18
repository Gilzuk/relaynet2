# Compiling this thesis on Overleaf

This project is self-contained and compiles on Overleaf with no extra setup.

## Quick start
1. Upload the project (or the `thesis_overleaf.zip`) to Overleaf.
2. **Menu → Settings → Compiler → XeLaTeX.**
   (The `% !TEX program = xelatex` line at the top of `main.tex` already
   requests this automatically, but set it in the menu if Overleaf doesn't
   pick it up.)
3. Main document: `main.tex`.
4. Recompile. First build runs `xelatex → bibtex → xelatex → xelatex`
   automatically via Overleaf's latexmk.

## Why XeLaTeX (not pdfLaTeX)
The thesis uses `fontspec` (real TrueType/OpenType fonts) and `polyglossia`
for the Hebrew abstract — both require XeLaTeX (or LuaLaTeX). pdfLaTeX will
not compile it.

## Fonts
All fonts are bundled in `fonts/` and loaded by explicit path in `main.tex`,
so the project does **not** depend on any font being installed on the
compile host:

| Role | Font | Files |
|------|------|-------|
| Main (serif) | Times New Roman | `fonts/TimesNewRoman-*.ttf` |
| Mono | Courier New | `fonts/CourierNew-*.ttf` |
| Hebrew | Arial | `fonts/Arial-*.ttf` |
| Sans / Hebrew sans | David CLM | `fonts/DavidCLM-*.otf` |

## Notes
- `hebrewcal.sty` is a local stub that disables polyglossia's Hebrew-calendar
  font (not needed here) so the build doesn't depend on the `othello`
  MetaFont font; it is written to work across MiKTeX / TeX Live / Overleaf.
- Figures live in `results/` and are found via `\graphicspath`.
- Bibliography: `chapters/references.bib`, IEEEtran style, via bibtex.
