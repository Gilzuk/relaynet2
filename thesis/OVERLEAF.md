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

## Package order (do not rearrange)
`polyglossia` is loaded at the very **end** of the preamble, right before
`\begin{document}`. It pulls in `bidi` (for the Hebrew abstract), and `bidi`
errors out for every package loaded after it. Those errors are survivable in a
manual `xelatex` run but make **latexmk — which Overleaf uses — stop rerunning**,
leaving every cross-reference and citation as `??`. If you add a package, put it
*before* the Language & Hebrew block at the bottom.

## Notes
- `hebcal.sty` is **no longer loaded**: `\usepackage{hebcal}` is commented out
  in `main.tex`. The stub existed to disable polyglossia's Hebrew-calendar font
  so the build would not depend on the `othello` MetaFont font; with the package
  out, nothing requests that font in the first place. The file is still in
  `thesis/` should it be needed again, but it no longer travels with the project
  — the bundle carries whichever local `.sty` files `main.tex` actually loads,
  and right now that is none.
- Figures live in `results/` and are found via `\graphicspath`.
- Bibliography: `chapters/references.bib`, IEEEtran style, via bibtex.
