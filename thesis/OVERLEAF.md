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
| Sans | David CLM | `fonts/DavidCLM-*.otf` |
| Mono | Courier New | `fonts/CourierNew-*.ttf` |
| Hebrew | David CLM | `fonts/DavidCLM-*.otf` |
| Hebrew mono | Courier New | `fonts/CourierNew-*.ttf` |

`fonts/` also carries Arial (`fonts/Arial-*.ttf`), which `main.tex` does not
load at present.

## Package order (do not rearrange)
`polyglossia` is loaded at the very **end** of the preamble, right before
`\begin{document}`. It pulls in `bidi` (for the Hebrew abstract), and `bidi`
errors out for every package loaded after it. Those errors are survivable in a
manual `xelatex` run but make **latexmk — which Overleaf uses — stop rerunning**,
leaving every cross-reference and citation as `??`. If you add a package, put it
*before* the Language & Hebrew block at the bottom.

## Notes
- `hebrewcal.sty` in the project root is a local stub, and `main.tex` loads it
  explicitly so that it travels with the project. Polyglossia's Hebrew support
  requires a package of that name; the stub takes its place, accepts the options
  polyglossia passes and defines the calendar commands as no-ops, since the
  thesis uses no Hebrew dates. Its header records why: on some installations the
  real package pulled in the `othello` METAFONT Hebrew-calendar font, which
  failed to build. Keep it next to `main.tex`.
- The older `hebcal.sty` stub is not used: `\usepackage{hebcal}` is commented
  out in `main.tex`, and the file stays in `thesis/` without travelling with the
  project.
- Figures live in `results/` and are found via `\graphicspath`.
- Bibliography: `chapters/references.bib`, IEEEtran style, via bibtex.
