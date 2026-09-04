# Thesis — Overleaf project (generated)

The M.Sc. thesis *Deep Learning Architectures for Two-Hop Relay Communication*,
laid out as an Overleaf project: `main.tex` at the root, the chapters it
includes, the figures those chapters use, `chapters/references.bib`, the bundled
fonts and `hebcal.sty`.

## Do not edit this tree

It is **generated**. The source is `thesis/` in the `Gilzuk/relaynet2`
repository, and `scripts/overleaf_sync.py` there rebuilds this tree in full on
every publish — anything changed here directly is overwritten the next time.

Overleaf is the compile-and-share surface; `thesis/` in git is the source of
truth. To change the document, change it there and republish. The publish step
refuses to overwrite commits it has not seen, so an edit made here blocks the
next publish rather than vanishing silently — but it still has to be ported
back into `thesis/` by hand.

Variant: **clean** (`clean` has the inline `\REV{...}` revision records
stripped; both render an identical PDF, since `\REV` discards its argument).

## Compiling

XeLaTeX, not pdfLaTeX — `fontspec` and `polyglossia` (the Hebrew abstract)
require it. See `OVERLEAF.md` for the font and package-order notes.
