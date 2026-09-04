# Syncing the thesis to Overleaf

Overleaf expects `main.tex` at the **root** of its project. This repository keeps
the thesis in `thesis/`, alongside the simulation code, results and reproduction
suite — and `thesis/` carries more than the document:

| In `thesis/` | Belongs on Overleaf? |
|---|---|
| `main.tex`, the chapters `main.tex` includes, `chapters/references.bib` | yes |
| the figures those chapters include, `fonts/`, `hebcal.sty` | yes |
| `chapters/_ch0*.tex` — superseded drafts | no |
| `chapters/ak_response_appendix.tex`, `chapters/appendix_f_review.tex` — review-response appendices `main.tex` no longer includes | no |
| `*.aux`, `*.log`, `*.bbl`, `*.toc`, `*.xdv`, `main.pdf` — build artefacts | no |
| `CHANGELOG.md`, `RERUN_CHANGELOG.md`, `ak_comments.json`, `submission/`, `results_reference.html` | no |
| the inline `\REV{...}` fix records | no |

So `git subtree push --prefix=thesis`, which this repository used to do, put the
entire working trail into the authoring surface. The sync below publishes the
**document** instead.

---

## What gets published

`scripts/overleaf_project.py` resolves the project from `main.tex`'s include
graph — the chapters it actually `\include{}`s, the figures those chapters
actually `\includegraphics{}`, the bibliography, the bundled fonts, and any
`.sty` that lives in `thesis/` rather than on CTAN. Today that is **54 files**:
`main.tex`, 12 chapters, `references.bib`, 22 figures, 16 fonts, `hebcal.sty`
and `OVERLEAF.md`. Nothing else.

The `\REV{...}` fix records are stripped on the way out. They render as nothing
either way — `main.tex` defines `\REV` to discard its argument — so the published
project produces a PDF identical to `thesis/main.pdf`, page for page.

To see the resolved list without building anything:

```bash
make overleaf-show
```

## One-time setup

1. In Overleaf: **Menu → Git**, copy the URL (`https://git.overleaf.com/<id>`).
2. `git remote add overleaf https://git.overleaf.com/<id>`
3. Authenticate with your email as the username and a **git token** (Account
   Settings → Git integration) as the password — not your account password.
   `git config credential.helper store` avoids retyping it.

## Publishing

```bash
make overleaf-sync     # rebuild the overleaf-dist branch from thesis/
make overleaf-push     # rebuild it and push it to Overleaf as the project root
```

`overleaf-dist` is a generated branch whose **root is the Overleaf project**. It
is rebuilt in full from `thesis/` on every run, so it is never edited by hand and
never merged into `main`; it is a build output that happens to be a branch. Each
publish is one commit, its message naming the `thesis/` commit it came from.

There is no `overleaf-dist` in a fresh clone — `make overleaf-sync` creates it.

## The sync is one-way

Stripping annotations is lossy, so an edit made in the Overleaf editor cannot be
replayed back into `thesis/` automatically. Overleaf is the **compile and share**
surface; `thesis/` in git stays the source of truth.

To keep that from silently destroying anyone's work, `make overleaf-push`
**refuses** when Overleaf carries commits the branch does not, and prints them:

```
REFUSING TO PUSH: overleaf/master has 1 commit(s) this branch does not have
e87b742 Overleaf: tweak by the author
```

When that happens:

```bash
make overleaf-pull                      # fetch and list the Overleaf-side commits
git diff overleaf-dist FETCH_HEAD       # see exactly what changed there
```

Port those edits into `thesis/` by hand, re-run `make overleaf-push`, and the
document is consistent again. `--force` publishes over them instead, discarding
them — `python3 scripts/overleaf_sync.py --push --force`.

After porting Overleaf edits back, re-run `make verify`: edits can change table
cells, and the verifier is what catches a number drifting from its data source.

## The zips

Independently of the git sync, two archives are generated from the same
definition, so a zip and a sync can never disagree about what the project is:

```bash
make bundles
```

- `thesis_overleaf.zip` — annotated working copy, `\REV` fix records intact.
- `thesis_overleaf_clean.zip` — submission copy, annotations stripped.

Use these to hand someone a self-contained project, or to start a **new**
Overleaf project. Do **not** upload a zip over an existing Overleaf project:
Overleaf merges rather than replaces, so deleted files survive and shadow the new
ones — that is what produced a stale `ch05_experiments.tex` earlier in this
project.

## Compiler

XeLaTeX, not pdfLaTeX — `fontspec` and `polyglossia` (Hebrew abstract) require
it. `main.tex` carries `% !TEX program = xelatex`; set it in **Menu → Settings →
Compiler** too if Overleaf does not pick it up. See `OVERLEAF.md`, which travels
inside the project, for the font and package-order notes.
