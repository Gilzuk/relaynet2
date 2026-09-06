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

## Two routes to Overleaf

Pick one. Both publish the same `overleaf-dist` tree; they differ only in what
sits between git and Overleaf.

| | **A — Overleaf's git remote** | **B — a dedicated GitHub repo** |
|---|---|---|
| Setup | `git remote add overleaf https://git.overleaf.com/<id>` | create a repo, link it in Overleaf: **Menu → GitHub** |
| Publish | `make overleaf-push` | `make overleaf-repo`, then **Pull** in Overleaf |
| Needs | an Overleaf plan with git access | an Overleaf plan with GitHub sync |
| Overleaf edits come back | only into Overleaf's own git | pushed into the GitHub repo by Overleaf |

Overleaf projects have no branches — their git remote has only `master`, and the
GitHub integration syncs one branch. That is not a constraint on *this*
repository: the push maps our branch onto theirs by refspec
(`overleaf-dist:master`, or `overleaf-dist:main` for route B), so a local branch
is fine either way.

### Route B setup

The repository must have the **project at its root**, which is exactly what
`overleaf-dist` is. Create it empty — no README, no `.gitignore`, no licence;
anything auto-created becomes a commit the first publish would have to refuse or
overwrite.

```bash
# 1. create an EMPTY private repo on GitHub, e.g. <you>/relaynet2-thesis
# 2. point this repository at it
git remote add thesis-repo https://github.com/<you>/relaynet2-thesis
# 3. publish
make overleaf-repo
# 4. in Overleaf: Menu -> GitHub -> Link to GitHub, pick that repo
```

Thereafter `make overleaf-repo` republishes, and you press **Pull** in Overleaf.
Keep the repo **private**: it carries the full unpublished thesis.

## Publishing

```bash
make overleaf-sync     # rebuild the overleaf-dist branch from thesis/
make overleaf-push     # ... and push it to Overleaf's git remote     (route A)
make overleaf-repo     # ... or to the thesis-repo GitHub remote      (route B)
```

`overleaf-dist` is a generated branch whose **root is the Overleaf project**. It
is rebuilt in full from `thesis/` on every run, so it is never edited by hand and
never merged into `main`; it is a build output that happens to be a branch. Each
publish is one commit, its message naming the `thesis/` commit it came from, and
the history is append-only.

It is also mirrored to `origin` so the published state is visible from GitHub
without Overleaf credentials:

```bash
make overleaf-mirror                       # rebuild and push it to origin
python3 scripts/overleaf_sync.py --origin --push   # origin and Overleaf at once
```

A fresh clone gets it with `git fetch origin overleaf-dist`; `make overleaf-sync`
rebuilds it locally either way. The tooling that generates it lives on
`claude/overleaf-sync`, not on the thesis branch.

## The sync is one-way

Stripping annotations is lossy, so an edit made in the Overleaf editor cannot be
replayed back into `thesis/` automatically. Overleaf is the **compile and share**
surface; `thesis/` in git stays the source of truth.

To keep that from silently destroying anyone's work, both publish targets
**refuse** when they carry commits the branch does not, and print them. This
matters most on route B, where Overleaf's GitHub integration pushes editor
changes back into the repository itself:

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
