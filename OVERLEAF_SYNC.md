# Working in Overleaf and in git at the same time

The obstacle is a structural mismatch. Overleaf expects `main.tex` at the **root**
of its project; this repository keeps the thesis in a `thesis/` subdirectory,
alongside the simulation code, results and reproduction suite. Pointing Overleaf
at the repository root would give you a project whose main document is buried one
level down, cluttered with ~100 MB of code and results that Overleaf has no use
for.

The fix is to sync only the `thesis/` subdirectory, so that **`thesis/` in git ==
the Overleaf project root**. Two ways to do that, below. Both need an Overleaf
plan that includes git access (Standard or Professional).

---

## Option A — git subtree against Overleaf's git remote (recommended)

Every Overleaf project has its own git remote. `git subtree` maps a subdirectory
of this repository onto that remote's root, which is exactly the mapping needed.
You keep one repository, full history, and no duplicated files.

### One-time setup

1. In Overleaf: **Menu → Git**, and copy the URL. It looks like
   `https://git.overleaf.com/abc123def456`.
2. Add it as a remote:

   ```bash
   git remote add overleaf https://git.overleaf.com/<your-project-id>
   ```

3. Authenticate. Overleaf asks for your email as the username and a **git
   token** (Account Settings → Git integration) as the password, not your
   account password. To avoid retyping it:

   ```bash
   git config credential.helper store   # or: osxkeychain / manager-core
   ```

### Day to day

```bash
make overleaf-push     # send thesis/ up to Overleaf
make overleaf-pull     # bring Overleaf edits back into thesis/
```

Which wrap:

```bash
git subtree push --prefix=thesis overleaf master
git subtree pull  --prefix=thesis overleaf master --squash
```

Note `master`: Overleaf's git remote uses `master`, regardless of what this
repository's default branch is called.

### Rules that keep it painless

- **Pull before you push.** If both sides changed, `git subtree pull` merges;
  pushing first will be rejected.
- **Commit locally before pulling.** Subtree operates on commits, not the
  working tree.
- Overleaf is the *authoring* surface; the reproduction suite, the verifier and
  the experiment code stay on the git side and never travel to Overleaf.
- After pulling, re-run `make verify` — Overleaf edits can change table cells,
  and the verifier is what catches a number drifting away from its data source.

---

## Option B — Overleaf's GitHub integration

Overleaf can link a project directly to a GitHub repository
(**Menu → GitHub → Link to GitHub**) and offers push/pull buttons in the UI.

The catch is that it syncs the **whole repository**, so the Overleaf project
would contain the code, the results tree and the reproduction suite, and you
would have to set the main document to `thesis/main.tex` in Overleaf's settings.
It works, but the project becomes slow and cluttered.

If you prefer this route, keep a **thesis-only repository**: publish the
`thesis/` subtree to a second GitHub repo and link *that* to Overleaf.

```bash
# one-time: create e.g. Gilzuk/relaynet2-thesis on GitHub, then
git remote add thesis-repo https://github.com/Gilzuk/relaynet2-thesis
git subtree push --prefix=thesis thesis-repo main
```

That repository's root is the thesis, so Overleaf is happy, and the code
repository stays clean. The cost is a second repository to keep in step.

---

## What not to do

- **Do not upload a zip over an existing Overleaf project.** Overleaf merges
  rather than replaces, so deleted files survive and shadow new ones — this is
  what produced a stale `ch05_experiments.tex` earlier in this project. If you
  must upload a zip, create a *new* project.
- **Do not commit Overleaf's build artefacts.** `.aux`, `.log`, `.out`, `.toc`
  and `.synctex.gz` are already in `.gitignore`; keep them there.
- **Do not edit the same file on both sides at once.** Subtree will merge, but
  LaTeX conflict markers inside a `longtable` are unpleasant to untangle.

---

## Regenerating the distributable bundles

Independently of syncing, the two Overleaf archives are generated from
`thesis/`, never edited by hand:

```bash
make bundles
```

- `thesis_overleaf.zip` — annotated working copy, with the inline fix records.
- `thesis_overleaf_clean.zip` — submission copy, annotations stripped.

Both contain only what `main.tex` actually compiles, so uploading either as a
**new** Overleaf project gives a clean, self-contained build.
