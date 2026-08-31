# Claude Code — Project Rules

This repo is an M.Sc. thesis ("Deep Learning Architectures for Two-Hop Relay Communication") plus its supporting simulation framework, `relaynet`. Two other rule sources already exist and apply alongside this file — read them first:

- **`context.md`** (repo root) — orientation pointer, read this first.
- **`memory-bank/`** — persistent project state (active work, progress checklist, architecture, tech gotchas). Read `memory-bank/activeContext.md` and `memory-bank/progress.md` at the start of every session before making changes.
- **`.clinerules/*.md`** — detailed rules for the thesis-writing side (LaTeX structure, citation format, chapter/appendix content boundaries, scientific-integrity constraints). These predate this file and remain authoritative for anything touching `chapters/*.tex`, `thesis_tau.tex`, `references.bib`, or `results/`.

## Scope boundary
- If a task touches `relaynet/**` or `e6_*_ported.py` / `e6_*_enhanced.py` (Python simulation code): this file + `memory-bank/` govern.
- If a task touches `chapters/**`, `thesis_tau.tex`, `references.bib`, or anything under `results/`: `.clinerules/*.md` governs — especially `90-safety.md` (never alter numerical results/figures/conclusions without explicit instruction and new data).
- **`main` is the sole authoritative branch for `chapters/**`.** The old `clean-thesis` branch (previously named here as authoritative) has been stalled since 2026-07-18 and is superseded by fact: `main` contains its entire history plus everything since. Always branch off `main` for thesis work and merge back through a pull request; do not push to `clean-thesis`. `.clinerules/90-safety.md` and `00-general.md` have been updated to match. See `memory-bank/techContext.md` gotcha #5 for the historical incident (now resolved) that this policy supersedes.

## Working agreement (this session, carried forward)
- Develop on the assigned feature branch (currently `claude/porting-md-file-l6xzsr`); never push elsewhere without explicit permission.
- Do not create a pull request unless the user explicitly asks for one.
- Follow the SNR convention documented in `memory-bank/techContext.md` exactly (γ = 10^(SNR_dB/10)) — this is load-bearing across every chapter's results, not just E6.
- New relays/channels for `relaynet` should follow the interface patterns in `memory-bank/systemPatterns.md` (`.process()` for relays, callable `channel(signal, snr_db)` for channels) rather than inventing new conventions.
- Simulation result numbers are scientific claims. Never fabricate, adjust, or silently drop Monte Carlo trials to hit an expected number — if a result doesn't match a spec (e.g., the E6_FLAT control gap ≤0.0036 target), report the discrepancy plainly rather than tuning until it matches.
- `/tmp/` outputs from this container do not persist across sessions. Anything meant to survive must be committed into the repo.
- **Before changing any published number, run `python provenance_audit.py`.** It links each experiment to its script, its committed data and the commit that produced it, and fails when data is uncommitted or older than the script that generates it. Both conditions were silently true for all five E6 datasets for ten days, during which the thesis reported three-seed results backed by single-seed data, and a later "correction" moved a table *towards* the stale data. Add a row to `REGISTRY` whenever a new experiment is added.
- Experiment scripts must write their results **into the repository**, never `/tmp/`. The E6 scripts wrote to `/tmp/`, so their output reached the repo only by manual copy; when that copy was forgotten the published numbers and the committed data diverged with nothing to detect it.
- Keep `memory-bank/activeContext.md` and `memory-bank/progress.md` current — update them as the last step of any substantive change, not as an afterthought.
- **`thesis/main.pdf` must never lag its sources.** Any commit that changes what the thesis renders — `thesis/chapters/**`, `thesis/main.tex`, `thesis/chapters/references.bib`, or a figure under `results/` that a chapter includes — must rebuild the PDF with `latexmk -xelatex` and commit the rebuilt `thesis/main.pdf` **in that same commit**. The PDF is the deliverable, so a commit that updates the LaTeX and leaves the PDF behind ships a document that does not contain the change. Report the page count whenever it moves. Commits that touch only Python, `results/*.json`, `memory-bank/**` or rule files do not re-render anything and need no rebuild.

## Repo-root clutter warning
The repo root contains ~150+ files unrelated to any single task (thesis build scripts, multiple `thesis*.{md,tex,docx,pdf}` variants, one-off `_*.py` processing scripts). Before assuming a root-level `.py` file is relevant, check `memory-bank/progress.md` and `memory-bank/systemPatterns.md` for the current file-naming conventions (`e6_<name>_ported.py`, `e6_<name>_enhanced.py`).
