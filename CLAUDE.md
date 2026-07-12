# Claude Code — Project Rules

This repo is an M.Sc. thesis ("Deep Learning Architectures for Two-Hop Relay Communication") plus its supporting simulation framework, `relaynet`. Two other rule sources already exist and apply alongside this file — read them first:

- **`context.md`** (repo root) — orientation pointer, read this first.
- **`memory-bank/`** — persistent project state (active work, progress checklist, architecture, tech gotchas). Read `memory-bank/activeContext.md` and `memory-bank/progress.md` at the start of every session before making changes.
- **`.clinerules/*.md`** — detailed rules for the thesis-writing side (LaTeX structure, citation format, chapter/appendix content boundaries, scientific-integrity constraints). These predate this file and remain authoritative for anything touching `chapters/*.tex`, `thesis_tau.tex`, `references.bib`, or `results/`.

## Scope boundary
- If a task touches `relaynet/**` or `e6_*_ported.py` / `e6_*_enhanced.py` (Python simulation code): this file + `memory-bank/` govern.
- If a task touches `chapters/**`, `thesis_tau.tex`, `references.bib`, or anything under `results/`: `.clinerules/*.md` governs — especially `90-safety.md` (never alter numerical results/figures/conclusions without explicit instruction and new data).
- **`chapters/**` has a separate authoritative branch: `clean-thesis`** (per `.clinerules/90-safety.md`, "Always push to `clean-thesis` branch only"). Before treating any `chapters/*.tex` content as wrong/violating `.clinerules`, diff it against `origin/clean-thesis`'s history first — the `.clinerules` docs can be (and have been) stale relative to deliberate restructuring already done there. A mismatch may mean the rules doc needs updating, not the thesis. See `memory-bank/techContext.md` gotcha #5 for a concrete case where this went wrong and had to be reverted.

## Working agreement (this session, carried forward)
- Develop on the assigned feature branch (currently `claude/porting-md-file-l6xzsr`); never push elsewhere without explicit permission.
- Do not create a pull request unless the user explicitly asks for one.
- Follow the SNR convention documented in `memory-bank/techContext.md` exactly (γ = 10^(SNR_dB/10)) — this is load-bearing across every chapter's results, not just E6.
- New relays/channels for `relaynet` should follow the interface patterns in `memory-bank/systemPatterns.md` (`.process()` for relays, callable `channel(signal, snr_db)` for channels) rather than inventing new conventions.
- Simulation result numbers are scientific claims. Never fabricate, adjust, or silently drop Monte Carlo trials to hit an expected number — if a result doesn't match a spec (e.g., the E6_FLAT control gap ≤0.0036 target), report the discrepancy plainly rather than tuning until it matches.
- `/tmp/` outputs from this container do not persist across sessions. Anything meant to survive must be committed into the repo.
- Keep `memory-bank/activeContext.md` and `memory-bank/progress.md` current — update them as the last step of any substantive change, not as an afterthought.

## Repo-root clutter warning
The repo root contains ~150+ files unrelated to any single task (thesis build scripts, multiple `thesis*.{md,tex,docx,pdf}` variants, one-off `_*.py` processing scripts). Before assuming a root-level `.py` file is relevant, check `memory-bank/progress.md` and `memory-bank/systemPatterns.md` for the current file-naming conventions (`e6_<name>_ported.py`, `e6_<name>_enhanced.py`).
