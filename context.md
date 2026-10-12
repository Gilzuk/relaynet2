# Context — start here

This repo has no persistent memory between AI assistant sessions. Read these, in order, before doing anything:

1. **`memory-bank/activeContext.md`** — what's in flight right now, immediate next step.
2. **`memory-bank/progress.md`** — dated session notes at the top, then the original E6-porting checklist.
3. **`HANDOVER.md`** — the task chain of the latest review round (Dr. Khina, October 2026): every task, its PR, what was found, and the open author decisions. The point-by-point reply is `thesis/submission/reply_to_khina_2026-10.md` (draft, not sent); the thesis-side records are `thesis/CHANGELOG.md` and `thesis/RERUN_CHANGELOG.md`.
4. **`memory-bank/projectbrief.md`** and **`memory-bank/productContext.md`** — what this project is and why, if you need the bigger picture.
5. **`memory-bank/systemPatterns.md`** and **`memory-bank/techContext.md`** — how `relaynet` is structured, conventions to follow, and known gotchas so you don't rediscover them the hard way.

## Rules
- **`.clinerules/`** — pre-existing rules for the thesis-writing side of the repo (LaTeX structure, citation format, chapter/appendix boundaries, safety constraints on not altering results). Still authoritative.
- **`CLAUDE.md`** — Claude-Code-specific operating rules for this repo, layered on top of `.clinerules/`.

## Update discipline
Whoever (human or assistant) finishes a work session should update `memory-bank/activeContext.md` (what changed, what's next) and `memory-bank/progress.md` (move items between done/in-progress/not-started) before stopping. Treat stale memory-bank entries as a bug.
