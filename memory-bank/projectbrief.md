# Project Brief

## What this is
M.Sc. thesis project: **"Deep Learning Architectures for Two-Hop Relay Communication"**, Tel Aviv University (2026).

Two codebases live in this one repo:
1. **The thesis itself** — LaTeX source (`chapters/*.tex`, `thesis_tau.tex`), compiled via XeLaTeX, synced to Overleaf.
2. **`relaynet`** — a Python simulation framework for two-hop relay networks (channels, relays, modulation, nodes) used to generate the thesis's experimental results (BER curves, tables, figures).

## Core research question
Can learned (neural) relays outperform classical relays (AF, DF, Viterbi/MLSE) in two-hop wireless relay networks, especially under channel impairments classical relays can't handle (ISI/memory, nonlinearity) — while classical relays remain competitive or superior on memoryless impairments?

## Current active workstream: E6 porting
Chapter 7 (E6) originally ran as standalone scripts (`experiments-standalone/e6_*.py`). Task: port these into the `relaynet` framework so all Chapter 7 results are reproducible through the same Channel/Relay/Node classes used elsewhere in the thesis, per `experiments-standalone/PORTING.md`.

Branch: `claude/porting-md-file-l6xzsr`

See `activeContext.md` for what's in flight right now and `progress.md` for the experiment-by-experiment checklist.
