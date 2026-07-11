# Product Context — why this exists

## The thesis narrative
Classical relays (Amplify-and-Forward, Decode-and-Forward, Viterbi/MLSE) are optimal or near-optimal when the channel is **known** and **memoryless**. The thesis's central claim, built up experiment by experiment (E1–E8 in the main thesis, plus the Chapter 7 / E6 addendum), is:

- Under **known, memoryless** channels: classical relays are hard to beat (control condition).
- Under **unknown channel with memory** (ISI, nonlinear bias): classical relays hit a hard performance floor (e.g., ~0.25 BER for 3-tap ISI); learned relays (MLP, Viterbi-MLSE with estimated/genie CSI) close the gap or beat the floor entirely.
- Under **unknown but memoryless** channels (unknown gain, unknown I/Q imbalance): classical relays remain robust — proving it's **memory**, not mere unknownness, that breaks them. This is the falsification control (E6_FLAT).

## Why relaynet (the framework) matters
Early Chapter 7 work was done as quick standalone scripts to iterate fast. For thesis integrity, every number reported must trace back to the same reusable, tested framework (`relaynet`) used elsewhere — not one-off scripts. Porting E6 into `relaynet` is about **reproducibility and consistency of the SNR convention, channel models, and relay implementations** across all chapters.

## Who "uses" this repo
Effectively a single user (the thesis author) plus AI coding assistants (Cline previously, Claude Code now) collaborating across many sessions with no shared memory — hence the need for `.clinerules/` (Cline rules), `CLAUDE.md` (Claude Code rules), and this `memory-bank/` (persistent project state) so any assistant picking up the repo cold can reconstruct context fast.
