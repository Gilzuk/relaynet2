# Khina-checklist pass over the thesis, 2026-10-17

**What this is.** A final pass over the whole thesis on branch `fix/khina-round2` (PR #183), made after
Dr. Khina's second round. Neither the committee audit nor the red-team review of 13 October caught his
round-2 points. This pass uses the classes of criticism from both of his rounds as a checklist and
reads every chapter against it.

**How it was done.**
- The rendered PDF was read chapter by chapter: the abstract, Chapters 1–8 and Appendices 9.2–9.6.
- Every claim of difference, equivalence or scope was checked against the committed data.
- Where the thesis declares a test, that test was applied.

**Results.**
- **Findings and fixes.** 22 findings. 19 are fixed in commits `6b9160d` (code and data) and `d0f4073` (thesis). Findings 20–22 were subsequently resolved in the local independent follow-up below.
- **Verifier.** 873 cells, 0 inconsistencies. A paired-test check was added, and the Table 6.6 checks now measure every column against one common reference.
- **Tests and build.** 351 tests pass. The PDF builds with 0 errors and 0 overfull boxes, at 120 countable pages.

## The checklist

| Class | Question asked of every passage | Source in Khina's feedback |
|---|---|---|
| K1 | Are compared quantities measured on the same basis: targets, reference curve, budget, seeds, side information, aggregation? | Round 2, point 2. Round 1, comments 3, 4 and 8 |
| K2 | Does each claim's scope match the variants actually evaluated? | Round 2, points 3 and 6. Round 1, comments 4 and 8 |
| K3 | Are corrected comparisons presented as principal and superseded ones as diagnostics? | Round 2, point 5 |
| K4 | Is the configuration of every reported run known, and its uncertainty stated? | Round 2, point 1. Round 1, comment 7 |
| K5 | Does any statement contradict another one elsewhere? | Round 2, point 6 |
| K6 | Are the formal statements correct? | Round 2, point 4. Round 1, comment 1 |
| K7 | Are the baselines strong, and is the information each relay receives stated? | Round 1, comments 3, 4 and 9 |
| K8 | Are cost and delay counted by stated conventions? | Round 1, comments 5 and 6 |
| K9 | Do captions and figures agree with the text and the data? | Round 1 presentation notes. Round 2, point 6 |
| Rule | Are script names cited as references rather than written into the text? | The author, 2026-10-17 |

## Findings

| # | Class | Where | Finding | Outcome |
|---|---|---|---|---|
| 1 | K1, K5 | Sec. 5.2 conclusion, Sec. 5.2.1, Fig. 5.1 caption | The text said the MLP, Hybrid and VAE "show no difference from DF detectable … from 4 dB upward, nor from each other". The thesis's own paired test (Wilcoxon, ten shared trials) disagrees: DF is ahead at 7 of 11 SNRs against the MLP and against the VAE, and at 1 against the Hybrid, and the Hybrid is ahead of the VAE at 0–12 dB. No learned relay is ever resolvably better than DF. All differences are at most 0.0035 | Fixed. The text gives the paired results from a new script and results file (ledger C6). H2's verdict is unchanged |
| 2 | K1 | Sec. 5.2.1, Fig. 5.1 | "The sequence models converge by 16 dB." In fact their gap to DF is under 0.002 from 16 dB but still resolved at 20 dB for the Transformer and Mamba-S6 | Fixed |
| 3 | K1 | Sec. 5.4 | "No evaluated model detectably better than the 169-parameter MLP." The two Hybrids are ahead of it at a few SNRs, by at most 0.0021 | Fixed |
| 4 | K9 | Fig. 5.2 caption | "The feedforward group tracks DF across the full range." In fact it stays within 0.009 of DF, but DF is resolvably ahead of each at low SNR | Fixed. The caption also notes that "Mamba-3K" is Mamba-S6-3K |
| 5 | K1 | Table 6.6 | Each row compared penalties against two different MLSE curves: the equalizer columns used the equalizer run's own curve, the learned-relay column the size sweep's. The shift is up to 0.09 dB | Fixed. All columns now use the sweep's curve. ISI (BPSK) moves by −0.09 dB and the composite by about +0.02 dB. The lead becomes 1.8–2.3 dB (was 1.8–2.4); the median-initialization lead stays 1.4–2.3 dB |
| 6 | K1 | App. 9.5.6 | The QPSK equalizer per-target penalties were on the equalizer run's own curve | Fixed. On the common curve they move by 0.01–0.03 dB |
| 7 | K3 | Sec. 7.2.1 | The coded size-sweep sentence led with the unweighted penalties | Fixed. The matched result (at least +9.75 dB) now comes first |
| 8 | K3 | Ch. 8, closing paragraph | The coded summary led with the superseded unweighted finding | Fixed. The matched result leads, and the unweighted one is explained after it |
| 9 | K3 | Sec. 5.5.5 | The MCS staircase was stated without naming its metric. It is the unweighted one; the matched block-DF staircase differs | Fixed. Both staircases are given, the matched one first |
| 10 | K3 | Ch. 2 (Sec. 2.2.4), Ch. 3 (H5 rationale), Sec. 6.4.1 | The joint trellis count was stated before the factorized one | Fixed. The factorized count now comes first and the joint count is a diagnostic |
| 11 | K5 | Sec. 7.1.3 | It said all training used 5, 10 and 15 dB. The Hybrid's sub-network trains at 2, 4 and 6 dB (Table 4.2) | Fixed |
| 12 | K5 | Sec. 4.3.4 | It said 347 tests in 24 modules, against App. 9.2's 351 in 25 | Fixed |
| 13 | K4, K5 | Sec. 5.5 setup | "All results use … 100 trials × 100,000 bits." This contradicts the 30-trial matched reruns and Table 5.8's 20M bits | Fixed |
| 14 | K5 | Sec. 5.5.1 | It read as if Tables 5.5, 5.6 and 5.9 themselves use the matched metric. This slip came from the script-reference edit | Fixed: "rerun with it" |
| 15 | K2 | Sec. 7.6, Table 8.1 | "The MLP passes LS-plus-MLSE below ten pilots." The only count below ten that was measured is five | Fixed: "leads … at five pilots" |
| 16 | K2 | Sec. 5.5.2 | The heading said the metric mismatch is sufficient to produce "the ordering". It was shown only for the MLP read-outs | Fixed: "the MLP ordering" |
| 17 | K2 | Ch. 3, Gap 1 | It cited Mamba and Mamba-2 as the compared state space models, without saying the implementations here are simplified | Fixed |
| 18 | K9 | App. 9.6 | A garbled "which … which" sentence, introduced by the script-reference edit | Fixed |
| 19 | Rule | Table 9.7 | The "Study (script)" column named scripts directly | Fixed. The column now holds ledger references |
| 20 | K3 | Fig. 6.7(a) | The figure plots the joint-trellis count. The caption states the factorized count and how far the plot overstates it | **Fixed locally.** Figure 6.7 now leads with factorized multiplication counts, with the joint trellis marked as a diagnostic |
| 21 | K3 | Tables 5.10, 5.12, 5.13 | The matched results are the last column, while the text leads with them | **Fixed locally.** Matched columns come first and obsolete winner emphasis is corrected; the table verifier follows the reordered columns |
| 22 | K1 | Table 9.5 caption | "On five of the nine channels the median spread is comparable to or larger than the entire effect of varying model size." Not rechecked after the comparable-target recomputation | **Fixed in the independent follow-up.** Removed the unverified five-of-nine comparison; the caption now describes configuration-dependent initialization variability without asserting an uncomputed size-effect count |

## Checked and found consistent

The following were checked against the data or the code and are left unchanged:
- **Chapter 5 numbers.** The DF closed-form agreement (worst 3.6%) and the 17.4% and 4.1% cross-model spreads of Sec. 5.3.
- **Chapter 6 numbers.**
  - The Table 6.7 leads (1.55, 3.14 and 0.32 dB).
  - The Table 6.1 layer outcomes.
  - The pilot-sweep values (0.0283, 0.0335, 0.0545 ± 0.0108, 0.0486, 0.1235).
  - The blind-regime resolution, 0–12 dB.
  - The MAC counts and crossovers of Sec. 6.5.1.
- **Other.**
  - The training settings in Sec. 4.2 against Table 4.2.
  - The window and context statements in Remark 4.1 and Sec. 4.4.
  - The matched-scope statements in Chapters 1, 5, 7 and 8.

## Data and scripts added by this pass

| Script | Output | Ledger |
|---|---|---|
| `canonical_paired_tests.py` | `results/canonical_paired_tests.json` | C6 |

The Table 6.6 recomputation needs no new data. The verifier recomputes the equalizer entries from the stored
equalizer BER curves and the size sweep's MLSE curve.

## Independent follow-up against PR head e58e94d (2026-10-17)

The latest PR already corrects the per-axis normalization, checkpoint tolerance, matched MLP scope, numerical architecture ordering and coded budgets. Those fixes were retained. The following remaining corrections are prepared locally; the author explicitly requires confirmation before any push.

- Use a block normalization operator in the formal objective, including the separate I/Q gains and the Hybrid's block switch. Limit the Gaussian smoothness argument to the uncoded slicer and exclude zero-energy and switching boundaries.
- Describe the optimizer options left at library defaults, with the missing historical library version explicit.
- Replace the nonexistent release-tag claim with the existing immutable experimental snapshot `e58e94d9d6136963ef56a9b3586f5b437d8075fd`.
- Reduce the two main unweighted diagnostic subsections to one paragraph each; preserve all five numerical tables and the soft-decision plot in Appendix 9.7. Bring matched constraint-length and high-SNR checks into the principal subsection.
- Remove the unverified five-of-nine size-effect comparison. A VAE spread bounded below by 0.156 dB cannot be claimed to be below 0.18 dB.
- Redraw the loss-versus-penalty plot from the stored BER curves using common targets, marking its three censored penalties as lower bounds. Remove the correlation interpretation that treated those bounds as measured penalties. No experiments were rerun.

An independent exact signed-rank calculation on integer bit-error differences confirms the canonical DF-ahead counts (MLP 7/11, VAE 7/11, Hybrid 1/11), with no learned relay ahead of DF. The table verifier checks 873 values with zero inconsistencies. The local test run passes 347 tests and skips four requiring the unavailable optional PyTorch/Mamba-2 module; this is distinct from the earlier author's full-environment test claim.

The final examiner-style pass also corrected the remaining causality and look-ahead claims, removed the censored-loss correlation from the discussion, qualified the regeneration and budget conclusions, and resolved both presentation choices. The five diagnostic tables and three plots are preserved in Appendix 9.7. See `docs/final_submission_review_2026-10-17.md` for the acceptance matrix and final verdict.
