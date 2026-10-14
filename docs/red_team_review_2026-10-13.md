# Red-team review of the thesis (main `4bac934`), 2026-10-13

**What this is.** This is an adversarial review of the merged thesis, done before the author approves
submission. The committee audit earlier today checked consistency. This review instead takes the
positions a hostile examiner would take, and tests each one against the data and code:

- statistics;
- baseline strength;
- train/test conditions;
- information asymmetry;
- reproducibility;
- figure provenance;
- citations;
- code-versus-text agreement;
- claim language;
- significance of the contribution.

Each finding is labelled **confirmed** (checked against committed data or code) or **checked and
cleared** (the attack failed). **Nothing in the thesis has been changed.**

**Outcome in one paragraph.** No finding overturns a headline conclusion. Two findings narrow a
stated claim, R1 and R2. R1, on the blind regime, is the more consequential. Five findings ask for a
sentence of framing, a caveat or a provenance fix. Four are minor. The thesis is unusually careful
about overclaiming. Its weak points are in the statistics behind two specific comparisons, not in the
language.

## High: these change a stated result

### R1. The blind-regime lead over CMA is resolved only at 0–12 dB, not at the 20 dB point the thesis cites (confirmed)

**Where.**
- Table 6.1, layer 4: "Learned relay narrowly leads. At 20 dB the MLP reaches 0.00262 against CMA's
  0.00329".
- Sec. 6.4 conclusion, which quotes intervals of ±0.0201, ±0.0026 and ±0.0078 at 8 dB.
- Ch. 8 finding 8: "narrowly leads the evaluated CMA implementation".
- The heading of App. 9.5.4: "CMA converges where decision-directed MLSE does not".

**What is wrong.** `e6_blind_ported.py` stores, for every relay, the interval
1.96·std/√150 over 150 columns, that is 3 trained MLP instances × 50 trials. Two problems follow:

- **Classical relays.** Their trial seed (`8000*si + tr`) does not depend on the MLP instance. Their
  150 columns are therefore three identical copies of the same 50 channel draws, and their intervals
  are √3 too narrow.
- **MLP.** Its three trained instances are pooled, not treated hierarchically.

Sec. 6.4 describes the correct rule ("classical-method intervals use those 50 draws; neural
uncertainty is assessed across trained instances") but quotes the pooled numbers.

**Evidence.** `e6_blind_intervals.py` → `results/e6_blind_intervals.json` reruns the identical
protocol with the same seeds. Every stored mean reproduces exactly.

| SNR (dB) | MLP − CMA | Paired over 50 draws | MLP better on | Correct intervals (stored) |
|---|---|---|---|---|
| 0–12 | −0.006 to −0.024 | p ≤ 0.003 | 30–38 of 50 | at 8 dB: MLP ±0.0046 (±0.0026), CMA ±0.0140 (±0.0078), DD-MLSE ±0.0361 (±0.0201) |
| 14 | −0.0060 | p = 0.63 | 20 of 50 | |
| 16 | −0.0054 | p = 0.40 | 25 of 50 | |
| 18 | −0.0017 | p = 0.84 | 22 of 50 | |
| 20 | −0.00067 ± 0.00087 | p = 0.55 | **19 of 50** | |

At 20 dB, CMA beats the MLP on 31 of 50 draws, yet averages worse. Its mean is therefore set by a
minority of much worse draws. That pattern fits occasional non-convergence, which also undercuts the
App. 9.5.4 heading.

**Suggested fix.**
- Say the MLP leads CMA at 0–12 dB. A good example is 8 dB, 0.0955 against 0.1182, which is resolved.
- Say the lead is not resolved at 14–20 dB, where CMA is better on most draws and loses on average
  through a minority of failed draws.
- Use 8 dB, not 20 dB, as the Table 6.1 example.
- Correct the three quoted intervals.
- Retitle App. 9.5.4.
- Add a verifier check against `results/e6_blind_intervals.json`.

### R2. The H1 margin is quoted against the weaker of the two conventional AFs (confirmed)

**Where.** Both abstracts ("the MLP's 0 dB margin is only 0.007"), Sec. 5.2, Sec. 7.1.1, Ch. 8
finding 1 and the Table 8.1 H1 row.

**What is wrong.** `results/af_block_and_conventional.json` stores two conventional AFs. The thesis
reports only the variable-gain one. The fixed-gain AF does better at 0–2 dB.

| 0 dB | Fixed-gain AF 0.3384 | Variable-gain AF 0.3418 |
|---|---|---|
| MLP margin (MLP 0.3350, 95% interval ±0.0030) | **+0.0034**, just outside the interval | +0.0068 |
| VAE margin (0.3359) | +0.0025, **inside** its interval | +0.0059 |
| Hybrid margin (0.3329) | +0.0055 | +0.0089 |

At 2 and 4 dB every feedforward relay leads both AFs by at least 0.012, so H1 itself survives. The
quoted 0 dB margin, however, is double the margin against the stronger evaluated AF. The intervals
also cover Monte Carlo noise of a single checkpoint, not training variance.

**Suggested fix.** Quote the stronger conventional AF at each SNR, or both: "0.003–0.007 depending on
the AF gain rule". State that the interval excludes training variance. This touches the English and
Hebrew abstracts, Ch. 5, Sec. 7.1.1, Ch. 8, Table 8.1, and the verifier check `prose:af-conventional`.

## Medium: framing, caveats and provenance

### R3. The matched coded comparison gives block DF information the learned relays do not get (confirmed)

Contribution 3 and Ch. 8 say decoding at the relay "outperforms the learned relays at every tested
SNR" with matched metrics. The learned relays are denied the same information:

- the matched relay decoder weights each symbol by |h₁|²;
- the learned relays receive no channel gain.

Sec. 5.5.2 does say that "a learned relay given |h₁|" was not evaluated. The contribution statement
and Ch. 8 do not. The coded comparisons therefore test mismatched decoding against blind learning,
and then matched decoding against blind learning. Neither tests learning against decoding with equal
information.

**Suggested fix.** Add one clause in Contribution 3 and in Ch. 8's coded paragraph.

### R4. The canonical learned relays never see the deployment channel, which limits what H1–H4 measure (confirmed; disclosed in part)

Every canonical learned relay is trained on BPSK ±1 over real AWGN. It is then applied per axis to
zero-forced QPSK/Rayleigh samples, which differ in two ways:

- the signal is ±1/√2, a factor √2 smaller;
- the noise is heavy-tailed.

Sec. 4.3.3 and Remark 4.1 say so. The H4 equal-budget comparison therefore measures transfer from
AWGN, not fit to the channel under test. The sequence models' 0 dB BER (0.401–0.407) is almost exactly
that of the block-normalized AF (0.4076). That fits near-linear behaviour on inputs unlike those they
were trained on. Sec. 5.2 offers only two readings of the sequence-model gap: no temporal structure,
and the training protocol.

**Suggested fix.** Name the train/deploy mismatch as a third reading in Sec. 5.2, and as a limitation
of H1–H4 in Sec. 7.4.

### R5. H5 is stated against baselines that are known to fail (judgement)

H5 says the MLP improves on AF and zero-threshold DF on ISI. The thesis itself derives that those
relays have a 0.25 error floor on the chosen taps, so an examiner can call H5 close to tautological.
The substantive evidence is all present but sits behind the strawman:

- the 0.26–2.53 dB gap to matched MLSE;
- the pilot crossover;
- the MMSE and CMA comparisons;
- the scalar-MAP control.

**Suggested fix.** In Table 8.1's H5 row and Sec. 7.6, lead with the MLSE gap and the pilot and
blind results (corrected per R1). Keep AF and DF as context.

### R6. The canonical spreads in Table 9.6 carry no cross-machine caveat, unlike Sec. 6.4.2 (judgement)

Sec. 6.4.2 now reports that three-seed spreads of trained sequence models change between machines
running identical software. The Transformer's seed moved by 0.89 dB. Table 9.6 supports H4's verdict
and the claim that the Transformer spread is "five times" the H4 effect. It was produced on one
machine with three seeds, yet carries no such caveat. Separately, the canonical result files predate
the repository and cannot be regenerated (Sec. 4.3.3).

**Suggested fix.** Add one sentence to the Table 9.6 caption or Sec. 7.6. Optionally, rerun
`seed_spread_architectures.py` on a second machine. That is long, because the sequence models train
for tens of minutes each.

### R7. Fig. 6.7(b) and the "30–90× faster" claim have no committed data or generator (confirmed)

- The panel image `results/e6_right_clean.png` has no generating script in the repository.
- `e6_complexity_ported.py` writes `e6_complexity_ported_results.npy`, which was never committed.
- Neither appears in `provenance_audit.py`.

The claim is hedged as implementation-specific, but it cannot be reproduced from the repository.

**Suggested fix.** Run `e6_complexity_ported.py`, commit its output, redraw panel (b) from it and
register it, or drop the panel and the ratio.

## Low

- **R8. Table 6.6 learned-relay column.** It takes the best configuration and the best seed. With the
  median seed, the lead over the best MMSE equalizer is 2.4, 1.4 and 1.8 dB, not "1.8 to 2.4". The
  QPSK value is about 0.4 dB optimistic. The conclusion is unchanged.
- **R9. Bibliography [34].** MambaCSP has since been published, at IEEE Globecom 2026 and in an
  ICML 2026 workshop. The entry could cite the venue.
- **R10. Table 5.11.** It prints "50–63× faster" while the text says MLP timings are too small for a
  precise speedup claim. Give ranges with that caveat, or drop the column.
- **R11. Pilot-study figures.** Table 6.1 quotes Viterbi-est at 0.0283 on 200 pilots, while Sec. 6.3
  quotes 0.0285 at 800. Both are likely correct, but side by side they look non-monotonic. Quote one
  pilot count in both places.

## Checked and cleared (attacks that failed)

| Attack | Result |
|---|---|
| Table 5.3's MLP-3K equals Table 5.2's MLP-169 to four decimals at 8–20 dB (copied column or evaluation bug?) | The per-trial arrays differ. The agreement is common random numbers plus rounding, e.g. 0.12291 against 0.12292. |
| Table 9.5's nonlinear-bias spread is uniformly about 1.23 dB (target-set artefact?) | Genuinely bimodal training outcomes (−1.63 or −2.86 dB). All seeds reach all targets. Sec. 7.2.1 notes the two clusters. |
| Train/test leakage in the Chapter 6 studies | None found. Training channels use separate seeds (99, 101, 42 and others) that never coincide with evaluation trial seeds. The fixed-ISI MLP is trained on the test filter, which is disclosed. |
| Pilot-sweep intervals (the A1 fix) | The classical intervals are computed over 50 unique trials, which is correct. "Resolved at 20 and 5, not at 10" stands. |
| Composite-study intervals | Same triplication as R1, but the thesis already calls those bands descriptive. |
| Code against the methods text | Sinusoidal positional encoding, Mamba-2 initialization log(1..N), the 5 dB Hybrid threshold and the canonical training budgets all match. |
| Fabricated or misattributed citations | [34] MambaCSP (Djuhera, Gacanin, Boche; arXiv 2604.21957) exists. So do [15] Akdemir et al. (whose official title contains the typo "Leaning", marked [sic]), [17] Karanov et al. and [20] Bergel. |
| Overclaiming language | No "novel", "first", "prove" or "state-of-the-art" claims. The one "clearly outperform" refers to the block AF and is immediately qualified. |
| Undisclosed statistical choices | No multiplicity correction (disclosed). The 30-trial matched budget was fixed after inspection (disclosed). Best-of-three seeds in Tables 6.6 and 6.7 (disclosed). |

## Examiner pre-mortem: the ten hardest questions

1. *"Your main AF baseline has infinite-mean output power. Isn't H1 a win over a broken baseline?"*
   Sec. 4.1.3 and Sec. 7.1.1 answer this. Quote the stronger conventional AF (R2).
2. *"Your learned relays never saw QPSK or Rayleigh in training. What does H4 measure?"* See R4.
3. *"Isn't H5 trivially true against memoryless relays on ISI?"* See R5. The MLSE, pilot and blind
   evidence is the answer.
4. *"Is the blind-regime advantage over CMA significant?"* Only at 0–12 dB (R1).
5. *"Matched MLSE beats your relay and is cheaper at L = 3. Why would anyone deploy it?"* Ch. 8 O1
   and O2 answer honestly: only where the channel model is unavailable or pilots are scarce.
6. *"Can I regenerate Table 5.2?"* No (Sec. 4.3.3), and seeded retraining is exact only on one
   machine (R6).
7. *"Is the coded reversal about learning against decoding, or about who gets |h|?"* See R3.
8. *"Three seeds for a spread?"* Disclosed, and the rerun shows the spreads are unstable (Sec. 6.4.2).
9. *"Best-of-three in Tables 6.6 and 6.7?"* Disclosed as optimistic. R8 quantifies it.
10. *"What is new?"* Sec. 2.2.4 claims a measured comparison, not a new principle. The rigour and the
    scoping are the contribution. An examiner may still judge it modest, but it is not overclaimed.

## Data and scripts added by this review

| Script | Output |
|---|---|
| `e6_blind_intervals.py` | `results/e6_blind_intervals.json`, `results/e6_blind_intervals_console.txt` |

It is registered in `provenance_audit.py`; after the fixes below it backs the Sec. 6.4 intervals, Table 6.1 layer 4 and Ch. 8 finding 8. The other checks used committed data
(`results/af_block_and_conventional.json`, `results/mlp_min_size_all_channels.json`,
`results/modulation/qpsk_rayleigh.json`, `results/normalized_3k/3k_rayleigh.json`) and are
reproducible from the commands given in each finding.

**Page budget.** The thesis is at 120 of 120 countable pages. R1 and R2 rewrite existing sentences.
R3–R6 add about one sentence each and would need offsets.

## Outcome (2026-10-14, branch `fix/red-team-findings`)

Applied in one PR into `main`, which supersedes this report-only PR (#177). Verifier 732 cells
(24 new), 0 inconsistencies; provenance clean; 347 tests; 120 countable pages (136 PDF pages,
unchanged). Wording elsewhere in Chapters 5, 7 and 8 was tightened to hold the page budget.

| Finding | Fix |
|---|---|
| R1 | Sec. 6.4, Table 6.1 layer 4, App. 9.5.4 heading, Ch. 8 finding 8, Sec. 7.6 and Table 8.1 H5 say the lead over CMA is resolved from 0 to 12 dB (Wilcoxon p ≤ 0.003) and not from 14 to 20 dB (p ≥ 0.40; better on 19 of 50 draws at 20 dB). Layer 4 now quotes the 8 dB pair (0.0955 against 0.1182). The intervals are the corrected ones (MLP ±0.0046 hierarchical, CMA ±0.0140, DD-MLSE ±0.0361 at 8 dB). |
| R2 | Both abstracts, Ch. 5, Sec. 7.1.1, finding 1 and Table 8.1 H1 quote a 0.003–0.007 margin depending on the gain rule; Sec. 7.1.1 notes that 0.003 lies just outside the MLP's Monte Carlo interval, which omits training variance, and inside the VAE's. The Hebrew abstract number changed and is re-drafted for the author (`thesis/HEBREW_ABSTRACT_SYNC.md`). |
| R3 | Ch. 1 contribution 3 and the Ch. 8 coded paragraph state that the matched relay decoder uses \|h₁\|², which the learned relays do not get, and that a learned relay with that input was not evaluated. |
| R4 | Sec. 7.1.1 states that H1–H4 measure transfer from the BPSK/AWGN training distribution; Ch. 5 adds it as a third reading of the sequence-model gap. |
| R5 | Sec. 7.6 and Table 8.1 H5: AF and zero-threshold DF are defeated analytically by the chosen taps, so the informative evidence is against model-based detectors. |
| R6 | Table 9.6 caption: three seeds on one machine, with the cross-machine pointer to Sec. 6.4.2. |
| R7 | `e6_complexity_ported.py` panel (b) now follows the Table 5.11 protocol (single-threaded, one warm-up, median of seven) and saves its settings; its output is committed, and `scripts/plot_e6_timing.py` draws Fig. 6.7(b) from it. The speedup is 58–82×, replacing "30–90×", in the caption, Sec. 6.5 and the README. A first, multithreaded run without warm-up gave 4–16×; the ratio depends on threading and machine, and the text says so. |
| R8 | Table 6.6 caption adds the median-initialization lead, 1.4 to 2.3 dB (the report's 2.4 was a rounding slip; the value is 2.346). |
| R9 | **Not applied.** The Globecom 2026 publication could not be confirmed before the conference (December 2026), so the arXiv entry stays. |
| R10 | The Table 5.11 follow-up says the MLP speedup ranges are coarse because the timings are sub-microsecond. |
| R11 | Sec. 6.3 quotes the LS-plus-MLSE payload BER at 200 pilots (0.0283), as Table 6.1 does. |

