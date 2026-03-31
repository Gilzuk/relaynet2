# Feedback on `thesis.md`

## Overall assessment

The thesis is ambitious, technically rich, and unusually comprehensive. It demonstrates strong command of digital communications, relay theory, and modern deep learning architectures. However, in its current form it contains a significant number of internal inconsistencies that weaken credibility. Most are fixable through careful harmonization of:

- experimental counts,
- section/table/figure numbering,
- claims vs. reported results,
- terminology,
- parameter/window sizes,
- device/training descriptions.

From an examiner’s perspective, the central risk is not lack of substance, but that the manuscript sometimes reads as if multiple drafting stages were merged without final reconciliation.

## High-priority inconsistencies

### 1. Core hypothesis H1 is contradicted by the results

#### What you claim
In **Section 5.2**, H1 states:

> AI-based relay strategies achieve statistically significantly lower BER than both AF and DF at low SNR (0–4 dB)

In **Section 8.7**, the summary table says:

> **H1 Confirmed — all 7 AI methods beat AF and DF at 0–4 dB on all 6 channels**

#### What the results show
This is not supported by your own tables:

- **Rayleigh SISO, 0 dB**: DF = 0.245, all AI relays are worse or not clearly better.
- **MIMO SIC, 0 dB**: DF = 0.134, no AI relay beats DF.
- **MIMO MMSE, 0 dB**: only Mamba S6 slightly beats DF.
- **AWGN**: AI advantage is marginal and not universal across all AI methods.
- **Rician**: again not all AI methods beat DF.

#### Recommendation
Rephrase H1 to something like:

> Some AI relays can outperform AF and occasionally DF at low SNR under selected channels, especially AWGN and Rician, but this advantage is not universal across all channels.

### 2. Contradiction between abstract and results on “all channels”

#### Hebrew abstract
States:

> all neural relays outperform classical methods at low SNR on all channels

#### English abstract
States:

> all neural network relays outperform classical methods at low SNR (0–4 dB)

#### Results
Not true for Rayleigh and SIC, and not true for all neural relays.

#### Recommendation
Both abstracts must be revised to reflect the actual findings:
- AI relays show selective low-SNR gains,
- DF remains strongest and most consistent baseline,
- channel dependence is important.

### 3. Number of automated tests is inconsistent

You mention:

- **75 automated tests** in the Hebrew abstract
- **126 automated tests** in Section 6.5.4
- **126 automated tests** again in Appendix C

#### Recommendation
Choose one verified number and use it everywhere.

### 4. Simulation sample counts are inconsistent

#### In Hebrew abstract
You state:
- **100,000 bits per SNR**
- **10 repetitions**

This suggests 1,000,000 bits total, unless you mean 100,000 total across all repetitions.

#### In Section 6.5.1
You state:
- 10,000 bits per trial
- 10 trials
- 100,000 total bits per SNR

#### In Section 6.2.6 channel validation
You state:
- 20 trials of 50,000 bits
- 1,000,000 bits per SNR

#### In Section 7.1 note
You again state:
- 20 trials × 50,000 bits

#### In Section 8.7
You summarize:
- 100,000 bits per SNR point, 10 trials

#### Recommendation
Explicitly separate:
- **Channel validation experiments**: 20 × 50,000 = 1,000,000 bits/SNR
- **Relay comparison experiments**: 10 × 10,000 = 100,000 bits/SNR

Then ensure all abstracts and conclusions reflect that distinction.

### 5. Table numbering is broken

In the **List of Tables**:

- Table 7 is followed by **9**, **10**, **11**, **12**
- Table 8 is missing
- Later there is **11b**
- In the text, “Table 0” and “Table 0b” appear

#### Recommendation
Do a full renumber pass globally.

### 6. Figure numbering/list mismatch

There are several figure-list inconsistencies:

- List of Figures says Figure 27 is **16-QAM activation experiment on AWGN — tanh vs linear vs hardtanh**
- In the text, Figure 27 is **Combined modulation comparison on AWGN**
- Figures 28 and 29 then correspond to activation experiments

#### Recommendation
Rebuild figure numbering from the manuscript source after final ordering.

### 7. Section numbering error: duplicate 6.7.6

Under Methods → Modulation Schemes:

- **6.7.6** = 16-PSK
- later another **6.7.6** = I/Q splitting
- then **6.7.7** = 2D joint classification

#### Recommendation
Renumber as:
- 6.7.6 16-PSK
- 6.7.7 I/Q splitting
- 6.7.8 2D joint classification

### 8. QPSK and 16-QAM mapping tables contain impossible bit labels

#### In QPSK table
You list:
- 00
- 01
- 11
- **12**

#### In 16-QAM PAM table
You list:
- 00 → +3
- **12** → -1
- 01 → +1
- 11 → -3

#### Recommendation
Correct Gray mapping immediately. Likely intended entries are:
- 00 → +3
- 01 → +1
- 11 → -1
- 10 → -3

And the QPSK table should use 10, not 12.

### 9. Window size descriptions are inconsistent

Examples:
- Section 6.6 says all 3K models use window size 11
- Appendix D says all 3K models use **window 12**
- Table in 6.6 says MLP-3K increased window **5→11**
- Text says “All 3K configurations use a window size of 11”

#### Recommendation
Verify actual implementation and make all references consistent.

### 10. Device usage is inconsistent across sections

#### Section 7.9 complexity table
- Transformer: CUDA
- Mamba S6: CUDA
- Mamba2: CUDA

#### Appendix B
- Transformer: **PyTorch (CPU)**
- Mamba S6: **PyTorch (CPU)**
- Mamba2: **PyTorch (CUDA)**

#### Recommendation
State clearly:
- implementation framework,
- training hardware for reported experiments,
- whether CPU versions existed only for development.

### 11. Training sample counts are inconsistent

- Section 6.3 says for MLP: 25,000 samples
- Section 6.5.3 says: 25,000 supervised / 20,000 generative / 10,000 sequence
- Section 7.9 says: **50,000 training samples**
- Section 7.11 says: **50,000 samples**
- Section 7.15 says: 50,000 training symbols, 25 epochs

#### Recommendation
State per experiment family:
- baseline BPSK relay comparison sample count,
- normalized 3K sample count,
- modulation extension sample count,
- CSI experiment sample count.

### 12. Sequence model epoch counts are inconsistent

Examples:
- Section 6.3: Transformer/Mamba/Mamba2 trained for **100 epochs**
- Section 7.11: “100 epochs (200 for sequence models)”
- Section 7.15: 25 epochs with early stopping
- Section 8.3.1: 20 epochs benchmark

#### Recommendation
Clarify that epoch counts vary by experiment and summarize them in one table.

### 13. SIC naming is inconsistent

You use:
- SIC
- MMSE-SIC
- V-BLAST
- MMSE-ordered V-BLAST

#### Recommendation
Define once:

> In this thesis, “SIC” refers specifically to MMSE-ordered V-BLAST successive interference cancellation.

Then use that term consistently.

### 14. Inconsistent claims about “all nine relay strategies”

In several places you say all nine relays are compared for QPSK and 16-QAM, but later:
- Hybrid behaves anomalously because of routing
- some models fail to train in 16-class mode
- some results omit variants or use dashes

#### Recommendation
Differentiate:
- “evaluated”
- “successfully converged”
- “included in final ranked comparison”

### 15. Contradictory “best overall” model claims

Different sections imply different winners:
- Hebrew abstract: Mamba S6 and Mamba-2 achieve lowest BER on all channels
- Section 7.2: CGAN best at low SNR in AWGN
- Section 7.3: DF best on Rayleigh low SNR
- Section 7.5: MLP best at 0 dB MIMO ZF
- Section 7.6: Mamba S6 best at 0 dB MMSE
- Section 8.7: “best AI relay is channel-dependent”

#### Recommendation
Revise abstract and discussion to say:

> No single neural architecture dominates all channels; performance is channel- and regime-dependent.

## Medium-priority inconsistencies

### 16. “Without loss of generality” is misused

In Section 4.1.2 you write that half-duplex assumption:

> simplifies the analysis without loss of generality

This is too strong. Half-duplex is a restriction, not WLOG.

#### Recommendation
Replace with:

> simplifies the analysis while remaining representative of many practical relay systems.

### 17. Capacity discussion is insufficiently aligned with half-duplex model

The information-theoretic discussion appears to present capacity statements that may not align cleanly with the half-duplex two-hop model used later.

#### Recommendation
Clarify whether these are full-duplex / idealized results versus the actual thesis model.

### 18. Direct-link strategies reviewed despite no-direct-link system model

You define the model with no direct source-destination link, but review CF/incremental relaying strategies that rely on direct-link behavior.

#### Recommendation
Add one sentence clarifying these are reviewed for completeness but not simulated.

### 19. Statistical significance claims are stronger than the visible evidence

You repeatedly state significance across many channels/SNR points, but the manuscript provides little tabulated p-value evidence.

#### Recommendation
Either add p-value tables/appendix or soften wording.

### 20. BER resolution statement is somewhat unclear

The statement about minimum detectable BER and reporting zeros should be made more precise, especially since tables report small nonzero averaged BER values.

### 21. “Theoretical analysis” for MIMO MMSE/SIC is weaker than for AWGN/Rayleigh

For MMSE/SIC, the treatment is approximate rather than closed-form.

#### Recommendation
Call this “approximate analytical characterization” instead of fully parallel “theoretical analysis.”

### 22. Training time descriptions for Mamba S6 vary

Reported values differ across sections (36 min, 37 min, other numerical forms).

#### Recommendation
Either standardize or explicitly tie each value to a specific experiment.

### 23. Hybrid inference speed is surprisingly faster than DF

This may be true, but as presented it is counterintuitive and should be explained.

### 24. Software architecture appendix omits 16-PSK module

Appendix C lists BPSK/QPSK/QAM but not 16-PSK.

### 25. Software architecture appendix omits Transformer/Mamba/Mamba2 relay files

Given their prominence, their absence in the package tree is noticeable.

### 26. Appendix D contradicts text on window values

This should be fixed as part of the window-size harmonization.

### 27. “100% pass rate” should ideally be tied to a revision or experiment snapshot

## Lower-priority editorial issues

### 28. Terminology drift

You alternate among:
- deep learning-based
- AI-based
- neural network-based
- neural relay

#### Recommendation
Pick one primary term and use the others sparingly.

### 29. “Generative superiority” is overstated

In Section 7.14, this phrase is not appropriate for Mamba.

#### Recommendation
Use “representation capacity” or “modeling advantage.”

### 30. Sections 7.14 and 7.15 read as contradictory rather than progressive

Section 7.14 suggests CSI injection solves QAM16 Rayleigh; Section 7.15 shows it is detrimental for QAM16 overall.

#### Recommendation
Frame 7.14 explicitly as a preliminary result later refined by the broader study.

### 31. H7 appears in conclusions but not in original hypotheses

You mention it as emergent, but this should be made explicit.

### 32. Overuse of “all” where “some” or “most” is more accurate

### 33. Mixed spelling convention

Example: “generalisability” vs “generalizability.”

### 34. Notation drift: “QAM16” vs “16-QAM”, “PSK16” vs “16-PSK”

### 35. Some table captions mismatch actual content due to numbering drift

## Conceptual comments

### 36. The thesis is strongest when framed as a communications thesis, not an AI replacement narrative

Your strongest contribution is showing where neural methods help and where classical DF remains superior. That is a more mature and credible framing than universal AI superiority.

### 37. The 16-QAM story needs tighter narrative control

A stronger narrative would be:
1. I/Q splitting works for QPSK.
2. It fails structurally for 16-QAM.
3. Activation-aware retraining helps but does not fully solve it.
4. CSI is not the core fix for QAM.
5. Joint 2D classification is the actual structural solution.

### 38. Literature review is strong but occasionally broader than the thesis scope

Some tightening may improve focus.

## Recommended revision priorities

### Priority A: Must fix before submission
1. Correct H1 and all related abstract/conclusion claims.
2. Reconcile all test-count, sample-count, epoch-count, and window-size inconsistencies.
3. Fix all table/figure numbering.
4. Correct invalid bit labels (“12”).
5. Reconcile device/hardware descriptions.
6. Harmonize results summaries so they do not overclaim universal AI superiority.

### Priority B: Strongly recommended
7. Reframe Section 7.14 as preliminary and Section 7.15 as definitive.
8. Standardize terminology and notation.
9. Add explicit distinction between channel-validation and relay-comparison experiments.
10. Add a compact summary table of experimental configurations.

### Priority C: Stylistic/academic polish
11. Soften overgeneralizations.
12. Make the 16-QAM chapter more logically staged.
13. Reduce promotional wording like “generative superiority.”
14. Align software appendix with actual implemented models.

## Examiner-style summary

This is a substantively strong thesis with clear originality in comparative evaluation and insightful negative results. However, the manuscript currently suffers from nontrivial internal inconsistencies that could cause an examiner to question the reliability of the reporting, even where the underlying work appears solid. The main issue is not technical weakness, but insufficient final editorial reconciliation between experimental chapters, abstracts, appendices, and concluding claims.
