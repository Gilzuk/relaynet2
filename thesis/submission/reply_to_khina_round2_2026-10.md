Subject: Re: Thesis — remaining points

Draft for Gil to send; not sent. Updated after PRs #185, #186 and #187 merged.
Reviewed submission: [`5eb34ab3ed01`](https://github.com/Gilzuk/relaynet2/tree/5eb34ab3ed01bc9df17498358656965af2fa916b).
[Revised thesis PDF](https://github.com/Gilzuk/relaynet2/blob/5eb34ab3ed01bc9df17498358656965af2fa916b/thesis/main.pdf).
Section/table numbers below refer to this PDF, including principal matched decoding in §5.5.1.

Dear Anatoly,

Thank you for the second round. I have addressed all six points, and the thesis has 120 Arabic-numbered pages from Chapter 1 through the appendices (136 physical PDF pages including front and back matter). The recheck you asked for in point 2 changes several published numbers, and I list each change below with its old and new value. The coded Mamba relay was not rerun with matched metrics, so the matched-decoding statements are now restricted to the two MLP read-outs that were.

**Additional training replications and seed sensitivity (17 October 2026).** The native and 3K canonical, sequence-memory and matched-coded repeats have now completed; the classical control arrays match exactly in those four hosted jobs, while the learned outputs provide additional training realizations. Numerical differences alone do not establish that the original measurements were incorrect. The new discussion subsection “Additional Training Replications and Seed Sensitivity” reports them separately from the historical BER measurements; derived comparisons and statistical summaries have been corrected where required. The native VAE–DF gap reaches 0.00640 at 4 dB, so the former 0.004 bound is explicitly historical. QPSK Transformer best penalty changes from 5.959 to 6.636 dB; Mamba2 spread changes from >3.282 to 0.973 dB, with its previously unsuccessful seed now reaching 1e-3. The sequence-memory best-of-three order survives, while some canonical pointwise ranks change. Fresh paired tests retain the matched block-DF advantage over both MLP read-outs at every tested SNR. An explicitly seeded full-budget MLP repeats exactly on one host; this does not recover the historical Torch state or prove cross-platform reproducibility. The minimum-size training grid was not rerun. The complete evidence is archived at [dc7c02c00cbe](https://github.com/Gilzuk/relaynet2/tree/dc7c02c00cbebdfb566362a810ab4b45cb6abbe0/results/reproduction). The sequence study repeats seed IDs 0, 1 and 2; coded initialization remains seed 0. Thus these runs supplement the multi-seed evidence, but all differences cannot be assigned to changed initial seeds, and repeated seed IDs are not independent new seeds. The descriptions below of original and earlier rerun values retain their historical scope.

| Point | Before | Now |
|---|---|---|
| 1. Canonical runs | The training section described the intended protocol, not the runs behind Tables 5.2 and 5.3 | Sec. 4.3.3 identifies both runs (command, code and data commits, budget, hardware). New Table 4.2 gives each relay's training as executed, and the section states what cannot be established |
| 2. SNR penalties | A target the relay missed was dropped, so a weaker relay could score better (Mamba-2's +4.69 dB, which never reaches 10⁻³) | The reference fixes the targets. A missed target gives a lower bound, marked ≥, and is not dropped. Tables 6.7 and 9.4–9.6 and their prose are recomputed |
| 3. Matched-decoding scope | "Both/all learned relays" lose to matched block DF | Only the MLP hard and soft read-outs were rerun, and the statements say so. Mamba was not rerun |
| 4. Formal problem | Normalization folded into the relay function. Expected BER called nondifferentiable | Sec. 4.1.5 separates the local function f from the block normalization operator N. The obstacle is now stated precisely |
| 5. Principal comparisons | Coded discussion led with the unweighted pipeline. Complexity tables gave joint-trellis counts | Matched decoding leads (Sec. 5.5.1). Factorized counts are primary in Tables 6.8 and 6.9 and Figure 6.7. The unweighted and joint-trellis results are short diagnostics |
| 6. Scope and presentation | Six inconsistencies | All six fixed (details below). Scripts are now cited as references to the ledger, not named in the text |

**1. Canonical runs.** Both tables come from runs recorded in the repository's history (Sec. 4.3.3):
- **Table 5.2.** Produced on 17 August 2026 by the canonical driver with fresh training and the cGAN skipped, ten trials of 10,000 bits at 0–20 dB. The code was commit b4059e6 and the data were committed as 69a470c. The run took 2,892 s on a four-core CPU without CUDA.
- **Table 5.3.** Produced by the same driver and options. The code was commit 4cf9753 and the data commit d27fcf7. The run took 746.6 s.

The exact commands are in ledger entry C4 (Appendix 9.3).

The archived checks reproduce both tables' AF and DF per-trial arrays bit for bit. Together with the traced driver and run metadata, this supports the documented classical channel, trial-seed and evaluation-budget configuration. It does not recover or independently verify the missing learned checkpoints, initialization states or historical software environment.

New Table 4.2 gives each learned relay as trained:
- parameters, window and training SNRs;
- the number of training windows, and the part used for gradient steps;
- Adam learning rate and batch size;
- the checkpoint kept: the final epoch, or for the sequence models the last epoch whose mean training loss improved on the best kept loss by more than 10⁻⁵ (early stopping was off).

All evaluated learned relays in the two canonical tables train for 100 epochs on BPSK ±1 over real AWGN, with seeded training data. The section also states how the relays are applied: per axis to the zero-forced observation without rescaling, so each axis carries ±1/√2 against the ±1 of training, with the output normalized over the 5,000-symbol block.

The sequence relays add the raw center sample after their tanh heads, so their complete outputs are not tanh-bounded. The text now states this skip connection, the Transformer dropout and positional encoding, disabled optional input normalization, and the complete VAE-3K widths. Table 6.7 uses a separate 40-epoch loop, now described with its own sample counts, optimizer and evaluation budget.

What cannot be established is stated as such. The archived canonical execution path does not seed PyTorch's RNG, including initialization, sequence hold-out/dropout, MLP/sequence batch order and VAE latent training noise. The VAE's NumPy batch order is seeded, and evaluation decodes its latent mean deterministically. The canonical checkpoints, per-epoch losses and historical library versions are absent from the archived record. The recoverable configuration is described explicitly, but exact replay of the historical learned columns is not established. Chapter 2 now also agrees with the implementation: the VAE samples its latent variable during training and decodes the encoder mean at inference. Equality of two historical point estimates is not presented as equivalence or evidence for a causal effect of variational regularization.

**2. SNR penalties on comparable targets.** The rule is now:
- the reference curve fixes the targets: those of 10⁻¹, 10⁻² and 10⁻³ it reaches by 20 dB;
- a relay that misses one of them gets a lower bound, 20 dB minus the reference's SNR there, marked ≥;
- best-of-seeds and spread statistics carry the bounds through.

A script recomputes every affected table from the stored per-seed curves. The changes:
- **Table 6.7.** The Mamba-2 initialization scored +4.69 dB at ISI (QPSK) needs more than +9.11 dB under the new rule. Mamba-2's best entry there is now +5.83 dB, with a spread above 3.28 dB (was +4.69 and 2.12). The MLP still has the best initialization on all three channels. Its lead on ISI (QPSK) is 3.14 dB; the thesis had quoted 2.11 dB, measured against the marked initialization. The other eleven cells are unchanged. The cross-machine rerun now reproduces eleven of the twelve best values within 0.25 dB, where it previously reproduced ten.
- **Table 9.4 (minimum size).** Only the coded row changes. Its entries for the three strictest budgets (372p w9, 244p w5, 244p w5) miss 10⁻³, so no tested configuration is reported as meeting those budgets on the common target set.
- **Table 9.5 (initialization spreads).** A configuration in which no seed reaches every target has no defined spread and is excluded:
  - pooled: 148 configurations (was 152), median 0.176 dB (was 0.180), 90th percentile ≥1.226 dB (was 1.229), maximum ≥1.555 dB (was 3.182);
  - the earlier 3.182 dB maximum was produced by the dropped target;
  - the ISI and ISI (complex) rows change accordingly.
- **Table 9.6.** One VAE and two Transformer initializations miss 10⁻² and are marked ≥. The Transformer spread is at least 0.892 dB (was 0.906), exceeding the roughly 0.18 dB scale associated with H4's spread. The VAE's spread is at least 0.156 dB, so it cannot be claimed to lie below that scale.
- **Training-loss plot.** The three Transformer penalties with missed targets are now shown as lower bounds, with open markers and upward arrows. The figure no longer treats those bounds as measured penalties to claim a loss–performance correlation.
- **Native architecture spreads.** The Transformer's is at least 0.075 dB and the VAE's at least 0.091 dB.
- **Window effect.** "8 to 11 dB between window 1 and window 7" becomes "more than 8 dB", since the window-1 values on the three-tap channels are lower bounds (open markers in Fig. 9.1).
- **Coded size sweep (Ch. 8).** The range against unweighted block DF is +1.24 to +1.92 dB. The thesis had printed +1.86 to +1.92 dB, which the data never supported; this is an earlier error found in the recheck. Against matched block DF the penalty is at least +9.75 dB, replacing +7.6 to +9.9. The lower values came from configurations that miss 10⁻³.

The MLP retains the best measured initialization on all three channels of historical Table 6.7. Reanalysis of the archived size grid leaves its non-coded rows unchanged; the grid was not retrained in the October validation. On ISI (QPSK) the best-seed numerical order of Mamba-S6 and Mamba-2 reverses (5.72 against 5.83 dB). This is a numerical ordering, not evidence that either architecture is better, and the thesis does not read the order of the sequence models as resolved.

**3. Matched-decoding scope.** Only the MLP hard and soft read-outs were rerun with matched metrics. Every statement that "both" or "all" learned relays lose to matched block DF now names those two read-outs and says that the coded Mamba relay was not rerun. This applies to Sec. 5.5.1, Chapter 1, Chapter 8, Chapter 9 and the abstract.

**4. Formal problem (Sec. 4.1.5).**
- **The relay map.** The relay applies a local function f to each observation, u[i] = f(I_R[i]), and transmits N(u). The operator N implements either common-gain or separate-axis normalization, so the complete relay map N∘f depends on the block.
- **As implemented.** AF and DF normalize the complex block. The canonical learned relays normalize each real axis separately, so they transmit α(u_I)·u_I[i] + j·α(u_Q)·u_Q[i]: unit power per axis and power 2 for the complex block, against 1 for AF and DF. Hop 2 sets its noise from the measured block power, so the operating SNR is the same for every relay. The Hybrid also chooses between its MLP and DF per axis and per block, from that axis's block power, so even its local rule depends on the block. Secs. 4.1.3, 4.1.5, 4.2 and 4.5.3 now describe this map.
- **Power and objective.** The appropriate complex-block or per-axis power constraint holds by construction of N for nonzero outputs. The local-design problem is min over f of P_b(N∘f), conditional on the specified block switch for the Hybrid.
- **Differentiability.** The blanket nondifferentiability statement is replaced. The empirical error count is piecewise constant in the parameters, so its gradient vanishes almost everywhere. Expected BER can be differentiable after noise averaging. In the uncoded Gaussian second hop, conditional slicer errors are smooth Q-functions; away from zero energy and discrete branch changes, differentiation can pass through N and differentiable local processing. That requires the second hop and the destination rule in the training loop, which a relay trained on Hop 1 does not have.

**5. Corrected comparisons as principal results.**
- **Coded section.** It opens with "Matched Decoder Metrics: The Principal Coded Comparison" (Sec. 5.5.1). The unweighted pipeline (Sec. 5.5.2) and soft-decision relaying under unweighted metrics (Sec. 5.5.3) are now one paragraph each, with the five detailed tables and the unweighted soft-decision figure retained in Appendix 9.7. Rate adaptation, equal throughput and the readiness budget now lead with the matched results.
- **Complexity.** Tables 6.8 and 6.9 give factorized-trellis counts first: 16 multiplications at three taps and 256 at seven, against the relay's 208. The joint-trellis counts appear as a diagnostic column or in brackets. The crossover is L* = log₂(2WH+4H) − 1 = 6.70 taps. The joint-trellis value of 3.35 is kept as a diagnostic.
- **Abstract, Ch. 8 and Ch. 9.** These lead with the factorized comparison: fewer multiplications for MLSE up to six taps, and about 1.23× more at seven.

**6. Scope and presentation.**
- **Causal Mamba (p. 32).** Remark 4.1 now says that past-only processing was evaluated only through the simplified Mamba relays. Those read their causal scan at the window centre, y_R[i−5:i].
- **Common input context (p. 44).** Sec. 4.4 says the eleven-sample window is not a common context. The Mamba relays use six of the eleven samples, so the common window equalizes input length only for the feedforward models and the Transformer.
- **Simplified Mamba.** Both implementations are identified as simplified in Chapters 1–5 and Table 6.7, and Sec. 4.2 gives the differences:
  - Mamba-S6 has neither the depthwise convolution nor the gating branch;
  - Mamba-2 omits the depthwise convolution and uses a sigmoid gate in place of the normalized SiLU gate.
- **Figure 6.5.** The caption says the difference at ten pilots is not resolved: 0.0545 ± 0.0108 against the MLP's 0.0486.
- **Archived revision.** Sec. 4.3.4 links the existing immutable code/data snapshot [80b4545adcc66e4be1f435364f7ec927e87f6414](https://github.com/Gilzuk/relaynet2/tree/80b4545adcc66e4be1f435364f7ec927e87f6414) of the Gilzuk/relaynet2 repository, including the revised plotting tools. It no longer claims that the planned release tag already exists. It also names commits b4059e6 and 4cf9753 for the canonical runs.
- **Chapter 5 headers.** All running headers after the chapter-opening page read "Core Experiments" (folios 51–63); the opening page has the correct chapter title.
- **Code references.** Scripts and result files are no longer named in the running text. The text cites them as [C1]–[C41], and the master experiment ledger (Appendix 9.3, Table 9.2) lists each entry's script, its result file and where the result is used.

**Further corrections from the earlier full-thesis check.** These changes to stated results are incorporated in the linked submission:
- **Canonical comparison (Sec. 5.2).** The thesis said the MLP, Hybrid and VAE show no detectable difference from DF from 4 dB, nor from one another. Paired over the ten shared trials, as Sec. 4.3.2 prescribes, DF has the lower BER at seven of the eleven SNRs against the MLP and against the VAE, and at one against the Hybrid. No canonical learned relay has a resolved advantage over DF at any tested SNR. The historical MLP/Hybrid/VAE gaps stay within 0.004; the separate new VAE realization reaches 0.00640, as reported above. The text distinguishes these scopes.
- **Table 6.6.** The equalizer and the learned relay had been measured against two different MLSE runs. Measured against one common curve, the learned relay's lead over the best equalizer is 1.8–2.3 dB, not 1.8–2.4.

The other corrections are to ordering and consistency:
- The seed-spread caption no longer asserts an unverified five-of-nine comparison with model-size effects.
- The coded summaries in Chapters 7 and 8 now lead with the matched result.
- Factorized trellis counts now come first throughout, including Figure 6.7. Matched table columns come first; the original unweighted plots are in Appendix 9.7.
- Local causal processing is distinguished from block normalization. The text does not claim that removing look-ahead preserves a trained model's BER, that a censored-loss correlation is established, or that arithmetic counts prove hardware feasibility.
- The Hybrid's 2–6 dB training SNRs, the test count and the coded trial budgets are now consistent.
- The pilot claim now says "at five pilots".

The revised table values are recomputed from the stored per-seed data and checked automatically against the text. Both abstracts now state the real-tap factorization condition and the evaluated hard/soft MLP scope explicitly; the Hebrew multiplication ratio is unambiguous.

**Final paired-test correction.** Floating-point subtraction of BERs could split ties between equal absolute error-count differences. The analysis now reconstructs integer error counts using the declared denominator (10,000 bits for canonical trials and 100,000 information bits for coded trials), validates it, and applies the two-sided Wilcoxon test to those differences. All 418 stored ten-pair contrasts agree with an independent sign-enumeration check. Two classifications cross the uncorrected 0.05 threshold:

- Historical 3K Transformer versus DF at 20 dB: p changes from 0.05078125 to 0.04296875, resolving a DF advantage.
- Additional native S6 versus DF at 16 dB: p changes from 0.03515625 to 0.05468750, leaving that difference unresolved.

Neither change establishes a learned canonical advantage over DF. All 24 new matched-coded contrasts retain p < 10⁻⁵. These are exploratory, uncorrected comparisons conditional on trained checkpoints, not equivalence tests. Raw BER observations are unchanged by the statistical correction.

**Reproduction evidence and final checks.** The retained-model audit loads 48 saved checkpoints and verifies 70 archived output hashes. These are the twelve new canonical and thirty-six sequence checkpoints; the trained classifier weights from the matched-coded rerun were not retained. Original canonical weights, complete training RNG states, per-epoch losses and historical library versions are also missing. Those limitations are stated in the thesis and researcher protocol.

The broader runner now records a default Python/NumPy/Torch bootstrap seed of 42, preserves driver-local seeds, requires fresh outputs, retains per-step output versions and saved checkpoints, and reports missing outputs or verifier failures as errors. It distinguishes three historical archive references from executable producers. The full 73-entry plan has not been executed in full. A cheap committed-source ISI-penalty analysis regenerated its JSON byte-for-byte and passed verification; it is not a new BER simulation.

The final merged snapshot passes 892 numerical checks with zero inconsistencies. Its code has 386 passing tests, and CI passed on Python 3.10, 3.11 and 3.12. The PDF, preview and both Overleaf bundles are synchronized. A final targeted source/PDF check against your six requests identified no further blocking correction within the evaluated and disclosed scope. This is a technical readiness assessment; the response remains a draft until sent.

Best regards,
Gil
