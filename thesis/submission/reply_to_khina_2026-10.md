Subject: Re: Thesis — response to your nine comments

Dear Anatoly,

Thank you for the detailed comments. I have addressed all nine and the presentation notes, and the thesis runs to 120 pages from Chapter 1 through the appendices, within the limit. Comments 3, 4 and 7 led to new measurements that change conclusions, and I state those plainly below. Two later reviews of the whole thesis led to further corrections, listed at the end.

**1. The relay-design problem.** A new Section 4.1.5 states the problem:
- the relay's observations and channel information;
- the permitted look-ahead;
- the block power constraint;
- the fixed destination rule;
- destination BER as the objective.

It presents MSE and cross-entropy training as surrogates for that objective. For relays that forward constellation points, relay-side bitwise MAP is the BER-optimal target. For soft-output relays the posterior mean is only a local target, and the thesis measures destination BER rather than claiming optimality.

Checking the methods against the code turned up several descriptions that did not match it. They are now corrected:
- The canonical relays are trained once on BPSK over AWGN and applied per axis to the zero-forced QPSK/Rayleigh signal. They are not trained over fading (Sections 4.2, 4.3.3 and 7.1.1).
- The Hybrid compares a fixed 5 dB threshold with a received-power estimate, so its switch falls between 6 and 8 dB rather than at 5 dB (Section 4.2, Appendix 9.1).
- The sequence models hold out 15% of their windows and keep the epoch with the lowest training loss. The PyTorch relays use Xavier initialization (Section 4.3.3).
- The Mamba blocks are simplified, and both read a causal scan at the window centre, so they never use the future half of the window (Remark 4.1). Section 6.4.2's conclusion now rests on the Transformer. Giving the Mamba relays the whole window made them worse.

**2. Research questions and hypotheses.** Section 3.1.1 states the hierarchy:
- O1 and O2 are answered in Chapter 6, the principal contribution. O1 is tested through H5 and the pilot sweep, and O2 is answered descriptively by the delay and arithmetic studies.
- H1–H4 belong to the memoryless control.

A new Table 3.1 maps each question to the experiments that answer it, and Chapter 8 answers O1 and O2 by name. The asymmetric nonlinear channel named in H5 is now reported (Section 6.2).

**3. AF implementation.** Section 4.1.3 now describes the implemented AF exactly: it zero-forces, normalizes by block power, and E[1/|h|²] is infinite. Its error rate depends on block length: at 0 dB it is 0.387 for 50-symbol blocks and 0.413 for 50,000.

I added conventional finite-power AF with coherent detection. Against it, the MLP's 0 dB margin shrinks from 0.073 to 0.003–0.007, depending on the gain rule, and the sequence relays fall behind it. H1 remains supported. However, the feedforward relays track DF, so the advantage comes from regeneration, not from learning. Section 5.2, Section 7.1.1, Chapter 8 and both abstracts say so.

**4. Coded metrics: this reverses the coded conclusion.** I implemented the matched metric: |h|² weights in the Viterbi decoder and per-symbol noise variance in the BCJR, at both the relay and the destination. I reran the comparison on the first 30 of the published trials. With matched metrics, block DF beats both learned relays at every tested SNR, in every paired trial. At 20 dB it makes 3 errors in 3×10⁶ bits, against 2,667 for the best learned relay. The published high-SNR advantage was therefore produced by the mismatched metric (new Section 5.5.2, Table 5.9).

Every secondary coded study was rerun with matched metrics and is reported beside the published values (Tables 5.6, 5.10, 5.12 and 5.13). Matched block DF is stronger in each of them, except where tight idealized buffering budgets leave no room for its frame.

Two limits are stated in the thesis:
- the matched relay decoder uses |h₁|², which the learned relays are not given;
- for relays that make decisions, the destination metric does not model relay errors.

You were right that the original results showed only an advantage over a mismatched pipeline.

**5. QPSK MLSE factorization.** I implemented the two binary trellises and verified them against the joint trellis, with no disagreements. Section 2.6 states the general case: with real taps, a square M-QAM trellis separates into two √M-ary trellises. Figure 6.7 and Section 6.5 now say that the plotted M^L count is that of the joint trellis.

One note on counting. The thesis counts one multiplication per real squared residual, which gives 16, 64 and 256 for L = 3, 5 and 7, and a crossover with the 208-multiplication network at about 6.7 taps. Your figures count two multiplications per residual. Either way, factorization largely removes the arithmetic argument.

**6. Delay accounting.** The thesis now separates detector look-ahead, normalization buffering, protocol scheduling and end-to-end delay (Sections 5.5.3, 6.5.1 and 7.3.1; Tables 5.11 and 6.8). Specifically:
- the delay columns are labelled as detector look-ahead;
- the soft relays' block normalization is disclosed;
- Figure 5.5's axis is an idealized buffer-readiness budget;
- the 202-symbol figure is attributed to the full-frame decoder.

I removed the operational-readiness claims rather than evaluate a streaming normalization rule, and the thesis says that rule was not evaluated.

**7. Model specifications.** A new Appendix 9.6 (Table 9.7), transcribed from the scripts, specifies:
- both hops and the impairment ordering;
- the amplifier;
- normalization, pilots and window boundaries;
- checkpoint selection and DBPSK handling;
- the SNR-penalty definition.

Appendix 9.4 (Table 9.3) specifies the size-sweep configurations.

Two findings came out of this work:
- The ISI penalties against genie MLSE had no committed derivation. A script now reproduces them.
- The flat-channel study added twice the intended Hop-2 noise. After the fix the error rates roughly halve. Below 8 dB the unknown-phase MLP trails differential detection, by up to 0.0106 at 0 dB, and this is reported as measured.

**8. Architecture comparisons.** The Transformer parameter count is corrected to 17,697, with a feedforward width of 64 (Section 4.2), and the initialization spreads are corrected (Section 6.4.2). The equal-budget discussion now says that each family uses its own untuned training protocol, so no claim isolates architecture or capacity. The training description now matches the code: all relays use Adam, and the sample counts are corrected. The thesis states that the canonical data cannot be reproduced exactly, because the PyTorch initialization was unseeded.

**9. Positioning.** A new Section 2.2.4 positions the work against estimate-and-forward (Gomadam and Jafar), Farsad–Goldsmith and ViterbiNet. It states that the contribution is a measured comparison, not a new relay principle, and Table 3.2 compares named studies. I also measured a non-monotonic scalar MAP relay on the ISI channel. It removes part of the gap over zero-threshold DF but stays far from the windowed MLP. Large gains over zero-threshold detection are therefore not, by themselves, evidence for learning.

**Presentation notes.**
- Chapter 5 has its own running headers.
- The experiment ledger uses table and figure references.
- The verification-coverage statements agree with one another.
- Figures 5.3–5.5 are redrawn from the 100-trial runs their tables report.
- Figure 6.3 shows only the AWGN case.
- Figure 6.7 names the networks it plots and times.
- The test counts and Table 9.1 are corrected.
- The appendices are numbered 9.1–9.6.
- The coded figures and Tables 5.5 and 5.8 state that they use the unweighted metric.

**Further changes from two later reviews.** Three of them change stated results:
- **Ten pilots.** The lead claimed at ten pilots is not resolved. The MLP trails pilot-aided detection at twenty pilots and leads at five (both abstracts, Table 6.1, Section 6.3, Chapter 8).
- **Blind regime.** With corrected intervals and a paired test over the 50 shared channel draws, the lead over CMA is resolved from 0 to 12 dB (p ≤ 0.003) but not from 14 to 20 dB (p ≥ 0.40), where the thesis had cited it (Section 6.4, Table 6.1, Chapter 8).
- **Inference time.** The MLP's speedup over the Python Viterbi in Figure 6.7(b) is now measured single-threaded from committed data: 58–82×, replacing an unsourced 30–90×.

The other changes are to wording and disclosure:
- H5 is stated against the model-based detectors, since the chosen taps defeat AF and zero-threshold DF analytically.
- H2's verdict reads "supported as matching".
- The thesis states that H1–H4 measure transfer from BPSK/AWGN training, and that per-seed spreads change between machines.
- Table 6.6 defines its learned-relay column and adds the lead at the median initialization.
- Several smaller numbers, labels and references are corrected.

Best regards,
Gil
