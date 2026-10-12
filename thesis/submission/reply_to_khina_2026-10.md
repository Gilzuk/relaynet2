Subject: Re: Thesis — response to your nine comments

Dear Anatoly,

Thank you for the detailed comments. I have addressed all nine and the presentation notes. Three of them (comments 3, 4 and 7) led to new measurements that change conclusions in the thesis, and I state those plainly below. The revised thesis runs to 119 pages from Chapter 1 through the appendices, within the 120-page limit. The table verifier now checks 678 cells with no inconsistencies, and all 347 tests pass.

**1. The relay-design problem.** A new Section 4.1.5 states the problem explicitly:
- the relay's observations and the channel information it has;
- the permitted look-ahead;
- the block power constraint;
- the fixed destination decision rule, including the matched destination metric of the coded rerun;
- the objective, destination BER.

It then explains MSE and cross-entropy training as surrogates for that objective. For relays that forward constellation points, P_b = p₁ + p₂ − 2p₁p₂ increases with the relay's own error rate, so relay-side bitwise MAP is the BER-optimal target in those uncoded studies. It coincides with symbol-MAP for BPSK or a factorized QPSK posterior; a joint four-class QPSK arg-max instead targets symbol error. For soft-output relays no such link holds: the posterior mean is only a local target, and the thesis measures destination BER rather than claiming optimality.

While writing this section I found that the methods text described the canonical relays as trained over fading. In fact they are trained once on BPSK over AWGN and applied per axis to the zero-forced QPSK/Rayleigh signal. Sections 4.2 and 4.3.3 and Eq. (4.16) now describe the code as it runs, and Section 7.1.1 no longer suggests that the relays learn an average over the fading.

A final recheck against the code found two more descriptions that did not match it:
- The Hybrid relay does not use a learned threshold on the operating SNR. It compares a fixed 5 dB threshold with a per-axis received-power estimate that assumes unit signal power, and its MLP sub-network is trained separately at 2, 4 and 6 dB. I replayed the canonical run's first-hop draws; the replay reproduces the published error rate of every trial on which both axes took the DF branch. On the zero-forced Rayleigh observation the switch falls between 6 and 8 dB rather than at 5 dB (Section 4.2, Appendix 9.1).
- The sequence models hold out 15% of their training windows for accuracy reporting and keep the epoch with the lowest training loss (Section 4.3.3).

**2. Research questions, hypotheses and progression.** Section 3.1.1 now states the hierarchy:
- O1 and O2 are answered in Chapter 6, the principal contribution. O1 is tested through H5 and the pilot sweep; O2 is answered descriptively by the delay and arithmetic studies.
- H1–H4 belong to the memoryless control.

A new Table 3.1 maps each question to the experiments and sections that answer it, and Chapter 8 answers O1 and O2 by name. The "cumulative relaxations" wording and the "distance between boundaries" claim had already been removed.

**3. AF implementation.** The methods (Section 4.1.3) now state that the implemented AF zero-forces y/h and normalizes by the empirical block power, with no clipping or regularization, and that E[1/|h|²] is infinite. I measured block-length sensitivity: at 0 dB the AF error rate rises from 0.387 (50 symbols) to 0.413 (50,000 symbols). I also added a conventional finite-power variable-gain AF with coherent destination processing, which matches its closed form to within 0.001 at every SNR. Against it, the learned relays' 0 dB margin shrinks from 0.073 to 0.007 (Section 7.1.1, and the H1 row of the hypothesis table in Chapter 8). H1 remains supported as stated, but the learned relays track DF, so the advantage is one of regeneration rather than of learning. The canonical results (Section 5.2), the Chapter 8 findings and both abstracts now say that the learned advantage is over the block-normalized AF, and give the 0.007 margin against the conventional AF.

**4. Coded metrics: this reverses the coded conclusion.** I implemented the matched metric: |h|² weights in the Viterbi decoder and per-symbol noise variance in the BCJR, at both the relay and the destination. I then reran the principal coded comparisons on the published seeds. The rerun uses the first 30 of the published 100 paired trials, each with 100,000 information bits. The unweighted rows reproduce those published per-trial results exactly. The 30-trial budget was retained after inspecting running means, so the paired tests are exploratory; the within-trial comparison isolates the metric change.
- With matched metrics, block DF has lower BER than both learned relays at all six tested SNR points (0, 4, 8, 12, 16 and 20 dB), in 30 of 30 paired trials at each point, under either destination metric for the learned relays.
- At 20 dB block DF makes 3 errors in 3×10⁶ bits, against 2,667 for the best learned relay.

The published high-SNR advantage of the learned relays was therefore produced by the mismatched metric (new Section 5.5.2, Table 5.9). I also reran every secondary coded study with matched metrics:
- With matched decoding, block DF's relay makes almost no symbol errors.
- In the tested K = 3, 5 and 7 sweep, larger K lowers measured BER from 4 dB upward; it remains worse at 0 dB. This is a result for these codes and this protocol, not a general monotonicity theorem.
- At the tested 24, 28 and 32 dB points, block DF and the clean-codeword oracle both record zero bit errors in 6,000,000 information bits per point. Equal observed counts at this budget do not establish equal underlying BER.
- At equal throughput, coding pays from 8 dB.
- Decoding at the relay gives the higher goodput under rate adaptation, except under tight idealized buffering budgets, where block DF's frame does not fit.

The matched values appear beside the published ones in Tables 5.6, 5.10, 5.12 and 5.13. Chapter 7 (coded-metrics limitation) and the Chapter 8 summary state the reversal. You were right that the original results showed only an advantage over a mismatched pipeline.

One limit of the rerun: for relays that make decisions, the matched destination metric weights by the second hop only, so it does not model the relay's decision errors. That is why it lowers the learned relays' error rate at low SNR but raises it at high SNR, where a wrong relay symbol received over a strong second hop is trusted. A destination metric that also models relay errors was not evaluated; Section 5.5.2 says so, and Chapters 7 and 8 list it as a remaining alternative.

**5. QPSK MLSE factorization.** I implemented the two binary trellises and verified them against the joint QPSK trellis: 264,000 decisions across L ∈ {3, 5, 7} and several traceback depths, with no disagreements. The factorization paragraph in Section 2.6 records the check, and the cost comparisons in Section 6.5 are updated. Section 2.6 also states the general case: with real taps, a square M-QAM trellis separates into two √M-ary trellises with 2M^(L/2) branch metrics, which for 16-QAM at five taps is 2,048 rather than about 10⁶. The caption of Figure 6.7 and the scaling paragraph of Section 6.5 now say that the plotted M^L count is that of the joint trellis.

One note on the numbers. The thesis counts one real multiplication per real squared residual, giving 2^(L+1) = 16, 64 and 256 for L = 3, 5, 7. That puts the crossover with the 208-multiplication network at about 6.70 taps, and the 7-tap ratio at 1.23. Your figures (32/128/512, 5.70 taps, 2.46) correspond to counting two multiplications per residual. The qualitative conclusion is the same under either convention: with factorization the arithmetic argument largely disappears. The counting rule is stated explicitly in Section 6.5.

**6. Delay accounting.** The thesis now distinguishes four things (Section 5.5.3 with Table 5.11, Section 6.5.1 with Table 6.8, and Section 7.3.1):
- detector look-ahead;
- normalization buffering (the soft relays, soft block DF and AF normalize over the whole block);
- protocol scheduling;
- end-to-end delay.

The hard relays in the latency study forward exact constellation points with no normalization, so their stated delays are true readiness delays; the caption of Table 6.8 now says its delay column is detector look-ahead only. Table 5.11's column is likewise labelled detector look-ahead, and its caption says that the soft relays' outputs are not final until the block ends. Figure 5.5 labels its axis as the idealized buffer-readiness budget rather than as a latency. The 202-symbol figure is attributed to the implemented full-frame decoder, with finite traceback noted as possible. I removed the operational-readiness claims rather than evaluate a streaming normalization rule, which is stated as not evaluated.

**7. Model specifications.** A new Appendix 9.6 (Table 9.7) is transcribed from the scripts that produced the data. It covers:
- both hop models and their parameter laws;
- impairment ordering and redraw intervals;
- the amplifier map, a Rapp limiter with saturation 1.2;
- normalization, pilot construction and window-boundary handling;
- checkpoint selection.

The size-sweep configurations are now specified in Appendix 9.4 (Table 9.3): per-channel modulation, both hops and the comparator, the training and evaluation budgets, seeds and padding.

Appendix 9.6 also covers DBPSK: the learned relay estimates information bits and forwards a real signal, with no differential decoding at the destination. It defines the single-number SNR penalty as well: interpolation of ln BER, last crossing, zero-error handling, unbracketed targets dropped, and the worst case over 10⁻¹ to 10⁻³, with each table's seed aggregation stated. The captions of the penalty tables (Tables 6.6 and 6.7, and Tables 9.4–9.6, which were 9.3–9.5 in your copy) point to that definition.

Two findings from this work:
- The ISI penalties against genie MLSE (0.26 / 1.18 / 1.88 / 2.53 dB) had no committed derivation. They are now reproduced by a script and checked by the verifier.
- The flat-channel study added twice the intended Hop-2 noise to real forwarded signals. I fixed this and reran it. The error rates roughly halve, and DF and the MLP stay within 0.0013 from 8 dB up. Below 8 dB, however, the unknown-phase MLP trails differential detection by up to 0.0106 at 0 dB, below its training range. This is reported as measured.

**8. Architecture comparisons.** The Transformer parameter count is corrected to 17,697, counted from the instantiated model, and the feedforward width is stated consistently as 64, twice d_model (Section 4.2); a width of 128 would give 26,017 parameters in the full model. The 25,408 count covers only the two encoder blocks and excludes the 609 parameters in the input and output projections. The tanh/ReLU protocol statement is qualified (Section 6.2). The initialization spread now reads 0.39 dB at BPSK, 0.07 dB at QPSK and 0.03 dB on the composite channel, so it is not the most stable at BPSK (Section 6.4.2). The equal-budget discussion (Sections 4.4, 5.2, 5.3 and 6.4.2) is now tied to the training protocols throughout: each family trains with its own protocol, nothing was tuned per architecture, and no claim isolates architecture or capacity. Section 5.3 no longer describes its goal as isolating inductive bias.

I also found that the training-protocol description did not match the code. All relays train with Adam; the thesis said the MLP used SGD. The sample counts were also wrong. These are now described from the committed code (Section 4.3.3). Exact reproduction of the canonical data is not possible, because PyTorch initialization was unseeded, and the thesis says so.

**9. Positioning.** A new Section 2.2.4 positions the work against:
- estimate-and-forward (Gomadam and Jafar, IEEE JSAC 2007). The MSE-trained soft relays are learned approximations of a conditional-mean forwarding rule, not a new principle.
- Farsad–Goldsmith and ViterbiNet.

It states four things the thesis adds, beginning with the statement that the contribution is a measured comparison, not a new relay principle. A non-monotonic scalar MAP relay is now measured on the ISI channel (Section 6.2). It removes part of the gap over zero-threshold DF but remains far from the windowed MLP (0.099 against 1.1×10⁻⁴ at 12 dB). So the large gains over zero-threshold detection are not by themselves evidence for learning. Table 3.2 now compares named studies and their properties.

**Presentation notes.** Chapter 5 now carries its own running headers. The experiment ledger (Appendix 9.3) uses table and figure references instead of internal labels. The verification-coverage statements in Section 4.3.4 and Appendices 9.2 and 9.3 now agree: the verifier checks a declared subset of numerical cells, not every number or figure. The ledger's channel descriptions are corrected, and it lists the new experiments. Figures and tables now show the same values. Figures 5.3–5.5 are redrawn from the 100-trial runs that their tables report; they had been drawn from earlier runs (Figures 5.3 and 5.4 printed 20 and 10 trials). Table 5.8's 20 dB soft block-DF cell is now the run that Figure 5.3 plots, with the doubled-variance value as a footnoted sensitivity check rather than as the reported value. Figure 6.3 shows only the AWGN case that Table 6.4 reports. Figure 6.7 labels the network in panel (a) as the 170-parameter relay and states which network panel (b) timed. The test counts in Section 4.3.4 and Appendix 9.2 now agree with the current suite, and the relay configuration table in Appendix 9.1 (Table 9.1) is corrected. The appendices are numbered 9.1–9.6; the former letter names, which began at B, are dropped. The coded figures and Tables 5.5 and 5.8 state that they use the unweighted metric and point to the matched results.

The reversed coded conclusion is stated wherever the coded results are discussed, including the English and Hebrew abstracts.

Best regards,
Gil
