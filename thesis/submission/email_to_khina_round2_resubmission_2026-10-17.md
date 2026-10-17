Subject: Revised thesis — response to round-2 feedback

Draft for Gil to send; not sent. Corrected PDF is in unmerged PR #189; integrate it and publish the generated mirror before sending. Confirm the accepted faculty page count. Reviewed submission snapshot: `908687650929a74a969f74fa679f67633b8c3793`.

Hi Anatoly,

Thank you for your detailed feedback. I have addressed the six remaining requests in the revised thesis. Below I quote the relevant corrections; page numbers refer to the printed thesis pages, rather than the PDF viewer's page count.

[Revised thesis PDF](https://github.com/Gilzuk/relaynet2/blob/908687650929a74a969f74fa679f67633b8c3793/thesis/main.pdf)

1. **Actual canonical configurations — §4.3.3, Table 4.2, pp. 42–43.** The table and accompanying text specify the architectures, training channels, scaling, sample counts, epochs, Adam settings, checkpoint selection and evaluation budgets. For example: “All are trained for 100 epochs on BPSK ±1 over real AWGN” (p. 42). The limits of the historical record are explicit: “The canonical checkpoints, per-epoch losses and historical library versions are absent from the archived record.” (p. 43). The same paragraph identifies the unrecorded training RNG states.

2. **Comparable SNR penalties — Table 6.7, p. 82; Appendix 9.6, pp. 115–116.** The corrected rule states that “every initialization is held to the targets MLSE reaches” (Table 6.7). Appendix 9.6 states that a missed target “is marked as a lower bound rather than dropped” (p. 115). For the QPSK Mamba-2 example, the failing initialization now has a penalty >+9.11 dB; the best-of-three entry is +5.83 dB with spread >3.28 dB. Architecture ordering and initialization spreads were rechecked, and the caption cautions that “the ordering of the three sequence architectures should not be read as resolved.” Minimum-size conclusions concern tested configurations, “not mathematical lower bounds on architecture size” (§5.4, p. 56).

3. **Matched-decoding scope — §5.5.1, p. 57.** The text now states: “The Mamba-coded relay of the unweighted study was not rerun with matched metrics, so every statement below concerns the two MLP read-outs.” The principal comparison therefore covers the evaluated hard and soft MLP variants.

4. **Formal problem and normalization — §4.1.5, pp. 33–34.** The local map is followed by a separate block-normalization operator, and the text states that “each transmitted symbol can depend on the entire block” (p. 33). It distinguishes the empirical loss — “The empirical hard-decision error count is piecewise constant in the relay parameters” — from the expected objective: “Expected BER can instead be differentiable after averaging over noise” (p. 34).

5. **Corrected comparisons presented first.** The coded discussion now begins with “Matched Decoder Metrics: The Principal Coded Comparison” (§5.5.1, p. 56). Table 6.8 states: “MLSE arithmetic is that of the exactly factorized detector, with the implemented joint trellis in brackets.” (p. 85). Table 6.9 directly lists factorized counts of 16, 64 and 256 multiplications at three, five and seven taps, against 208 for the relay (p. 87). Unweighted decoding and the joint trellis remain diagnostics.

6. **Scope and presentation.** Remark 4.1 specifies that “Past-only local processing was evaluated only through the two simplified Mamba relays” (p. 32). Section 4.4 qualifies the shared window with “but not a common context” and explains the six-versus-eleven-sample dependence (p. 45). The architecture descriptions explicitly say “This is a simplified Mamba” and “This too is a simplified Mamba-2” and identify their departures from the reference implementations (§4.2.1, pp. 38–39).

   Figure 6.5 now says: “at 10 pilots the difference is not resolved (0.0545 ± 0.0108 against the MLP’s 0.0486)” (p. 79). Section 4.3.4 identifies the “immutable repository snapshot 80b4545adcc6” (p. 44). Chapter 5's running headers read “Chapter 5. Core Experiments” (pp. 51–63).

The final review also corrected signed-rank ties using integer error-count differences and reconciled the VAE description with its implementation: “at inference it decodes the encoder’s latent mean, so its output is deterministic” (p. 11). The principal conclusions remain bounded by the evaluated scope: no resolved canonical learned-relay advantage over DF, and a matched block-DF advantage over both evaluated coded MLP read-outs.

The additional training runs are reported separately in §7.1.4: “Differences between trained models do not by themselves invalidate the original measurements, which remain unchanged.” (p. 89). Classical control arrays match in the four archived hosted replications; learned results vary. Repeated sequence seed IDs and unchanged coded initialization prevent attributing all differences to changed initial seeds. A separate seed-42 MLP control repeats exactly on one host. Missing historical states and absent trained classifier weights in the coded rerun remain explicit limitations.

The revised document has 120 numbered body pages and 136 physical pages including front and back matter. The latest review also aligned both repository README sources with the evidence, clarified the BPSK bit-label conventions and placed the acknowledgments before the abstract. The numerical verifier again reports 892 checks with no inconsistencies; 32 document/verifier regression tests pass. The earlier 386-test software-suite result remains applicable to the unchanged experiment code. The repository includes the point-by-point response, reproducibility instructions, seeded execution script, logs, available checkpoints and comparison JSON files. The broader full-experiment plan has not been executed in full.

Thank you again for your careful review and guidance.

Best,
Gil
