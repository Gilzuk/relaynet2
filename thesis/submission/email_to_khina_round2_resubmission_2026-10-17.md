Subject: Revised thesis — response to round-2 feedback

Draft for Gil to send; not sent. Reviewed submission snapshot: `5eb34ab3ed01bc9df17498358656965af2fa916b`.

Hi Anatoly,

Thank you for your detailed feedback. I have addressed the six remaining requests in the revised thesis:

[Revised thesis PDF](https://github.com/Gilzuk/relaynet2/blob/5eb34ab3ed01bc9df17498358656965af2fa916b/thesis/main.pdf)

1. **Canonical runs:** Section 4.3.3 and Table 4.2 now identify the actual architectures, training channels and scaling, sample counts, epochs, optimizer settings, checkpoint selection and evaluation budgets. Missing historical weights, training RNG states and software versions are explicitly disclosed.
2. **SNR penalties:** The comparisons use reference-defined BER target sets and retain unsuccessful targets as bounds. I rechecked the architecture ordering, initialization spreads and minimum-size statements against the saved per-seed curves; size conclusions are limited to the tested grid.
3. **Matched decoding:** The principal coded comparison in Section 5.5.1 covers the evaluated hard and soft MLP read-outs. It explicitly states that matched Mamba was not evaluated.
4. **Formal problem:** The local relay function is separated from block-wide normalization. The training difficulty of empirical hard decisions is distinguished from the possible differentiability of noise-averaged expected BER.
5. **Principal comparisons:** Matched decoding leads the coded discussion, and factorized-trellis counts appear directly in the main complexity tables. Unweighted decoding and the joint trellis remain diagnostic comparisons.
6. **Scope and presentation:** The causality and input-context statements are qualified, the simplified Mamba implementations are identified consistently, the ten-pilot difference remains unresolved in Figure 6.5's caption, an immutable code revision is provided, and Chapter 5's running headers are corrected.

The final review also corrected signed-rank ties by using integer error-count differences and reconciled the VAE description with its implementation: latent sampling occurs during training, while inference decodes the latent mean. These corrections preserve the principal conclusions within the evaluated scope: no resolved canonical learned-relay advantage over DF, and a matched block-DF advantage over both evaluated coded MLP read-outs.

Section 7.1.4 reports the additional training runs separately from the historical measurements. Classical control arrays match in the four archived hosted replications; learned results vary. The repeated sequence seed IDs and unchanged coded initialization mean these differences cannot all be attributed to changed initial seeds. A separate seed-42 MLP control repeats exactly on one host. Missing historical states and the absent trained classifier weights in the coded rerun remain explicit limitations; the original BER measurements are retained.

The revised document has 120 numbered body pages. The numerical verifier checks 892 values with no inconsistencies; the current software suite passes 386 tests. The repository contains the point-by-point response, reproducibility instructions, seeded execution script, logs, checkpoints where available, and comparison JSON files. The broader full-experiment plan has not been executed in full.

Thank you again for your careful review and guidance.

Best,
Gil
