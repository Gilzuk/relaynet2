Subject: Revised thesis — round-2 corrections and resubmission

Draft for the author; not sent.

Hi Anatoly,

Thank you for your detailed feedback. I have addressed all six remaining points and am resubmitting the revised thesis:

[Revised thesis PDF](https://github.com/Gilzuk/relaynet2/blob/experiments/round2-reproduction-20261017/thesis/main.pdf)

The revisions cover the following:

1. **Canonical results:** I documented the actual architectures, training configurations, scaling, sample counts, optimizer settings, checkpoint selection and evaluation budgets. Missing historical information, including the original learned checkpoints and PyTorch RNG state, is explicitly identified.
2. **SNR penalties:** I corrected the comparisons to use common BER targets, retaining unsuccessful cases as explicit bounds, and rechecked the architecture rankings, initialization spreads and minimum-size conclusions.
3. **Matched decoding:** The conclusions now explicitly cover the evaluated hard and soft MLP variants. The absence of a matched Mamba comparison is stated.
4. **Formal problem:** I separated the local relay function from block-wide normalization and corrected the discussion of empirical hard-decision loss versus expected BER.
5. **Principal comparisons:** Matched decoding now leads the coded discussion, and factorized-trellis counts appear in the main complexity comparisons. The earlier comparisons remain as shorter diagnostics.
6. **Scope and presentation:** I reconciled the causality and input-context statements, consistently identified the simplified Mamba implementations, corrected the ten-pilot caption, provided an identifiable code revision and fixed Chapter 5’s running headers.

I also added full-budget training replications and repeated multi-seed tests. These supplement the original measurements; differences between trained realizations do not by themselves show that the original results were incorrect. The revised thesis separates the new canonical realizations, the repeated three-seed sequence study and the same-seed repeatability control. Classical controls reproduce exactly, and fresh paired tests preserve the matched-coded conclusion. Original tables remain intact, with the numerical variation and limits of seed attribution reported alongside them. The numerical verifier checks 892 values against the recorded evidence.

Thank you again for your careful review and guidance.

Best,
Gil
