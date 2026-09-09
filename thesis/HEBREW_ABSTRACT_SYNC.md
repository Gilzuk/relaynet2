# Hebrew abstract: needs rewriting against the new English abstract

**Status as of 2026-09-09: superseded task.** The previous version of this note
listed four passages missing from the Hebrew. That delta no longer applies. The
English abstract has been rewritten from scratch, cut from 695 words to 381 in
5 paragraphs, to lead with the research question and the headline results
rather than carrying every scope qualification inline. The Hebrew abstract still
renders the old English text, so it is now a translation of a document that no
longer exists.

The Hebrew is the author's to write. The full new English source is below,
paragraph by paragraph. Nothing else in the thesis depends on the Hebrew wording,
so this can be done last.

## What changed and why

- Paragraph 1 is now the **question**, in one sentence, rather than a description
  of what the thesis compares.
- Paragraph 2 is the **setup**, compressed to the fixed configuration and the
  eight strategies.
- Paragraph 3 is the **matched-channel result**: DF wins from 6 dB, and channel
  memory rather than parameter count sets the size floor.
- Paragraph 4 is the **unknown-channel result and the two boundaries**, the pilot
  count and the arithmetic. This is the thesis's principal claim.
- Paragraph 5 states the **bound**: never better than a correctly modelled
  classical receiver; valuable where no such model or estimate exists.
- The coded study, the four-layer ladder enumeration and the traceback-depth
  latency detail were **dropped from the abstract** and remain in the body. The
  Hebrew should drop them too.

## New English source

### Paragraph 1 (43 words)

> A relay carries traffic between a source and a destination that have no direct path between them. This thesis asks a single question: when is a \emph{learned} relay worth using in place of a classical one, and what sets the boundary between them?

### Paragraph 2 (62 words)

> The comparison is run on one configuration, fixed in advance --- a single-relay SISO link with i.i.d.\ Rayleigh fast fading on both hops, complex baseband, Gray-coded QPSK, uncoded transmission --- leaving the relay function as the only variable. Eight strategies are evaluated, from amplify-and-forward (AF) and symbol-wise decode-and-forward (DF) to generative and sequence models, as Monte Carlo estimates with 95\% confidence intervals.

### Paragraph 3 (95 words)

> On a matched, memoryless channel the classical relay wins. Learned relays beat AF at low SNR, but from $6$~dB upward DF matches or exceeds every learned relay at no parameter cost. Capacity is not the binding constraint: a size sweep replicated over three initializations locates the smallest relay that costs nothing measurable, and \emph{channel memory} rather than parameter count sets it. Four parameters suffice on the memoryless channel, while the evaluated three-tap channels need $73$ to $145$ and a window spanning the interference. Error does not rise beyond the spread across initializations as capacity grows.

### Paragraph 4 (114 words)

> The picture inverts once the channel is unknown. Against an unmodeled three-tap ISI filter both memoryless classical relays fail outright, DF's error rate rising as transmit power grows, while a $170$-parameter windowed network restores reliable relaying to within $1$--$1.5$~dB of a Viterbi detector handed the exact taps. Two measured boundaries locate the crossover. The first is the reliability of the channel estimate: at a $10$~dB operating point, pilot-aided classical detection stays ahead at twenty pilots and loses by ten. The second is arithmetic: sequence detection costs $2M^{L}$ multiply-accumulates per symbol against the fixed relay's $2WH+4H$, the two equal near three and a half taps for the architecture deployed here and $158\times$ apart at seven.

### Paragraph 5 (67 words)

> The learned relay is therefore never better than a classical receiver that has been given the right model. Its value is that it still works when no such model, or no reliable estimate of one, is available, and that its cost does not grow with channel memory. Both statements are bounded to the classical pipelines implemented here and to the impairment family the relay was trained on.

## Current Hebrew (to be replaced)

`chapters/hebrew_abstract.tex` holds 4 content paragraphs rendering the
superseded text. Replace them wholesale rather than patching.
