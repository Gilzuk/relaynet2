# Thesis update — September 2026

**To:** Dr. Anatoly Khina
**Subject:** M.Sc. thesis — structure, content, and one correction

Dear Dr. Khina,

A brief summary of how the thesis has changed since the July draft. This
follows the major revision in `cover_letter.md` and covers the work after it.

---

## Structure

Seven chapters have become nine. A new chapter, *Unknown and Mismatched
Channels*, now sits between the core experiments and the discussion; it is the
material I would describe as the thesis's principal contribution, and it did
not exist in July. Discussion and Summary moved to Chapters 8 and 9, and a list
of symbols and a Hebrew abstract were added.

Despite the additions the thesis is shorter — 127 pages then, 120 now — as the
canonical experiments chapter was condensed.

## Content

The new chapter organises the argument as a ladder of relaxed assumptions: a
matched memoryless channel, then channel memory, then finite pilot budgets,
then unseen realisations from the trained impairment family. At each rung the
learned relay is measured against the strongest classical comparator
implemented. A supplementary study covers coded block decode-and-forward and
rate adaptation. Both abstracts were rewritten to match what the experiments
support.

## A correction I should flag

Final review found that the Viterbi results had been generated under an earlier
noise convention, 3.01 dB pessimistic, while the learned-relay results used the
current one — and the two were tabulated together. Re-measuring under a single
protocol changed three readings, all against the thesis's earlier framing:

- an apparent crossing at 8 dB was an artefact;
- the classical advantage is not a fixed 1–1.5 dB but grows from about 0.26 to
  2.5 dB as the error target tightens;
- at 16 dB the ordering was inverted, the matched detector being better rather
  than worse.

The claim is now narrower and, I believe, more defensible: a compact learned
relay restores reliable communication where the classical pipeline's
assumptions fail, at fixed arithmetic and with no per-block channel estimation
— but it does not outperform a correctly informed sequence detector.

## Verification

Supporting this is new machinery: an automated checker tying every published
number to its source data, a provenance audit linking each experiment to its
script and results, and a test suite grown from six files to twenty-one.

---

I would be glad to discuss any of this, particularly the revised scope of the
central claim.

With kind regards,

Gil Zukerman
