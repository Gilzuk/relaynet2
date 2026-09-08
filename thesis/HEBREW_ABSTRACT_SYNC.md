# Hebrew abstract: four passages to write

The English abstract (`chapters/frontmatter.tex`) and the Hebrew one
(`chapters/hebrew_abstract.tex`) diverged during the 2026-09-07 port of the
un-ported `relaynet2-thesis` edits. Both abstracts received the
copilot-swe-agent rewrites, but the English one additionally received a hybrid
that kept a sentence those rewrites had deleted. The Hebrew never got that
hybrid, and three smaller pieces are missing with it.

Paragraph lengths show where: paragraphs 1 and 2 track (71/74 and 195/191
words), paragraphs 3 and 4 do not (205 vs 129, 207 vs 117). Roughly 165 words
are missing, all in the two paragraphs that were touched.

The Hebrew is the author's to write. Each item below gives the English source
and the exact anchor to place it against.

---

## 1. Paragraph 3 — the bounding claim and the BER-optimal result

**Insert** in line 27, after `...מגלאי Viterbi שעלותו גדלה כ־$M^{L}$.`
and before `בשכבה 3, בנקודת העבודה היחידה שנמדדה של 10~dB...`

English source (from `frontmatter.tex`):

> The claim is bounded deliberately: the learned relay does not beat a
> model-aware receiver that has been given the model, and it is not shown to.
> What it offers is near-optimal reliability at fixed arithmetic with no
> per-block channel identification stage, and that ordering is unchanged when
> the classical comparator is made BER-optimal rather than sequence-optimal.

This is the most important of the four. It is the thesis's central bounding
claim, and its final clause carries the BCJR/APP benchmark result. Verified
absent from the live Hebrew: `אינו עולה` occurs only in the commented-out
older block at lines 11-21.

Terms needing an author decision: *model-aware receiver*, *BER-optimal* vs
*sequence-optimal*, *per-block channel identification*.

## 2. Paragraph 2 — the hybrid relay is currently unhedged

**Replace**, in line 25:

> ממסר Hybrid העובר ל-DF מעל נקודת המעבר הנמדדת קרוב לאופטימלי ללא עלות נוספת, אם ה-SNR מוערך באמינות.

The Hebrew still says "near optimal, **at no extra cost**" and carries no
caveat. The English was narrowed and now reads:

> A hybrid relay switching to DF above the measured crossover performs near
> the best evaluated relay if SNR is reliably estimated; threshold robustness
> was not tested.

Two changes: "near optimal" becomes "near the best *evaluated* relay", and the
untested-robustness caveat is added. The Hebrew currently makes the stronger
claim of the two, which is the wrong direction for them to differ.

## 3. Paragraph 4 — the scope caveat

**Append** to line 29, after `...ולא הראה יתרון דיוק מדיד בניסוי זה.`

> Whether this holds at longer memory or under a tighter code is not tested
> here.

## 4. Paragraph 4 — the closing sentence

**Append** to line 29, after item 3.

> The regimes are delineated by identifiability and by arithmetic: classical
> processing where the model matches the channel and its memory is short, the
> learned relay where the channel is unmodeled, poorly estimated, missing per
> block, or has memory long enough to price the trellis out.

Term needing an author decision: *identifiability*.

---

## After editing

`hebrew_abstract.tex` is rendered, so the PDF must be rebuilt and committed in
the same commit (`CLAUDE.md`, PDF-must-not-lag rule):

```
cd thesis && latexmk -xelatex main.tex
```

Then re-check the paragraph balance; paragraphs 3 and 4 should land near the
English word counts rather than at two-thirds of them.

This file is working notes, not part of the document: `main.tex` does not
include it, so it never reaches the published Overleaf project.
