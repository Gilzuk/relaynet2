# Hebrew abstract: approved by the author (2026-10-13; margin change approved 2026-10-14)

**Status 2026-10-13: approved.** The author reviewed the rendered Hebrew abstract (page xii of the
#175 build) and the two changes of the committee-audit round below, and approved them ("Hebrew ok").
The earlier notes below are kept as the record of how the text was drafted.

**Update 2026-10-14 (red-team findings, R2) — approved by the author on 2026-10-14 ("Approve Hebrew").** One number in the
approved text changed, mirroring the English abstract: the MLP's 0 dB margin against conventional
AF is now "only 0.003–0.007, depending on the gain rule" (a fixed-gain AF does better than the
variable-gain one at 0 dB). The drafted Hebrew reads
"…הוא \textenglish{0.003--0.007} בלבד, בהתאם לכלל ההגבר;" and is approved as it stands in the merged build (main fe8198b, mirror abf77f8).

**Update 2026-10-13 (committee audit).** Two more changes to the English
abstract were mirrored, in drafted Hebrew for the author to read:
- the pilot sentence now says the learned relay trails pilot-aided detection at
  twenty pilots, is not resolved from it at ten and leads at five
  ("אינו נבדל ממנו באופן מובהק בעשרה ומוביל בחמישה"); the ten-pilot difference is
  inside the stored 95% interval;
- the low-SNR sentence now names the feedforward relays ("בהזנה קדמית") and
  attributes the 0.007 margin to the MLP ("הפער של ממסר ה־MLP").

**Update 2026-10-12 (Khina revision round).** Two changes to the English
abstract were mirrored in `chapters/hebrew_abstract.tex`, again in drafted
Hebrew that the author should read:
- the qualifier that the learned relays' low-SNR advantage is over the
  implemented block-normalized AF, with the 0.007 margin at 0 dB against a
  conventional variable-gain AF;
- the coded-study sentence: with reliability-weighted (matched) decoder metrics,
  block DF beats the learned relays at every tested SNR.

The Hebrew abstract renders on page 134 of the 136-page build (printed page
xii); the rest of this note describes the 2026-09-09 rewrite.

**Status 2026-09-09: drafted, not author-approved.** The Hebrew abstract was
505 words over a page and a half, translating the superseded 695-word English
abstract. It has been rewritten to 321 words on one page, mirroring the new
English abstract's five paragraphs.

**This draft is not the author's own Hebrew and must be read before submission.**
It was built by reusing the existing Hebrew wording verbatim wherever the concept
survived the English rewrite, and composing only the connective text and the new
opening question. Established terminology was carried over unchanged --
ממסר, תקשורת דו־דילוגית, דעיכת ריילי מהירה בלתי־תלויה ושווה־התפלגות, פס־בסיס מרוכב,
הגבר־והעבר, פענח־והעבר סימבולי, זיכרון הערוץ, מקדמים, פיילוטים, פעולות כפל־צבירה,
משפחת הפגמים -- so the vocabulary is the author's even where the sentences are
not.

## What to check

- The opening question in paragraph 1 is entirely new; there was no Hebrew for it.
- Paragraph 5 (the bound on the claim) is new; the old Hebrew ended differently.
- Bidirectional rendering: Latin acronyms, `$math$` and units inside Hebrew
  follow the patterns already used in the file, but line breaking around them is
  worth eyeballing in the built PDF (page 145).

## What was dropped, matching the English

The coded study (rate-1/2 convolutional code, block-DF, Mamba-S6 at 16 dB, the
modulation-and-coding grid), the four-layer ladder enumeration, and the
traceback-depth latency detail. All remain in the body.

## Result

145 pages total; the Hebrew abstract occupies one page (145) after its title page
(144), down from two. The countable body is unchanged at 130.
