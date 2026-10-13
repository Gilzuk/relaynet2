# Committee-level audit of the thesis, 13 October 2026

**Baseline.** `main` at 82875bc, after PR #173 merged. `thesis/main.pdf` has 135 PDF pages, of which 119 are
countable (Chapter 1 through the appendices; the limit is 120). The Overleaf mirror `Gilzuk/relaynet2-thesis`
is at 748962e, which equals main 82875bc.

**Method.** Every rendered chapter (1–8), the appendices (9.1–9.6), both abstracts and the symbols list were
read the way an examining committee would read them. Each was checked:

- against the other chapters;
- against the committed code and result files;
- against numbers recomputed from those files.

Parameter counts were recomputed by hand or from instantiated models. Two bibliography attributions were
checked online.

**What this audit did not do.** It changed no thesis text, figure, table or result. Every finding is a report.
Each suggested fix is tagged with what it needs:

| Tag | Meaning |
|---|---|
| **[text]** | A wording change |
| **[text + figure]** | Also needs a figure regenerated |
| **[data]** | Needs a new run before any number can change |
| **[decision]** | The author's call |

Under `.clinerules/90-safety.md` and `CLAUDE.md`, no published number moves without new data and explicit
approval. Any fix that changes what the thesis renders must rebuild `thesis/main.pdf` in the same commit.

**Severity.**

- **A, substantive.** It overstates or contradicts a stated conclusion, or a methods statement contradicts the
  code. A committee member who checks will raise it.
- **B, inconsistency.** Two places disagree, thesis against thesis or thesis against data. The conclusions
  survive, but a reader cannot reconcile them.
- **C, minor.** Wording, labels, rounding or formatting.

Locations give `file:line` on main 82875bc and the rendered number (folio page).

## Outcome: fixes applied (branch `fix/committee-audit-findings`)

All findings were acted on in one PR, on the author's instruction "fix thesis".

**Build after the fixes:**

- 136 PDF pages, **120 countable** (the limit is 120; it was 119).
- 0 errors, undefined or multiply defined references, overfull boxes or missing glyphs.
- BibTeX: 0 warnings, 52 entries. The LeCun entry is no longer cited.
- Verifier: **708 cells** (30 new checks, 19 of them for the A2 rerun), 0 inconsistencies.
- 347 tests pass, and `provenance_audit.py` is clean.

| ID | Outcome |
|---|---|
| A1 | Fixed in both abstracts, Table 6.1, Sec. 6.3, Ch. 8 finding 9, the App. 9.5 heading and the README. The 10-pilot 95% interval (±0.0108) is now stated and checked. |
| A2 | Disclosed in Remark 4.1, the look-ahead item, the Transformer rationale, Ch. 2, Sec. 6.4.2 and App. 9.6. The Sec. 6.4.2 headline is qualified and the energy shares (42.5% and 34%) are checked from the taps. **Rerun (the author's choice):** a full-window and a bidirectional readout at the same parameter count both made the Mamba relays worse, so Table 6.7's conclusion stands. The rerun also found that the published spreads do not reproduce across machines, and that the Mamba-2 QPSK cell's best seed never reaches 10⁻³ (now footnoted). Sec. 6.4.2 reports all three. See `docs/mamba_lookahead_rerun_2026-10-13.md`. |
| A3 | Fixed: training is BPSK over AWGN; testing is over Rayleigh. |
| A4 | Fixed: Xavier for the PyTorch relays, PyTorch's default for the VAE, √(2/n_in) for the NumPy Chapter 6 MLPs. |
| A5 | Fixed: the Mamba-S6 and Mamba-2 blocks are described as implemented. |
| A6 | Source found: each value is the best-seed penalty of the best size-sweep configuration (window 7: 433, 433 and 289 parameters). The caption now defines the column, App. 9.6 lists its aggregation, and the verifier recomputes it (3 cells). The numbers are unchanged. |
| A7 | The nonlinear-bias result (S3, already committed) is now reported in Sec. 6.2: AF 0.0820, DF 0.0377, MLP 0.00108 at 8 dB, checked. H5 is kept; only "170-parameter" became "169–193 parameters". |
| B1–B17 | Fixed as suggested. For B5, H2's title is now neutral ("DF leads at medium-to-high SNR"). Its Table 8.1 verdict, at the author's choice, reads "Supported as matching: … none is detectably better". |
| C1–C22 | Fixed. C14 was more serious than first logged: 20.23% and 10.58% are in no committed file, so they now read 19.98% and 10.64%, the committed run of Table 5.7 (checked as `prose:coded-fer`). C18: 1.15 became 1.14 (the data give 1.1446). |
| C23 | Tsai et al. now cite IEEE SiPS 2020 and Chen et al. IEEE WCNC 2024. The co-author name order ("Wu Yan" in one paper, "Yan Wu" in the other) follows each arXiv listing and is left for the author. |

**Page budget.** To stay at 120, the new wording was kept short. Ch. 8's closing sentence listing untested items now points to Sec. 7.4, which lists the same items.

**Consistency with earlier changes:**

- **A4 reverses 7e3f933** (2026-09-14, terminology sweep B1). That commit changed "Xavier" to "the LeCun rule, not Xavier", stating that "the label was wrong, not the code". The code uses `xavier_uniform_`/`xavier_normal_`, so the original word was right and its formula wrong.
- **B5 adjusts 9dde972** (2026-09-11, examiner review). That review renamed H2 from "DF dominance" to "DF leads the evaluated symbol-wise classical relays". The new title keeps the softening (no "dominance") but drops a scope that contradicted H2's own statement.
- **No other fix contradicts an earlier change.** The rest complete earlier corrections:
  - A3 finishes the training-description fix of the Khina round;
  - B3 and B4 follow c5f9451's "feedforward learned relays" logic;
  - B15 finishes #169's single-AWGN Fig. 6.3;
  - B16 matches the seeding sentence #173 added.
- **Reply** (drafted, not sent), updated:
  - comment 1 lists the two new code mismatches;
  - comment 2 notes the S3 result;
  - comment 3 gives the margin as the MLP's;
  - the opening gives 120 pages and 708 cells;
  - the Mamba bullet reports the rerun.

## Summary

| ID | Sev. | Where | Finding | Fix |
|---|---|---|---|---|
| A1 | A | Both abstracts; Table 6.1; Sec. 6.3; Ch. 8 finding 9; App. 9.5 | "MLP leads at ten pilots" is not resolved by the thesis's own intervals | [text] |
| A2 | A | Sec. 4.2, 6.4.2; Table 6.7; App. 9.6 | The Mamba relays are causal scans read at the window centre, so they never see the five future samples | [text]; [data] to repair |
| A3 | A | Sec. 7.1.3 | Still says training uses the Rayleigh distribution | [text] |
| A4 | A | Sec. 4.2, 4.3.3 | The initialization paragraph (He/LeCun, "not Xavier") contradicts the code, which is Xavier throughout | [text] |
| A5 | A | Sec. 4.2 | The Mamba-S6 and Mamba-2 block descriptions do not match the implemented blocks | [text] |
| A6 | A | Table 6.6 | The "Learned relay" column has no committed source, the verifier skips it, and it conflicts with Table 6.7 | [data] + [text] |
| A7 | A | H5 (Sec. 3.3) | The principal hypothesis names a nonlinear-bias channel whose result is never reported | [text] or [data] |
| B1 | B | Sec. 6.2.2; Sec. 7.5 | "A plain-QPSK-ISI run was not performed", but Tables 6.6, 6.7 and 9.3–9.5 contain one | [text] |
| B2 | B | Tables 6.6, 6.7 | "Composite" is the size-sweep channel, not Sec. 6.2.4's, with noise at twice the convention and a linear MLSE | [text] |
| B3 | B | Sec. 5.2; Fig. 5.1 | "The two sequence models": there are three | [text] |
| B4 | B | Sec. 7.1.1; Ch. 8; abstracts | The sequence relays lose to the conventional AF at 0–4 dB; "0.007" and "track DF" are the MLP's | [text] |
| B5 | B | H2 in Sec. 3.3, 5.2, Table 8.1, Ch. 8 | H2 is stated four ways; "outperforming all AI methods" is not what the data show | [text] |
| B6 | B | Eq. 2.30 vs Sec. 5.5.4, Fig. 5.4 | Goodput is η(1−FER) in Ch. 2 and R(1−FER) in Ch. 5 | [text + figure] |
| B7 | B | Table 5.5 | The 20 dB MLP-coded W/L reads "99/1"; the data have 99 wins, 0 losses and 1 tie | [text] |
| B8 | B | Sec. 5.4; Table 9.4 | The size-sweep reference is the window-5 MLP-169 everywhere, not "each scenario's own published relay" | [text] |
| B9 | B | Table 9.4, Fig. 9.2 captions | The captions contradict the table's composite row | [text] |
| B10 | B | Table 6.7 caption | "Both Mamba variants become the least reproducible": Mamba-2 has the smallest BPSK spread | [text] |
| B11 | B | Sec. 6.2 (slicer floor) | The "measured" rows are said to come from Table 6.3's run, but the 12/16 dB cells differ | [text] |
| B12 | B | Sec. 6.2 vs Table 6.3 | Two MLP BERs at 12 dB on the same channel (1.09×10⁻⁴ and 1.456×10⁻⁴), sources not distinguished | [text] |
| B13 | B | Sec. 6.2 | Titled "Perfect CSI", but the trial list says the channel is "unknown to all relays" | [text] |
| B14 | B | Sec. 4.1 (ch04:73, 119) | The AWGN Hop-2 statements are incomplete, and an AWGN-only study does exist | [text] |
| B15 | B | Tables 9.2, 9.7 | QPSK-ISI Hop 2 is listed as "AWGN or Rayleigh"; only AWGN is reported | [text] |
| B16 | B | App. 9.2 vs Sec. 4.3.3 | Unseeded initialization: "sequence models" in one place, all canonical PyTorch relays in the other | [text] |
| B17 | B | Sec. 4.2, 4.4, Tables 4.2, 9.1 | The cGAN is described and normalized but never reported, and its counts include the critic | [text] |
| C1–C23 | C | various | Wording, labels, rounding and formatting (listed below) | [text] |

Two items affect the reply in `thesis/submission/reply_to_khina_2026-10.md`:

- **B4.** Comment 3 says "the learned relays' 0 dB margin shrinks from 0.073 to 0.007". That is the MLP's margin.
- **B12.** Comment 9 compares the scalar MAP with "1.1×10⁻⁴ at 12 dB". That value comes from the scalar-MAP run,
  not from Table 6.3.

No other item touches a number the reply cites.

---

## A. Substantive

### A1. The ten-pilot "lead" is not resolved, but both abstracts state it

**Locations:**

- `frontmatter.tex:11` (English abstract): "the MLP trails pilot-aided detection at twenty pilots and leads at ten".
- `hebrew_abstract.tex:11`, the same sentence ("…ומוביל בעשרה").
- `ch07_unknown_and_mismatch_channels.tex:27` (Table 6.1, layer 3): "by 10 pilots it has degraded to 0.0545 and
  the MLP leads".
- `ch07:266` (Sec. 6.3): "The measured ordering crosses between 20 and 10 pilots".
- `ch09_summary.tex:20` (finding 9).
- `appendices.tex:271` (App. 9.5 heading).

**Evidence.** From `e6_unknown_channel_results/e6_partial_ported_results.npy`; the second figure in each entry
is the 95% half-width that Sec. 6.3 quotes at five pilots:

| Case | Viterbi-est | MLP | Ordering |
|---|---|---|---|
| Panel (a), 20 pilots | 0.0335 ± 0.0022 | 0.0486 ± 0.0017 | Resolved: Viterbi better |
| Panel (a), 10 pilots | 0.0545 ± 0.0108, interval [0.044, 0.065] | 0.0486 ± 0.0017 | **Not resolved**: the intervals overlap |
| Panel (a), 5 pilots | 0.1235 ± 0.0304 | 0.0486 ± 0.0017 | Resolved: MLP better |
| Panel (b), 10 pilots per block | 0.0487–0.0501 at every block length | ≈0.049 | No difference; Fig. 6.5(b) and Sec. 6.3 already say so |

The thesis therefore contains both "leads at ten" and "no detected difference at ten pilots per block".

**Fix [text].** Say "trails at twenty pilots, is not resolved from it at ten, and leads at five" in:

- both abstracts;
- Table 6.1, Sec. 6.3 and finding 9;
- the App. 9.5 heading ("between twenty and five pilots").

O1's answer in Ch. 8 ("once pilots are scarce") can stay.

### A2. The Mamba relays have no look-ahead, so the sequence-on-memory comparison is confounded for them

**Code:**

- `checkpoints/checkpoint_20_mamba_s6_relay.py:111–119`: the S6 scan is a strictly causal loop.
- `checkpoint_20…:251–252`: the output is read at `center_idx = window_size // 2`.
- `checkpoints/checkpoint_23_mamba2_relay.py:141–142`: the SSD kernel is masked lower-triangular (`torch.tril`).
- `checkpoint_23…:355–356`: the output is read at the centre.

Both Mamba relays therefore decide x[n] from the current and five past samples only. The Transformer has no
attention mask, so it does see the whole window.

**Thesis:**

- `ch02_literature_review.tex:166` warns that a causal Transformer "would not see the right-hand half of the
  symmetric window this thesis feeds its relays", but says nothing about the Mamba relays.
- `ch04_methods.tex:236`: the 11-symbol window "provides the same temporal context as the Mamba-S6/SSD models".
- `ch04:157` defines look-ahead w > 0 for the windowed learned relays.
- `appendices.tex:316`: "a window of W = 11 samples centered on the decided symbol".

**Why it matters.** On the canonical memoryless channel it does not, because symbols and fades are i.i.d. On the
ISI channels of Sec. 6.4.2 (Table 6.7, `ch07:322–346`) it does. With taps ∝ [1, 0.7, 0.5], 42.5% of x[n]'s
energy arrives in y[n+1] and y[n+2], which a causal reader at the centre cannot see; with the composite's taps
∝ [1, 0.6, 0.4], 34%. The Mamba rows of Table 6.7 (+4.30/+4.55 dB BPSK, +5.72/+4.69 dB QPSK) are consistent
with that handicap. The paragraph's conclusion ("The prediction fails") is therefore tested cleanly only by the
Transformer, which also loses. The same holds for the "reproducibility inverts" observation.

**Fix.** [text]: disclose in Sec. 4.2 and 6.4.2 that both Mamba relays are causal and read the centre token, so
their effective look-ahead is zero, and restrict the Sec. 6.4.2 conclusion accordingly. [data], optional: rerun
the Mamba rows reading the last token of a window shifted to end at n + 5.

### A3. Sec. 7.1.3 still says the relays are trained on Rayleigh fading

**Location:** `ch08_discussion.tex:21`: "Training and testing use the specified Rayleigh distribution at sampled
SNRs without online adaptation."

**Contradicts:**

- Sec. 4.3.3 (`ch04:309`): BPSK over real AWGN, no fading.
- Sec. 4.2 (`ch04:211`).
- Sec. 7.1.1, second paragraph.
- The reply, comment 1 ("Section 7.1.1 no longer suggests that the relays learn an average over the fading").

**Fix [text].** "Training uses BPSK over AWGN at 5/10/15 dB; testing uses the specified Rayleigh distribution
at the sampled SNRs, without online adaptation."

### A4. The weight-initialization text contradicts the code

**Thesis:**

- `ch04:317` (Sec. 4.3.3): He initialization for ReLU layers and N(0, 1/n_in) "LeCun" for tanh layers,
  explicitly "not Xavier/Glorot".
- `ch04:211`: "He initialization is used for ReLU layers".

**Code:**

| Relay | Initialization | Where |
|---|---|---|
| Canonical MLP and Hybrid (PyTorch path, used per `ch04:315`) | `xavier_uniform_` | `relaynet/relays/genai.py:93`; the Hybrid builds a `MinimalGenAIRelay` |
| cGAN | `xavier_normal_` | `relaynet/relays/cgan.py:123, 145` |
| Transformer, Mamba-S6, Mamba-2 | `xavier_uniform_` | `checkpoint_18…:146`, `checkpoint_20…:220`, `checkpoint_23…:335` |
| VAE (PyTorch) | PyTorch's default `nn.Linear` initialization | `relaynet/relays/vae.py:37–43` |
| Chapter 6 MLP (tanh layers) | √(2/n_in) | `relaynet/relays/mlp.py:35–37` |

No path uses N(0, 1/n_in). This sits in the same section the reply describes as "now described from the
committed code".

**Fix [text].** Describe what each path does: Xavier-uniform for the PyTorch relays (Xavier-normal for the
cGAN), PyTorch's default for the VAE, and √(2/n_in) in the NumPy Chapter-6 MLP. Drop the LeCun sentence.

### A5. The Mamba block descriptions do not match the implemented blocks

**Mamba-S6.**

- Thesis (`ch04:237`): "LayerNorm → expand (32→64) → split to Conv1D/SiLU/S6 main branch and SiLU gate →
  contract (64→32) → residual".
- Code (`checkpoint_20…:126–167`): LayerNorm → `in_proj` → SiLU → S6 → `out_proj` → residual. There is no
  Conv1D and no gate branch.
- Inside the S6 layer, the state input is the mean over the 64 channels (a single scalar input, line 113). The
  input-dependent C is computed and then overwritten by `y = self.output_proj(state) + self.D * x` (lines
  116–117), so it has no effect.

**Mamba-2.**

- Thesis (`ch04:243`): "parallel gate/SSD branches → SiLU gate".
- Code (`checkpoint_23…:269–280`): SiLU on the SSD input branch, and a sigmoid gate.

The parameter counts are unaffected, because they are counted from the instantiated models.

**Fix [text].** Describe the implemented blocks and state that they simplify the reference Mamba and Mamba-2
designs. This is the same class of correction as the reviewer's SGD/Adam point.

### A6. Table 6.6's "Learned relay" column has no committed source

**Location:** Table 6.6 (`tbl:mmse-baseline`, `ch07:306–318`), column "Learned relay": +1.67, +3.03, −2.26 dB.
The caption derives its headline from it: "the learned relay leads the equalizer by 1.8 to 2.4 dB on every
channel".

**Evidence:**

- `verify_thesis_tables.py:1625` skips the column ("copied from E6 results").
- `results/mmse_equalizer.json` holds only the equalizer columns.
- No JSON under `results/` or `e6_unknown_channel_results/` contains any of the three values (searched to
  ±0.005).
- The column is defined as "the best learned relay on the same channel … window of 7". On ISI (QPSK), however,
  Table 6.7's MLP-3K reaches +2.58 dB, which is better than the column's +3.03. The +3.03 is therefore not the
  best learned relay on that channel.

**Fix:**

- [data] Commit the source: which relay, which seeds, which aggregation. Add a `REGISTRY` row in
  `provenance_audit.py` and a verifier check.
- [text] Define the column precisely. If the source cannot be recovered, remove the column and the "1.8 to
  2.4 dB" sentence, or recompute them from a committed run.

### A7. H5 names a channel whose result is never reported

**Location:** `ch03_objectives.tex:103` (H5, the principal hypothesis): "On the evaluated channels with memory
(intersymbol interference) **or asymmetry (biased nonlinearity)**, the memoryless classical relays … exhibit
error floors or severe degradation, whereas the 170-parameter MLP mitigates…".

**Evidence:**

- No section of Chapter 6 reports AF, DF or MLP results on the nonlinear-bias channel.
- The channel appears only in the size sweep (Tables 9.3–9.5, as a memoryless case against DF) and as a row of
  Table 9.7.
- The H5 verdict (Table 8.1) does not mention it.
- H5 also names the 170-parameter MLP, while its evidence uses the 193-parameter (QPSK), 169-parameter (flat,
  composite) and 1,153-parameter MLPs.

**Fix.** Either [text]: drop "or asymmetry (biased nonlinearity)" and say "a small MLP (169–193 parameters)". Or
[data]: report the nonlinear-bias result. `e6_sim_ported.py` runs that channel, but its committed result has not
been checked for this audit.

---

## B. Inconsistencies

### B1. Plain-QPSK ISI is said not to have been run, but it was

**The thesis says it was not run:**

- `ch07:180`: "Isolating the constellation would need a plain-QPSK-ISI run, which was not performed."
- `ch07:213`: such a control "is required".
- `ch08_discussion.tex:74` (future work): "plain-QPSK ISI would isolate questions left open".

**It was run.** Table 9.3's "ISI (complex)" channel is QPSK with the same taps, AWGN on Hop 2 and an exact-tap
QPSK MLSE. That channel is the "ISI (QPSK)" row of Table 6.6 (`mmse_equalizer.json`, key `isi_complex`) and of
Table 6.7 (`seq_models_on_memory.json`), and it is swept in Tables 9.4 and 9.5.

**Fix [text].** Say that plain-QPSK ISI was run in the size, MMSE and sequence studies, with different relays and
budgets from Sec. 6.2.2. Either cite it as the constellation control (MLP-3K: +2.58 dB against MLSE), or narrow
the statement to "not with the Sec. 6.2.2 relays and budget".

### B2. "Composite" in Tables 6.6 and 6.7 is a different channel from Sec. 6.2.4

Both tables use the size-sweep channel set (`mlp_min_size_all_channels.CHANNELS`). The two composites differ as
follows:

| | Tables 6.6, 6.7 (Table 9.3 row "composite") | Sec. 6.2.4 |
|---|---|---|
| Modulation | BPSK | DBPSK |
| Block phase | None in effect | Random |
| Noise variance | **1/γ, twice the convention** | Convention |
| Reference detector | MLSE given exact taps, i.e. a linear model mismatched to the amplifier | Pilot-LS Viterbi with differential decoding |
| Outcome | MMSE and the learned relays *beat* "MLSE" (negative penalties down to −2.26 dB) | Viterbi beats the MLP at 8 dB (0.0674 against 0.1073) |

**Why it matters.** Neither table says any of this. Read beside Sec. 6.2.4, and beside "Viterbi MLSE is the
optimal sequence detector" (`ch07:302`), the composite rows look contradictory. The project treats the SNR
convention as load-bearing, and the doubled noise is disclosed only in Table 9.3.

**Fix [text].** A caption note on both tables:

- channel definitions as in Table 9.3;
- the composite MLSE assumes a linear channel, which is why negative penalties occur;
- the composite noise variance is 1/γ.

### B3. "The two sequence models": there are three

**Locations:** `ch05_experiments.tex:92` ("The two sequence models are the ones that separate") and the Fig. 5.1
caption (`ch05:101`, "the two sequence models trail at low SNR").

**Evidence.** Table 5.2 has three sequence models: the Transformer, Mamba-S6 and Mamba-2 (0.4014, 0.4071 and
0.4032 at 0 dB).

**Fix [text].** Change "two" to "three" in both places.

### B4. Against the conventional AF, the sequence relays are worse at 0–4 dB, and "0.007" is the MLP's margin

**Evidence.** From Table 5.2, against the conventional AF of 0.3418 (0 dB) and 0.2450 (4 dB):

| Relay | 0 dB | 4 dB |
|---|---|---|
| Transformer | 0.4014 | 0.2758 |
| Mamba-S6 | 0.4071 | 0.2812 |
| Mamba-2 | 0.4032 | 0.2772 |

All three are worse than the conventional AF at both points.

**What the thesis says:**

- Sec. 7.1.1 (`ch08:9`) says only that "the sequence models are closer to AF", meaning the block-normalized AF.
- Ch. 8 finding 1 and the Table 8.1 H1 row say "the learned relays track DF". That holds for the feedforward
  relays only.
- The margin 0.007 is the MLP's (0.3418 − 0.3349). The best learned relay, the Hybrid, has 0.009.
- The English abstract says "learned feedforward relays". The Hebrew abstract (`hebrew_abstract.tex:9`) says
  only "ממסרים נלמדים" (learned relays).

H1 is phrased "some AI relays", so it still holds.

**Fix [text]:**

- Add one sentence to Sec. 7.1.1.
- Write "the feedforward learned relays" in Ch. 8 and Table 8.1.
- Attribute the 0.007 to the MLP, also in the reply's comment 3.
- Add "הזנה קדמית" to the Hebrew abstract.

### B5. H2 is stated four ways

**The four statements:**

| Where | Wording |
|---|---|
| `ch03:87` (title) | "DF leads the evaluated symbol-wise classical relays" |
| `ch03:87` (body) | "achieves the lowest BER … (≥6 dB), outperforming all AI methods" |
| `ch05:64` | "the best-performing evaluated symbol-wise classical relay" |
| Table 8.1 | "DF lowest-BER among per-symbol relays" (the learned relays are windowed, not per-symbol); result "matches or beats all AI methods" |

**Evidence.** In Table 5.2 the Hybrid is nominally below DF at 12 dB (0.0579 against 0.0580) and ties it at 16
and 20 dB. "Outperforming all AI methods" is therefore not supported; "matches or exceeds" is.

**Fix [text].** State H2 once, as "DF matches or outperforms every evaluated learned relay at ≥ 6 dB", and reuse
that wording in Ch. 5, Table 8.1 and Ch. 8.

### B6. Goodput is η(1−FER) in one chapter and R(1−FER) in another

**Locations:**

- Eq. 2.30 (`ch02:332`) and the symbols list (`symbols.tex:17`): G = η(1−FER), where η = R·log₂M (Eq. 2.29)
  and R = k/n.
- Sec. 5.5.4 (`ch05:392`): "G = R(1−FER)", with R "measured per channel symbol".
- The Fig. 5.4 y-label (`scripts/plot_coded_df.py:206`): "Goodput R(1−FER)".

Table 5.12 reaches 3.73 bits per symbol, which no code rate can reach.

**Fix [text + figure].** Use η in `ch05:392` and in the figure label, then regenerate Fig. 5.4 and rebuild.

### B7. Table 5.5: the "99/1" cell

**Location:** `ch05:203`, the MLP-coded row at 20 dB, W/L "99/1". The caption defines W/L as trials won/lost.

**Evidence:** `results/coded_high_budget_test.json` gives 99 wins, 0 losses and 1 tie
(`summary/20/mlp_coded/vs_viterbi`). The text ("at least 99 trials") is right.

**Fix [text].** "99/0 (1 tie)".

### B8. The size-sweep reference relay is not the "published relay" on the ISI channels

**Locations:** `ch05:146, 148` and the Table 9.4 caption: "against each scenario's own published relay (MLP-169,
or MLP-756 for the coded row)".

**Code.** `mlp_min_size_all_channels.py` (comments at lines 151 and 358–359) scores every row against the
in-sweep window-5, 24-hidden MLP-169. That includes the three ISI channels. The thesis's published ISI relays
are the window-11 MLP-170 (BPSK) and the 193-parameter MLP (QPSK). On those rows, "zero degradation" is
therefore measured against a window-5 control.

**Fix [text].** Write "against the 169-parameter window-5 reference configuration" and drop "published" for the
ISI rows.

### B9. Two captions contradict Table 9.4

- **Table 9.4 caption** (`appendices.tex:161`): "every evaluated three-tap row requires window 5 or 7 until a full
  decibel is conceded". The composite row has 21p w3 at ≤0.5 dB.
- **Fig. 9.2 caption** (`appendices.tex:195`): "the three-tap channels need between 21 and 145". The composite row
  has 6p at ≤2 dB.

**Fix [text].** Correct both captions to match the table.

### B10. Table 6.7 caption: Mamba reproducibility

**Location:** `ch07:330`: "the two Mamba variants … become the least reproducible here".

**Evidence.** In the ISI (BPSK) row, Mamba-2's spread (0.14 dB) is the smallest of the four. `ch07:344` already
says so.

**Fix [text].** Restrict the caption to Mamba-S6 on both ISI channels, Mamba-2 at QPSK, and both on the
composite.

### B11. The slicer-floor "measured" rows and Table 6.3 disagree at 12 and 16 dB

**Location:** `ch07:97` says the measured rows "come from the same three-seed run … that produces Table 6.3".

**Evidence.** Table 6.3's 12 and 16 dB AF/DF cells come from the fixed-budget reruns, as its caption says:

| Cell | Table 6.3 | "Measured" row |
|---|---|---|
| DF, 16 dB | 0.2280 | 0.2296 |
| AF, 16 dB | 0.2075 | 0.2067 |
| AF, 12 dB | 0.1835 | 0.1834 |

**Fix [text].** Write "the three-seed run behind Table 6.3's 8 dB column (its 12–16 dB AF/DF cells are the
fixed-budget reruns)".

### B12. Two MLP BERs at 12 dB on the fixed ISI channel

- **Scalar-MAP comparison** (`ch07:114`, and the reply's comment 9): 1.09×10⁻⁴. This comes from
  `results/e6_scalar_map.json`: MLPs retrained under the matched protocol, 3×10⁶ bits.
- **Table 6.3:** 1.456×10⁻⁴, from the fixed-budget checkpoints, 3×10⁷ bits.

The two values are 33% apart, and neither place says which run it uses.

**Fix [text].** Name the source at `ch07:114`, e.g. "(its own retrained MLPs; Table 6.3's fixed-budget value is
1.46×10⁻⁴)".

### B13. "Perfect CSI" and "unknown to all relays" in the same section

**Locations:** Sec. 6.2 is titled "Channel Memory with Perfect CSI", and Table 6.1 layer 2 reads "CSI still
perfect". The section's own trial list (`ch07:72`), however, says "Hop-1 channel (unknown to all relays)" and
lists genie-CSI Viterbi among those relays.

**Why it matters.** "Unknown" is used in three senses across the chapter:

- not supplied at inference;
- not in the canonical model;
- redrawn per block.

**Fix [text].** Write "Hop-1 channel (not given to AF, DF or the MLP; given to genie Viterbi)". Optionally
define "unknown" once in the chapter introduction.

### B14. AWGN on Hop 2: incomplete statements

**Locations:**

- `ch04:73`: "AWGN for the unknown-ISI comparison, Rayleigh otherwise".
- `ch04:119`: "AWGN also appears as the Hop-2 channel of one unknown-channel relay comparison … no standalone
  AWGN-only relay study is run".

**Evidence.** AWGN Hop 2 is used by the fixed ISI, QPSK ISI, nonlinear-bias and coded-memory studies (Table 9.7).
The size sweep also has an AWGN-only channel (Table 9.3, "awgn").

**Fix [text].** List the studies, or point to Table 9.7.

### B15. QPSK-ISI Hop 2 is listed as "AWGN or Rayleigh"

**Locations:** Table 9.7 (`appendices.tex:301`) and the ledger, Table 9.2 (`appendices.tex:110`).

**Evidence.** Since the Fig. 6.3 change, only the AWGN Hop 2 is reported (Table 6.4).

**Fix [text].** Write "AWGN (a compensated-Rayleigh variant is in the data file but not reported)".

### B16. Which relays were unseeded

**Locations:**

- App. 9.2 (`appendices.tex:74`): initialization "not consistently seeded for every original sequence-model
  checkpoint".
- Sec. 4.3.3 (`ch04:319`, added in #173): "the canonical training never seeds PyTorch's weight
  initialization". That covers the MLP, Hybrid and VAE as well.

**Fix [text].** Align App. 9.2 with Sec. 4.3.3.

### B17. The cGAN is described but never reported

**Where it appears:**

- A full design entry in Sec. 4.2.
- Sec. 4.4 says "all seven AI models were scaled to approximately 3,000 parameters", and Table 4.2 lists
  cGAN-3K.
- Table 9.1 lists the cGAN "as configured for the canonical comparison".

**The gap.** No reported comparison includes the cGAN. Ch. 1 says it was excluded on cost grounds, and Table 5.3
has six learned columns with no note explaining the missing one. Its counts also include the training-only
critic: 833 of 2,946 parameters, and 961 of 3,004 for cGAN-3K (`checkpoint_22_normalized_3k.py:17`).

**Fix [text].** Add one sentence in Sec. 4.2 or 5.3 stating the exclusion. Note that the cGAN counts include the
critic. Remove "as configured for the canonical comparison" from the cGAN row of Table 9.1.

---

## C. Minor

| ID | Location | Finding | Suggested fix |
|---|---|---|---|
| C1 | `ch04:283` and Sec. 4.1.5 vs `symbols.tex:23` | P_e and P_b both denote bit error probability; only P_e is in the symbols list | Use one symbol, or list both |
| C2 | `ch01:92` (Fig. 1.1) | "Canonical two-hop relay benchmark used throughout the thesis"; it is the canonical comparison only | "used in the canonical comparison" |
| C3 | `ch02:38` | "on the simulator's γ = E_b/N_0 axis"; the axis is E_s/N_0, equal to E_b/N_0 for BPSK only, as `ch02:216` and `ch05:94` say | "γ (= E_b/N_0 for BPSK)" |
| C4 | `ch02:23` | "AF … bottlenecked by the weaker hop" is true of DF as well; what is AF-specific is the noise accumulation | Reword |
| C5 | `ch02:171` | "four projection matrices per head"; it is Q, K and V per head plus one shared output projection (4·d_model² per layer) | Reword |
| C6 | `ch04:313` vs `ch04:441` | The in-range training points (6–14 dB) are given in nominal SNR, but each QPSK axis carries ±1/√2 against ±1 in training, a 3 dB shift in signal-to-noise per axis | One clause |
| C7 | `ch04:315` | "Adam-based supervised loss for sequence models"; Adam is the optimizer, and the loss is MSE (Table 9.1) | "MSE (Adam)" |
| C8 | `relaynet/relays/cgan.py:137` vs `ch04:226` | The critic also uses spectral normalization; the text does not say so | One clause |
| C9 | `ch05:62` | "with ReLU hidden and tanh output units": the Mamba blocks use SiLU, and Mamba-2 a sigmoid gate | Reword |
| C10 | Table 5.1 (`ch05:10–56`); `ch04:427` | (a) The experiment IDs E2, E3, E6, E7 and E8 are used with no E1, E4 or E5. (b) The table has E7 and E8 rows, though its caption says "core and unknown-channel" and `ch05:6` says the coded configuration is described separately. (c) `ch04:427` says BPSK appears "only in the unknown-channel study", but E8 is BPSK too | Renumber or explain; fix the caption and the BPSK sentence |
| C11 | `ch05:144` | "no evaluated model improves on the 169-parameter MLP": the Hybrid is nominally lower at every SNR in Table 5.2 | "no model improves on it detectably" |
| C12 | `ch05:158` | "the thesis-standard budget of 100 trials × 100,000" against Sec. 4.3.1 ("No single trial count … applies") and the canonical 10 × 10,000 | "this study's budget" |
| C13 | `ch05:158` | "window 21, ≈5×K for K = 3": 5K = 15, and 21 = 7K | Fix the parenthesis |
| C14 | `ch05:238` vs Table 5.7 | 20.23% coded-DF frame errors at 20 dB, against Table 5.7's 0.1998; different runs, not labelled | Name the run |
| C15 | Table 5.8 at 12 dB; Table 5.7 at 28 dB | Identical printed values, only one bold; Table 5.7's caption says "both marked where they tie" | Bold both |
| C16 | Fig. 5.5 caption (`ch05:465`) | "the four SNRs where the two relay strategies differ (16, 20, 24, 28 dB)"; Sec. 5.5.4 shows them differing at seven SNRs | "the four SNRs plotted" |
| C17 | `ch05:386` | "measured milliseconds"; Table 5.11 reports µs/symbol | "measured timings" |
| C18 | `ch05:260` | Retrained-MLP ratio at 28 dB printed as 1.15; the data give 1.1446 (1.14) | 1.14 |
| C19 | Table 6.1, layer 4 | Headline "Learned relay wins", against "narrowly leads" (Sec. 6.4) and "means are close" (Fig. 6.6); a resolved margin is shown only at 8 dB | "Learned relay narrowly leads" |
| C20 | `ch08:57`, `ch09:24` | "a three-symbol MLSE setting"; Table 6.8 shows D = 2 already gives the same BER | "two-symbol" |
| C21 | `appendices.tex:254` | "The channel used throughout is [1, 0.7, 0.5]"; that holds for the fixed ISI only | "The fixed-ISI channel" |
| C22 | `appendices.tex:316` vs Table 6.6 caption | "the learned relays of the uncoded Chapter 6 studies read a window of W = 11", but the Table 6.6 relay has window 7 and the size sweep uses 1–7 | Qualify |
| C23 | `references.bib:568, 578, 588` | (a) Tsai et al. (IEEE SiPS 2020) and Chen et al. and Karanov et al. (2024) are cited as arXiv preprints although published (WCNC 2024 for Chen et al.). (b) One co-author is "Wu, Yan" in Karanov2024 and "Yan, Wu" in Chen2024, so the bibliography renders two different people | Cite the venues; check the name against each paper |

---

## D. Verified, no action

- **Closed forms:**
  - The Table 5.2 "DF th." column, with relative errors 1.0%, 2.5% and 3.6% recomputed.
  - The ISI slicer amplitudes (1.6678, 0.9097, 0.6065, −0.1516) and the 0.25 floor. DF's measured 6 dB value
    matches once the omitted Hop-2 term is added (0.1801).
  - The ergodic-capacity column (2.40 at 8 dB, 12.46 at 40 dB).
  - The minimum exact Wilcoxon p for five pairs (0.0625), the rule of three, and t₀.₉₇₅ = 2.262 and 4.303.
- **Parameter counts** (by hand, or from instantiated models):

  | Model | Count |
  |---|---|
  | MLP | 169 |
  | MLP-3K | 3,004 |
  | VAE | 1,777 |
  | cGAN | 2,946 (2,113 + 833) |
  | cGAN-3K | 3,004 (2,043 + 961) |
  | Transformer | 17,697 |
  | Mamba-S6 | 24,001 (instantiated) |
  | Coded Mamba-S6 | 24,084 (instantiated) |
  | Mamba-2 | 26,179 |
  | MLP-170 | 156 MACs, ≈330 FLOPs |
  | MLP-756 | hidden width 16 |
  | QPSK MLP | 193 (hidden width 7) |
  | I/Q flat relay | 169 |
  | Coded-memory MLP | 220 (208 MACs) |

  The 3K reduction ratios (5.9×, 7.9×, 8.7×) and the "within 1.3%" claim also check out.
- **Coded study** (each row recomputed from its JSON):

  | Item | Checked values |
  |---|---|
  | Equal-SNR gain | 5.7× and 7.2× |
  | Table 5.10 gains | all |
  | Table 5.9 error counts in 3×10⁶ bits | 30, 3, 13 and 2 |
  | Matched block DF to oracle | within 2.4× everywhere |
  | Best learned relay at 20 dB | 2,667 errors |
  | Matched reliable regime | 1 error in 6×10⁶ bits at 20 dB, as many as the oracle; 0 at 24–32 dB |
  | Coded-DF to oracle ratios | 1.96 / 1.92 / 2.03 / 2.00 |
  | Table 5.5 p-values | 1.4×10⁻⁵ at the largest |
  | Goodput gaps | 0.105 at 24 dB, 6.7%; MCS spread 12× |
  | Readiness-budget ratios | 14.3×, 1.9×, 1.36× and 1.51× |
  | Timing ratios | 1.944, 1.941; 40% between machines |
- **Chapter 6** (each value recomputed):

  | Item | Checked values |
  |---|---|
  | Crossover arithmetic | L* = 3.35 / 2.90 / 3.76; 6.70; 5.70; ratios 1.23, 2.46 and 158; growth factor 256 |
  | MMSE leads | 2.37, 1.81 and 1.82 dB, matching "1.8 to 2.4" (subject to A6) |
  | Table 6.7 margins | 1.55, 2.11 and 0.32 dB |
  | Table 6.5 relative gains | all |
  | Pilot-sweep values | all |
  | Flat-control maximum difference | 0.0013 |
  | Hop-2 Rayleigh floors | 0.0061 and 0.0025 |
- **Bibliography:**
  - Gomadam and Jafar (IEEE JSAC 2007, preprint
    [arXiv cs/0510002](https://arxiv.org/abs/cs/0510002)): estimate-and-forward is optimal among memoryless relay
    functions for the paper's generalized SNR, and "for Gaussian inputs, AF, DF and EF become identical", as
    Sec. 2.2.4 says.
  - Tsai, Teng, Ou and Wu, [arXiv 2006.01125](https://arxiv.org/abs/2006.01125) (IEEE SiPS 2020): a neural
    network replaces the channel-model computation of the BCJR branch probabilities, as Sec. 2.2 says.
- **Environment claims:** Python 3.11.17, PyTorch 2.14.1 and 24 test modules all match App. 9.2.
- **Reply numbering:** every table and section number the reply cites maps to the right label (checked against
  the `.aux` files).

## Suggested order

1. **A1, A3, B3.** Text only, a few minutes each. A1 touches both abstracts.
2. **A4, A5.** Make the methods text match the code.
3. **A2.** Disclose. Decide whether to rerun the Mamba rows with a shifted window [decision].
4. **A6.** Commit a source for Table 6.6's learned-relay column, or remove it, and add a verifier check.
5. **A7, B1, B2.** Scope and label fixes in Chapters 3 and 6.
6. **The rest,** as time allows.

**Page budget.** The thesis has 119 countable pages against a limit of 120, and A2, A4, A5, B2 and B17 all add
text. Rebuild after each step and report the countable page count.

**Still open from earlier rounds** (author decisions):

- the title-page date ("September 2026", in four places in `main.tex`); **set to October 2026 by the author on 2026-10-13**;
- sending the reply, using the committed text rather than the pasted one;
- reviewing the Hebrew abstract (**approved by the author on 2026-10-13**);
- pulling the mirror into Overleaf.
