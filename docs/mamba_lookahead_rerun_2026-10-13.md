# Mamba look-ahead rerun (committee-audit finding A2), 2026-10-13

**Status:** integrated into the thesis in PR #175 at the author's direction (2026-10-13): Sec. 6.4.2 and
Table 6.7, with the Mamba-2 QPSK cell footnoted rather than the penalty definition changed. The last
section lists what changed.

## Question

Audit finding A2 (`docs/committee_audit_2026-10-13.md`, PR #175) found the following:

- Both Mamba relays scan their 11-sample window causally and read the output at its centre, so the
  decision on x[n] never sees y[n+1], …, y[n+5].
- On the three-tap channels of Table 6.7, those later samples carry part of x[n]'s energy: 42.5% on the
  ISI channels (taps ∝ [1, 0.7, 0.5]) and 34% on the composite channel (taps ∝ [1, 0.6, 0.4]).
- The audit suggested that the Mamba rows were consistent with that handicap.

PR #175 discloses the issue. This rerun asks whether the Mamba rows improve when the relays can see the
later samples, and whether the Table 6.7 conclusion still holds. That conclusion is that the
feedforward relay leads every sequence architecture on every channel.

## What was run

Both Mamba relays now take an optional `readout` argument. The default, `"center"`, is the published
behaviour, and its outputs are bit-identical to the earlier code. Every variant keeps the parameter
count (3,027 for Mamba-S6, 3,004 for Mamba-2), so the equal-budget comparison is preserved.

| Readout | What the decision on x[n] sees |
|---|---|
| `center` (published; rerun as the control) | y[n−5] … y[n], the causal half |
| `last` (the rerun suggested in the audit) | the whole window: the scan is read after y[n+5], and the residual still adds the centre sample |
| `bidirectional` | the whole window: the blocks scan the window and its reverse with shared weights, and both outputs are read at the centre and averaged |

**Protocol.** The protocol, seeds, training data and evaluation draws are those of
`seq_models_on_memory.py`, which produced Table 6.7:

- 20,000 samples, 40 epochs, batch 64, learning rate 10⁻³;
- training at 5, 10 and 15 dB;
- seeds 0, 1 and 2, with `torch.manual_seed` called before construction and again before training.

**Self-checks before any training.** Perturbing the five samples after the centre:

- leaves a centre-read model's output unchanged (difference 0.0);
- changes a last-read model's output (1.41 for S6, 1.88 for Mamba-2);
- changes a bidirectional model's output. Perturbing the five samples before the centre changes it too.

**Same-run controls.** The published values of the other two architectures cannot be mixed with new
Mamba values, because per-seed values do not carry across machines (see
Finding 3). MLP-3K and Transformer-3K were therefore rerun alongside.

**Determinism check.** Further runs retrained three configurations in fresh processes and compared
them with the committed runs (Finding 3).

**Agreement with the published run.** The MLSE baselines agree with the published ones exactly
(maximum difference 0.0) on all three channels, so every difference below comes from training.

| Script | Data | Console log |
|---|---|---|
| `seq_models_on_memory_readout.py` | `results/seq_models_on_memory_readout.json` | `results/seq_models_on_memory_readout_console_{isi,isi_complex,composite}.txt` |
| `seq_models_on_memory_bidirectional.py` | `results/seq_models_on_memory_bidirectional.json` | `results/seq_models_on_memory_bidirectional_console.txt` |
| `seq_models_on_memory_rerun_controls.py` | `results/seq_models_on_memory_rerun_controls.json` | `results/seq_models_on_memory_rerun_controls_console.txt` |
| `seq_models_on_memory_repeat.py` | `results/seq_models_on_memory_repeat.json` | `results/seq_models_on_memory_repeat_console.txt` |
| `seq_models_on_memory_repeat_probe.py` | `results/seq_models_on_memory_repeat_probe.jsonl` | `results/seq_models_on_memory_repeat_probe_console.txt` |

All five experiments are registered in `provenance_audit.py`. None of them backs a thesis artefact.

## Results

### Penalty at each BER target

The table gives the penalty against MLSE in dB, as the range over three seeds:

- "(k/3)" means only k seeds reach that target within 20 dB.
- "—" means no seed reaches it.
- On the composite channel MLSE itself reaches only 10⁻¹, so that is the only row.

All columns come from the committed reruns, made on one machine between 06:42 and 08:04 UTC.

| Channel | Target | MLP | Transformer | S6 centre | M2 centre | S6 bidir. | M2 bidir. | S6 last | M2 last |
|---|---|---|---|---|---|---|---|---|---|
| ISI, BPSK | 10⁻¹ | +0.56 to +0.59 | +1.60 to +1.67 | +2.03 to +2.10 | +2.12 to +2.15 | +5.53 to +5.98 | +4.23 to +5.40 | — | — |
| ISI, BPSK | 10⁻² | +1.92 to +2.31 | +3.70 to +4.60 | +4.32 to +4.44 | +4.80 to +5.03 | +11.73 to +14.16 (2/3) | +8.66 to +12.26 | — | — |
| ISI, BPSK | 10⁻³ | −1.66 to −1.26 | +1.02 to +3.01 | +1.26 to +1.62 | +1.98 to +2.61 | — | — | — | — |
| ISI, QPSK | 10⁻¹ | +0.52 to +0.55 | +0.97 to +1.22 | +1.63 to +1.70 | +1.66 to +1.74 | +4.67 to +5.27 | +4.38 to +4.72 | — | — |
| ISI, QPSK | 10⁻² | +1.70 to +1.85 | +2.96 to +3.74 | +4.05 to +4.13 | +4.19 to +4.39 | +9.95 (1/3) | +10.60 (1/3) | — | — |
| ISI, QPSK | 10⁻³ | +2.44 to +2.62 | +4.25 to +5.56 | +5.65 to +5.84 | +5.87 to +6.22 | — | — | — | — |
| Composite | 10⁻¹ | −2.08 to −2.01 | −1.85 to −1.76 | −1.55 to −1.49 | −1.48 to −1.46 | +1.36 to +1.57 | +1.07 to +1.29 | +8.20 (1/3) | — |

The negative MLP penalties at 10⁻³ on BPSK ISI are an artefact of the zero-error rule in Appendix 9.6. MLSE
records no errors at 12 dB, so its crossing is placed at 12 dB, which overstates the SNR it needs. The
published values show the same feature (−1.79 to −0.92).

**BER at 20 dB, range over seeds:**

| Channel | MLSE | Centre readout | Bidirectional readout | Last readout |
|---|---|---|---|---|
| ISI, BPSK | 0 | 0 to 1.3·10⁻⁵ | 1.3·10⁻³ to 1.0·10⁻² | 0.113 to 0.130 |
| ISI, QPSK | 0 | 3.0·10⁻⁵ to 9.8·10⁻⁵ | 7.0·10⁻³ to 2.0·10⁻² | 0.102 to 0.134 |
| Composite | 0.038 | 2.6·10⁻³ to 4.1·10⁻³ | 0.045 to 0.058 | 0.096 to 0.109 |

### Table 6.7 convention: published run against the rerun

Each entry is the best of three seeds, with the spread in brackets.

| Channel | Architecture | Published (Table 6.7) | Rerun |
|---|---|---|---|
| ISI, BPSK | MLP-3K | +1.97 (0.39) | +1.92 (0.38) |
| ISI, BPSK | Transformer-3K | +3.52 (0.33) | +3.70 (0.90) |
| ISI, BPSK | Mamba-S6-3K | +4.30 (1.15) | +4.32 (0.12) |
| ISI, BPSK | Mamba2-3K | +4.55 (0.14) | +4.80 (0.23) |
| ISI, QPSK | MLP-3K | +2.58 (0.07) | +2.44 (0.18) |
| ISI, QPSK | Transformer-3K | +5.96 (0.45) | +4.25 (1.31) |
| ISI, QPSK | Mamba-S6-3K | +5.72 (1.33) | +5.65 (0.19) |
| ISI, QPSK | Mamba2-3K | +4.69 (2.12)\* | +5.87 (0.35) |
| Composite | MLP-3K | −2.08 (0.03) | −2.08 (0.06) |
| Composite | Transformer-3K | −1.76 (0.03) | −1.85 (0.09) |
| Composite | Mamba-S6-3K | −1.56 (0.08) | −1.55 (0.06) |
| Composite | Mamba2-3K | −1.60 (0.06) | −1.48 (0.03) |

\* See Finding 4: this cell's best seed never reaches 10⁻³.

## Findings

**1. Neither readout that adds look-ahead helps the Mamba relays, so the Table 6.7 conclusion stands.**

- **Last readout.** It fails to train on every channel. At 20 dB its BER is about 0.10 to 0.13, against
  10⁻⁵ to 10⁻⁴ for the centre readout on ISI. Only one run of eighteen reaches any target: Mamba-S6
  seed 2 reaches 10⁻¹ on the composite channel, at +8.20 dB.
- **Bidirectional readout.** It trains, but it is worse than the centre readout at every target on every
  channel. It never reaches 10⁻³, and on the composite channel it falls behind MLSE, where the centre
  readout beats MLSE by about 1.5 dB.
- **Centre readout.** The published readout, which has no look-ahead, is the best of the three.
- **Feedforward relay.** It leads every architecture and readout on every channel.

The audit's suggestion that the Mamba deficit reflects the missing look-ahead is not supported for
either of these two ways of providing it.

Two caveats apply:

- These runs use the shared hyperparameters of Table 6.7. A readout with a longer path from the centre
  sample to the output may need a different learning rate or more epochs, and no tuning was attempted.
  Section 6.4.2 already carries this limitation for all four architectures.
- Why the extra samples hurt rather than help is not established by these runs.

**2. On the ISI channels, the published per-seed values and spreads do not reproduce.** The
best-of-three values agree within 0.25 dB in ten of the twelve cells. The two exceptions are both at
QPSK:

- the Transformer: +4.25 in the rerun against +5.96 published;
- Mamba-2: +5.87 against +4.69 (the cell of Finding 4).

The per-seed values differ by up to 1.73 dB on ISI (the Transformer, seed 2, at QPSK) and by at most
0.11 dB on the composite channel. The spreads therefore differ much more than the best values:

| Architecture, channel | Published spread | Spread here |
|---|---|---|
| Mamba-S6, BPSK ISI | 1.15 dB | 0.12 dB |
| Mamba-S6, QPSK ISI | 1.33 dB | 0.19 dB |
| Transformer, BPSK ISI | 0.33 dB | 0.90 dB |
| Transformer, QPSK ISI | 0.45 dB | 1.31 dB |

In this rerun the Transformer, not Mamba, has the widest spreads on ISI.

**3. Seeded training repeats bit for bit on one machine, but not on another machine with the same
software.**

The committed reruns all ran between 06:42 and 08:04 UTC. The session then moved to a new machine,
booted at 12:32. The filesystem carried over, including the installed PyTorch 2.14.1, so only the
hardware changed. Two scripts retrained configurations in fresh processes on the new machine:

- `seq_models_on_memory_repeat.py` retrained three configurations once each;
- `seq_models_on_memory_repeat_probe.py` retrained two of them twice more, alone on the machine. It
  records a hash of the trained weights and a description of the machine.

| Configuration | Earlier machine (committed) | Current machine | Published |
|---|---|---|---|
| MLP-3K, seed 0, BPSK ISI | +1.92 | +1.92, same BER curve | +2.36 |
| Mamba-S6-3K, seed 1, BPSK ISI | +4.32 | +4.10 in all three runs | +5.45 |
| Transformer-3K, seed 2, QPSK ISI | +4.25 | +5.13 in all three runs | +5.98 |

On the current machine the following hold:

- The two solo runs of each configuration give identical weight hashes, and their BER curves match
  the repeat's exactly. Training there is therefore deterministic.
- The Transformer repeat shared the CPU with other training jobs and still matched the solo runs bit
  for bit, so CPU contention is not the cause.
- The machine is a 4-core Intel Xeon with AVX-512 and AMX.

The earlier machine's CPU was not recorded, and neither was the published run's.

A likely mechanism is that the math libraries pick kernels by CPU instruction set, and the rounding
differences then compound over the roughly 12,500 steps of a training run. This is plausible but not
demonstrated here.

The consequence is that a training seed fixes a sequence model only on one machine:

- Across the two machines, the same seed moved by 0.21 dB for Mamba-S6 and by 0.89 dB for the
  Transformer.
- Against the published run, per-seed values moved by up to 1.73 dB.
- The MLP is the exception: it reproduced exactly.

A three-seed spread is therefore one draw from a distribution about as wide as the spreads themselves,
and it cannot rank the architectures by reproducibility.

**4. One published cell picks, as its best seed, the seed that never reaches 10⁻³.** This is Mamba-2 at
QPSK ISI, +4.69 (2.12), in Table 6.7.

- **What happened.** Seed 1 ends at BER 1.8·10⁻³ at 20 dB, so it never reaches 10⁻³. The penalty
  definition (Appendix 9.6) drops a target that a curve does not bracket. Seed 1's single number is
  therefore its 10⁻² penalty, 4.69. The other two seeds also reach 10⁻³, so theirs are 5.83 and 6.80.
- **Effect on the best.** The best-of-three is then the seed that performs worst at high SNR. Its 10⁻³
  penalty is more than 9.1 dB, since MLSE reaches 10⁻³ at 10.9 dB and seed 1 is still above it at 20 dB.
- **Effect on the spread.** The 2.12 dB spread compares a 10⁻² penalty with a 10⁻³ penalty.

If the comparison is restricted to targets that all three seeds reach, the published Mamba-2 QPSK values
are 4.43 to 4.69 at 10⁻². At 10⁻³, two seeds give 5.83 and 6.80 and the third gives no value.

This cell feeds the following thesis text:

- Section 6.4.2 says the feedforward relay leads by "2.11 dB" at QPSK. That figure is 4.69 − 2.58. If
  seed 1 is set aside because it misses 10⁻³, Mamba-2's best is +5.83. The runner-up is then Mamba-S6
  at +5.72, a margin of 3.14 dB.
- The "reproducibility inverts" paragraph cites "Mamba-2 by 2.12 dB at QPSK".
- In PR #175, the Table 6.7 caption says the Mamba variants "spread the most here, except Mamba-2 at
  BPSK".

The other eleven cells are not affected: in each, all three seeds reach the same targets. Nothing in this
cell weakens the headline, since the feedforward relay leads either way.

The first reading of the bidirectional runs in this session made the same mistake. Their best-of-three
at QPSK, +4.85 for S6 and +4.44 for Mamba-2, looks better than the centre readout (+5.65 and +5.87).
The per-target table shows the reason: those runs reach only 10⁻¹, where the centre readout's penalty
is +1.6 to +1.7 dB.

The same mechanism, a seed that misses a target scored over fewer targets, occurs in the canonical-channel
spread table (Table 9.6, `tbl:seed-spread-3k`):

- **VAE.** One seed reaches only 10⁻¹, so its 0.16 dB spread mixes targets.
- **Transformer.** Both seeds that set its 0.91 dB spread reach only 10⁻¹, so that comparison is
  like-for-like.

No claim there changes.

## What changed in the thesis (PR #175)

The author chose to fold these results into PR #175 and to footnote the Mamba-2 QPSK cell. All edits
are in `thesis/chapters/ch07_unknown_and_mismatch_channels.tex`, Sec. 6.4.2:

1. **Table 6.7.**
   - The Mamba-2 QPSK cell carries a dagger. The caption explains it: the initialization does not
     reach 10⁻³ by 20 dB, so its penalty covers 10⁻¹ and 10⁻² only, and the other two give +5.83 and
     +6.80.
   - The caption's claim that the Mamba variants "spread the most here" is replaced by a statement
     that the spreads did not reproduce in a rerun.
2. **Margin sentence.** "2.11 dB" now carries "(3.14 dB at QPSK without the marked initialization)".
3. **"Reproducibility inverts" paragraph.** It is replaced by the rerun's result:
   - ten of twelve best values reproduced within 0.25 dB, but not the spreads;
   - seeded training is bit-exact on one machine, but the same Transformer seed moved by 0.89 dB
     between machines;
   - of the three architectures retested, only the MLP retrained identically.
4. **Limitation paragraph.** One sentence is added: both look-ahead readouts made the Mamba relays
   worse under this protocol.

The page count is unchanged: 136 PDF pages, 120 countable.

`verify_thesis_tables.py` has a new check, `prose:seq-rerun`, with 19 cells. It recomputes every new
number and the three claims made in words. The verifier now covers 708 cells with 0 inconsistencies.
The provenance labels of the five experiments now name the Sec. 6.4.2 statements they back.
