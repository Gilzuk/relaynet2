#!/usr/bin/env python3
r"""One definition of what the Overleaf project *is*, shared by every consumer.

The Overleaf project is not "the thesis/ directory". thesis/ also carries the
working trail -- superseded chapter drafts (chapters/_*.tex), the review-response
appendices main.tex no longer includes, LaTeX build artefacts, changelogs, the
supervisor-comment log -- none of which the document compiles, and none of which
belongs in the authoring surface.

What the project *is*, is the transitive closure of main.tex: the chapters it
actually \include{}s, the figures those chapters actually \includegraphics{},
the bibliography, the bundled fonts, and any .sty that lives in thesis/ rather
than on CTAN. Everything else stays in the repository.

Two variants:
  annotated  the sources verbatim, \REV{...} fix records intact
  clean      the same document with the annotations stripped (they render as
             nothing either way -- \REV discards its argument -- so the two
             produce an identical PDF)

Consumers: scripts/build_bundles.py (the two zips) and scripts/overleaf_sync.py
(the git branch published to Overleaf).
"""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from strip_rev import strip_rev

THESIS = os.path.join(ROOT, "thesis")
MODES = ("annotated", "clean")

# Shipped at the project root. The repository this publishes into is a target,
# not a source: anyone who edits it directly loses that work at the next
# publish, and this is the notice that says so before it happens.
README = r"""# Deep Learning Architectures for Two-Hop Relay Communication

M.Sc. thesis · Tel Aviv University · Gil Zukerman · 2026
**Source:** `main.tex`, `\pdftitle` / `\pdfauthor`; compiled artefact `thesis.pdf` (136 pages).

---

## In one sentence

On a matched, memoryless two-hop relay channel a classical decode-and-forward relay is not improved on by any learned relay evaluated here, while on channels whose memory or impairments fall outside that model a compact learned relay restores reliable relaying — without ever overtaking a correctly informed classical sequence detector.
**Source:** `chapters/ch09_summary.tex`, closing paragraph — "A compact learned relay supplies useful amortized processing where the modelling or estimation requirements of the particular classical receiver it is measured against are not met".

## Research question

> Under what channel-information and channel-memory conditions can a learned relay improve upon or complement classical AF/DF processing in two-hop wireless communication?

**Source:** `chapters/ch01_introduction.tex`, §Scope — the thesis frames two questions whose boundaries are "the *reliability of the channel estimate*" and "*arithmetic*"; `chapters/ch03_objectives.tex`, §Two Research Questions.

---

## The experimental ladder

| Layer | Conditions | Classical comparator | Main result |
|---|---|---|---|
| 1 | Memoryless, known channel, perfect CSI | Symbol-wise DF, zero parameters | **Classical wins.** DF $0.1218$ against the MLP's $0.1229$ at 8 dB; the learned relay only matches DF, at 169 parameters against none |
| 2 | + 3-tap ISI, CSI still perfect | Viterbi MLSE with exact taps | **Split.** DF rises from $0.1802$ at 8 dB to $0.2457$ at 20 dB; the MLP restores the link ($0.0065$ at 8 dB) but genie-CSI Viterbi leads by 1–1.5 dB ($0.0072$) |
| 3 | + finite pilot budget | Pilot-aided LS estimate, then MLSE | **Crossover.** At 10 dB, Viterbi-est holds $0.0283$ on 200 pilots and $0.0335$ on 20, both beating the pilot-free MLP's $0.0486$; by 10 pilots it degrades to $0.0545$ and the MLP leads |
| 4 | + unseen realization from the trained family, redrawn per block | CMA blind equalization | **Learned relay wins.** At 20 dB the MLP reaches $0.00262$ against CMA's $0.00329$; decision-directed blind MLSE is unstable and worse at 20 dB ($0.0498$) than at 16 dB ($0.0374$) |

**Source for all four rows:** `chapters/ch07_unknown_and_mismatch_channels.tex`, Table `tbl:layers` ("The four layers of the argument").

---

## Canonical experiment

Two-hop SISO link, i.i.d. per-symbol Rayleigh fading on both hops, complex baseband, Gray-coded QPSK, uncoded, with receiver-side CSI used for one-tap zero-forcing equalization; the relay function is the only variable. Eight strategies — AF and symbol-wise DF against MLP, Hybrid, VAE, Transformer, Mamba-S6 and Mamba-2 — as Monte Carlo estimates with 95% confidence intervals.
**Source:** `chapters/ch04_methods.tex`, §System Model; scope table `tbl:assumptions-claims` in `chapters/ch01_introduction.tex`.

**Result:** DF is best or tied-best at every reported SNR from 6 dB upward, at zero parameter cost. Learned relays match it rather than beat it; the sequence models trail at low SNR on this memoryless channel.
**Source:** `chapters/ch05_experiments.tex`, Table `tbl:table2` ("BER on the canonical setup"); Figure `fig:fig10`.

*"Fast fading" here means the strict memoryless case, $h[i]\sim\mathcal{CN}(0,1)$ drawn independently per symbol — not a process with a finite Doppler spectrum.*
**Source:** `chapters/ch04_methods.tex`, §System Model.

## Unknown-channel experiment — the principal contribution

Against an unmodeled three-tap ISI filter, both evaluated memoryless classical relays (AF and zero-threshold symbol-wise DF) fail outright, DF's error rate rising as transmit power grows. A 170-parameter windowed MLP restores reliable relaying, reaching $4.79\times10^{-8}$ by 16 dB.
**Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, chapter opening; Figure `fig:figE6`.

The high-SNR floor of $0.25$ belongs to the evaluated detector and this tap vector, not to ISI in general: it is $k/4$ with $k$ the number of negative amplitudes $A(s_1,s_2)$, here one.
**Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, Eq. `eq:slicer-floor` and the two caveats following it.

The claim is bounded deliberately: *"The evaluated zero-threshold symbol-wise slicer cannot compensate for the selected ISI channel at any transmit power"* — which does not exclude symbol-by-symbol MAP detection.
**Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, chapter opening.

## Blind / unseen-realization regime

The relay receives no per-block channel estimate and never adapts online, but is trained and tested within the same parametric impairment family; layers 3 and 4 redraw realizations from that family.

> The supported claim is therefore **realization-agnostic**, not family-agnostic: the relay amortizes the family distribution into its weights, and generalization to a structurally different family is not tested.

**Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, §Layer-2 baselines paragraph (verbatim).

## Model size

A size sweep replicated over three initializations locates the smallest relay that costs nothing measurable and shows channel memory, not parameter count, sets it: four parameters suffice on the memoryless channel while the evaluated three-tap channels need 73 to 145 and a window spanning the interference. The median across-initialization spread over 152 configurations is $0.180$ dB, and no upward arm clears it.
**Source:** `chapters/appendices.tex`, §Appendix E, Table `tbl:seed-spread`; `chapters/ch08_discussion.tex`, §Does the Complexity Curve Turn Upward?

One architecture is unstable: the Transformer's across-initialization spread at the 3K budget is $0.906$ dB, five times the rest, traced to a single non-converged run in eight.
**Source:** `chapters/appendices.tex`, Table `tbl:seed-spread-3k`; Figures `fig:transformer-seed-curves`, `fig:transformer-loss-penalty`.

## Cost

Under the MAC-counting convention adopted in the thesis, full-state Viterbi MLSE requires $2M^{L}$ MAC-equivalents per symbol over $M^{L-1}$ states, against a learned relay's $O(LH)$ — linear against exponential in the memory. The two are equal near $L^{\star}=3.35$ taps for the deployed architecture: an arithmetic crossover at fixed $(W,H,M)$, not a universal threshold.
**Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, Eq. defining $L^{\star}$, §The Cost Side; `chapters/ch09_summary.tex`, finding 10.

---

## Figures worth reading first

| Figure | Source | Question it answers |
|---|---|---|
| `fig:fig10` | `chapters/ch05_experiments.tex` | Does any learned relay beat DF on the matched memoryless channel? |
| `fig:fig-3k-ber` | `chapters/ch05_experiments.tex` | At an equal parameter budget, do the architectures converge? |
| `fig:figE6` | `chapters/ch07_unknown_and_mismatch_channels.tex` | What happens to AF and DF when the channel has unmodeled memory? |
| `fig:minsize-crossover` | `chapters/appendices.tex` | How much window does a relay need as channel memory grows? |

---

## Conclusion

Classical processing remains preferable when its model assumptions are satisfied. A compact learned relay becomes attractive when channel memory, model mismatch, or unreliable channel estimation makes the *evaluated* classical processing chain poorly matched to the channel.
**Source:** `chapters/ch09_summary.tex`, closing paragraph; `chapters/ch08_discussion.tex`, §Interpretation of Results.

## What this thesis does not claim

- **Not a verdict on classical receiver design.** The comparators are the specific pipelines implemented here; reduced-state sequence estimation, decision feedback and hybrid model-based detectors were not evaluated. **Source:** `chapters/ch01_introduction.tex`, Table `tbl:assumptions-claims`, row "Comparator knowledge".
- **No generalization outside the trained impairment family.** **Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, "realization-agnostic, not family-agnostic".
- **No MIMO, no multiple relays, no full duplex, no constellation denser than 16-QAM as a relay question.** **Source:** `chapters/ch01_introduction.tex`, Table `tbl:assumptions-claims`, row "Channel".
- **No hardware, quantization or energy validation.** Parameter count bounds none of these. **Source:** `chapters/ch08_discussion.tex`, §Deployment footprint.
- **No confirmatory statistics.** Comparisons were not prespecified, no multiplicity correction is applied, and intervals cover Monte Carlo realizations at fixed weights rather than training-seed variability. **Source:** `chapters/ch04_methods.tex`, §Statistical Significance Testing.
- **Viterbi is not a BER bound.** MLSE minimizes sequence error; the BER-optimal comparator is bit-MAP/BCJR, measured at QPSK only and unmeasured at BPSK. **Source:** `chapters/ch07_unknown_and_mismatch_channels.tex`, §A BER-Optimal Benchmark: BCJR/APP Against Viterbi MLSE.

---

## Repository and build

This tree is **generated**. The source is `thesis/` in `Gilzuk/relaynet2`, and `scripts/overleaf_sync.py` there rebuilds it in full on every publish — anything changed here directly is overwritten at the next publish. `thesis.pdf` is build output, not a source; it is named `thesis.pdf` rather than `main.pdf` because Overleaf writes its own `main.pdf` when it compiles.

Compile with **XeLaTeX**, not pdfLaTeX — `fontspec` and `polyglossia` (the Hebrew abstract) require it. See `OVERLEAF.md` for the font and package-order notes.
"""


def no_comments(t):
    return re.sub(r'(?m)(?<!\\)%.*$', '', t)


def local_styles(sources):
    r"""Names of \usepackage'd style files that live in thesis/ rather than CTAN.

    Hardcoding this list is what broke the bundles once already: main.tex was
    switched from \usepackage{hebrewcal} to \usepackage{hebcal}, the builder
    went on copying hebrewcal.sty, and every bundle produced after that point
    was missing the .sty its own main.tex loads -- an Overleaf project that
    cannot compile. Derive it from the sources instead: any package for which
    thesis/<name>.sty exists is local and must travel with the project.
    """
    names = set()
    for text in sources:
        for m in re.finditer(r'\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]+)\}',
                             no_comments(text)):
            names |= {n.strip() for n in m.group(1).split(",")}
    return sorted(n + ".sty" for n in names
                  if os.path.exists(os.path.join(THESIS, n + ".sty")))


def manifest():
    """What the project consists of, resolved from main.tex. No files written."""
    main_src = open(os.path.join(THESIS, "main.tex")).read()
    inc = re.findall(r'\\(?:include|input)\{chapters/([^}]+)\}', no_comments(main_src))

    chapters, chapter_srcs, figs = [], [], set()
    for c in inc:
        p = os.path.join(THESIS, "chapters", c + ".tex")
        if not os.path.exists(p):
            continue                      # commented-out or renamed include
        src = open(p).read()
        chapters.append(c)
        chapter_srcs.append(src)
        figs |= {m.group(1) for m in re.finditer(
            r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', no_comments(src))}

    return {"main_src": main_src,
            "chapters": chapters,
            "chapter_srcs": chapter_srcs,
            "figures": sorted(figs),
            "styles": local_styles([main_src] + chapter_srcs)}


def stage(dest, mode, man=None):
    """Materialise the project into `dest` (created if absent, not cleared).

    Returns the manifest. Raises if a resolved style file did not land: a
    project missing a .sty its own main.tex loads cannot compile, and that
    must fail the build rather than ship.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    man = man or manifest()
    conv = (lambda s: s) if mode == "annotated" else strip_rev

    os.makedirs(os.path.join(dest, "chapters"), exist_ok=True)
    with open(os.path.join(dest, "main.tex"), "w") as fh:
        fh.write(conv(man["main_src"]))
    for name, src in zip(man["chapters"], man["chapter_srcs"]):
        dst = os.path.join(dest, "chapters", name + ".tex")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            fh.write(conv(src))
    shutil.copy(os.path.join(THESIS, "chapters", "references.bib"),
                os.path.join(dest, "chapters", "references.bib"))

    for f in man["figures"]:
        src_path = os.path.join(THESIS, f)
        if not os.path.exists(src_path):
            print(f"  WARNING: referenced figure missing: {f}")
            continue
        dst = os.path.join(dest, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src_path, dst)

    fonts_dst = os.path.join(dest, "fonts")
    if os.path.isdir(fonts_dst):
        shutil.rmtree(fonts_dst)
    shutil.copytree(os.path.join(THESIS, "fonts"), fonts_dst)

    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write(README)

    for extra in man["styles"] + ["OVERLEAF.md"]:
        src_path = os.path.join(THESIS, extra)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest)

    # The built PDF ships alongside the sources so the published repository can
    # be read without compiling it. It is named thesis.pdf rather than main.pdf
    # deliberately: Overleaf writes its own main.pdf as build output, and a
    # source file of that name in the project root collides with it.
    built = os.path.join(THESIS, "main.pdf")
    if os.path.exists(built):
        shutil.copy(built, os.path.join(dest, "thesis.pdf"))
    else:
        print("  WARNING: thesis/main.pdf not found; publishing sources only")

    missing = [f for f in man["styles"] if not os.path.exists(os.path.join(dest, f))]
    if missing:
        raise SystemExit(
            f"ERROR: staged project is missing local style file(s) {missing}, "
            f"which main.tex loads -- it would not compile on Overleaf.")
    return man
