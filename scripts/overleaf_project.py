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

M.Sc. thesis, Tel Aviv University, Gil Zukerman, 2026.

This project is generated from `thesis/` in Gilzuk/relaynet2. Edit that source
repository, not this mirror. Compile `main.tex` with XeLaTeX; `thesis.pdf` is
the accompanying built artifact.

## Scope and main observations

The canonical experiment is uncoded QPSK on a two-hop SISO i.i.d. Rayleigh link.
DF matches or exceeds the tested learned relays from 6 dB upward. Separate
studies examine selected ISI/composite channels, finite-pilot and blind
receivers, and finite-length coded forwarding.

The fixed three-tap BPSK MLP has BER 0.0065 at 8 dB versus matched genie MLSE
0.001354, and 1.20e-7 at 16 dB versus below 3.0e-8 for the same detector at a
predeclared 1e8 bits. Matched MLSE therefore leads at every point where both
are resolved, and the required-SNR penalty grows with the target (0.26–2.53 dB
at the stated targets) rather than holding at a fixed offset. The 16-dB MLP
confirmation records 36 errors in 300M bits; the 18–20-dB MLP tail is not
validated.

The QPSK BCJR study measures relay-output BER only; it is not an end-to-end
BCJR-versus-MLP comparison. Zero observed errors are insufficient data for a
positive BER estimate, not proof of zero true BER. Nominal rule-of-three
bounds require an independent-bit assumption for their stated coverage.

The learned relays are trained on the impairment families tested. Their results
do not establish new-family generalization, a universal advantage over
classical receiver design, or hardware latency and energy savings. The separate
coded-QPSK cost study uses a 220-parameter, 208-MAC relay, not the 170-parameter
real BPSK model.

For configurations, uncertainty, and limitations, read the compiled thesis
and its source tables. Do not treat this short overview as a substitute for the
experiment-specific methods.
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
    main_src = open(os.path.join(THESIS, "main.tex"), encoding="utf-8").read()
    inc = re.findall(r'\\(?:include|input)\{chapters/([^}]+)\}', no_comments(main_src))

    chapters, chapter_srcs, figs = [], [], set()
    for c in inc:
        p = os.path.join(THESIS, "chapters", c + ".tex")
        if not os.path.exists(p):
            continue                      # commented-out or renamed include
        src = open(p, encoding="utf-8").read()
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
    with open(os.path.join(dest, "main.tex"), "w", encoding="utf-8") as fh:
        fh.write(conv(man["main_src"]))
    for name, src in zip(man["chapters"], man["chapter_srcs"]):
        dst = os.path.join(dest, "chapters", name + ".tex")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
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

    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as fh:
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
