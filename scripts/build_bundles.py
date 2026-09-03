#!/usr/bin/env python3
"""Build the two Overleaf bundles from thesis/, containing only what main.tex compiles.

  thesis_overleaf.zip        annotated working copy (inline REV fix records)
  thesis_overleaf_clean.zip  submission copy, annotations stripped

Both are self-contained: main.tex, the compiled chapters, references.bib, every
referenced figure, the bundled fonts, every local .sty main.tex loads, and
OVERLEAF.md. Files that main.tex does not include (drafting history, superseded
figures) are left out of the bundles but kept in the repository.

Usage:  python3 scripts/build_bundles.py     (or: make bundles)
"""
import os, re, shutil, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from strip_rev import strip_rev

THESIS = os.path.join(ROOT, "thesis")


def no_comments(t):
    return re.sub(r'(?m)(?<!\\)%.*$', '', t)


def local_styles(sources):
    r"""Names of \usepackage'd style files that live in thesis/ rather than CTAN.

    Hardcoding this list is what broke the bundles once already: main.tex was
    switched from \usepackage{hebrewcal} to \usepackage{hebcal}, the script
    went on copying hebrewcal.sty, and every bundle produced after that point
    was missing the .sty its own main.tex loads -- an Overleaf project that
    cannot compile. Derive it from the sources instead: any package for which
    thesis/<name>.sty exists is local and must travel with the bundle.
    """
    names = set()
    for text in sources:
        for m in re.finditer(r'\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]+)\}',
                             no_comments(text)):
            names |= {n.strip() for n in m.group(1).split(",")}
    return sorted(n + ".sty" for n in names
                  if os.path.exists(os.path.join(THESIS, n + ".sty")))


def build():
    main_src = open(os.path.join(THESIS, "main.tex")).read()
    inc = re.findall(r'\\(?:include|input)\{chapters/([^}]+)\}', no_comments(main_src))

    figs = set()
    chapter_srcs = []
    for c in inc:
        p = os.path.join(THESIS, "chapters", c + ".tex")
        if os.path.exists(p):
            src = open(p).read()
            chapter_srcs.append(src)
            figs |= {m.group(1) for m in re.finditer(
                r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', no_comments(src))}

    styles = local_styles([main_src] + chapter_srcs)

    for mode, name in (("annotated", "thesis_overleaf.zip"),
                       ("clean", "thesis_overleaf_clean.zip")):
        stage = os.path.join("/tmp", f"_bundle_{mode}")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(os.path.join(stage, "chapters"))
        conv = (lambda s: s) if mode == "annotated" else strip_rev

        open(os.path.join(stage, "main.tex"), "w").write(conv(main_src))
        for c in inc:
            p = os.path.join(THESIS, "chapters", c + ".tex")
            if os.path.exists(p):
                dst = os.path.join(stage, "chapters", c + ".tex")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "w").write(conv(open(p).read()))
        shutil.copy(os.path.join(THESIS, "chapters", "references.bib"),
                    os.path.join(stage, "chapters", "references.bib"))
        for f in figs:
            src = os.path.join(THESIS, f)
            if not os.path.exists(src):
                print(f"  WARNING: referenced figure missing: {f}")
                continue
            dst = os.path.join(stage, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        shutil.copytree(os.path.join(THESIS, "fonts"), os.path.join(stage, "fonts"))
        for extra in styles + ["OVERLEAF.md"]:
            src_path = os.path.join(THESIS, extra)
            if os.path.exists(src_path):
                shutil.copy(src_path, stage)

        out = os.path.join(ROOT, name)
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, fs in os.walk(stage):
                for f in sorted(fs):
                    fp = os.path.join(root, f)
                    z.write(fp, os.path.relpath(fp, stage))

        entries = zipfile.ZipFile(out).namelist()
        missing = [f for f in styles if f not in entries]
        if missing:
            raise SystemExit(
                f"ERROR: {name} is missing local style file(s) {missing}, which "
                f"main.tex loads -- the bundle would not compile on Overleaf.")
        rev = sum(zipfile.ZipFile(out).read(n).decode(errors="ignore").count("\\REV{")
                  for n in entries if n.endswith(".tex"))
        print(f"  {name}: {len(entries)} entries, {len(figs)} figures, {rev} annotations")


if __name__ == "__main__":
    build()
