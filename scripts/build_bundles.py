#!/usr/bin/env python3
"""Build the two Overleaf bundles from thesis/, containing only what main.tex compiles.

  thesis_overleaf.zip        annotated working copy (inline REV fix records)
  thesis_overleaf_clean.zip  submission copy, annotations stripped

What goes in is decided by scripts/overleaf_project.py, which resolves the
project from main.tex's include graph; see that module for why "the project" is
not simply "the thesis/ directory". The same definition backs the git branch
published by scripts/overleaf_sync.py, so a zip and a sync can never disagree
about what the project contains.

Usage:  python3 scripts/build_bundles.py     (or: make bundles)
"""
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overleaf_project import ROOT, manifest, stage


def build():
    man = manifest()
    for mode, name in (("annotated", "thesis_overleaf.zip"),
                       ("clean", "thesis_overleaf_clean.zip")):
        tmp = tempfile.mkdtemp(prefix=f"overleaf_{mode}_")
        try:
            stage(tmp, mode, man)
            out = os.path.join(ROOT, name)
            if os.path.exists(out):
                os.remove(out)
            # Deterministic: fixed entry timestamps and a stable order, so a
            # zip is a pure function of the project's contents. Staging into a
            # fresh directory gives every file a new mtime, which zip records,
            # so without this a rebuild that changed nothing still produced
            # different bytes -- dirtying the tree and putting another 4.7 MB
            # blob into history on every `make bundles`.
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for rel in sorted(os.path.relpath(os.path.join(base, f), tmp)
                                  for base, _, fs in os.walk(tmp) for f in fs):
                    info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    with open(os.path.join(tmp, rel), "rb") as fh:
                        z.writestr(info, fh.read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        entries = zipfile.ZipFile(out).namelist()
        missing = [f for f in man["styles"] if f not in entries]
        if missing:
            raise SystemExit(
                f"ERROR: {name} is missing local style file(s) {missing}, which "
                f"main.tex loads -- the bundle would not compile on Overleaf.")
        rev = sum(zipfile.ZipFile(out).read(n).decode(errors="ignore").count("\\REV{")
                  for n in entries if n.endswith(".tex"))
        print(f"  {name}: {len(entries)} entries, "
              f"{len(man['figures'])} figures, {rev} annotations")


if __name__ == "__main__":
    build()
