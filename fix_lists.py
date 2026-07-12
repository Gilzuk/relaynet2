"""
fix_lists.py
------------
Fixes missing/broken List of Figures, List of Tables, and References in thesis_tau.tex.
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ── 1. Fix LOF: phantomsection + addcontentsline BEFORE \listoffigures ────────
tex, c1 = re.subn(
    r"\\listoffigures\s*\n\\addcontentsline\{toc\}\{chapter\}\{List of Figures\}\s*\n\\clearpage",
    r"\\cleardoublepage\n\\phantomsection\n\\addcontentsline{toc}{chapter}{\\listfigurename}\n\\listoffigures\n\\clearpage",
    tex
)
print(f"[1] LOF fix: {c1}")

# ── 2. Fix LOT: phantomsection + addcontentsline BEFORE \listoftables ─────────
tex, c2 = re.subn(
    r"\\listoftables\s*\n\\addcontentsline\{toc\}\{chapter\}\{List of Tables\}\s*\n\\clearpage",
    r"\\cleardoublepage\n\\phantomsection\n\\addcontentsline{toc}{chapter}{\\listtablename}\n\\listoftables\n\\clearpage",
    tex
)
print(f"[2] LOT fix: {c2}")

# ── 3. Remove duplicate Abstract (English) addcontentsline ────────────────────
abs_marker = r"\addcontentsline{toc}{chapter}{Abstract (English)}"
count = tex.count(abs_marker)
print(f"[3] Abstract TOC entries: {count}")
if count > 1:
    first = tex.find(abs_marker)
    second = tex.find(abs_marker, first + 1)
    tex = tex[:second] + tex[second:].replace(abs_marker, "", 1)
    print("    Removed duplicate")

# ── 4. Add phantomsection to front matter unnumbered chapters ─────────────────
for old, new in [
    (
        "\\chapter*{Abstract}\n\\addcontentsline{toc}{chapter}{Abstract (English)}\n\\markboth{Abstract}{Abstract}",
        "\\cleardoublepage\n\\phantomsection\n\\chapter*{Abstract}\n\\addcontentsline{toc}{chapter}{Abstract (English)}\n\\markboth{Abstract}{Abstract}"
    ),
    (
        "\\chapter*{Acknowledgments}\n\\addcontentsline{toc}{chapter}{Acknowledgments}\n\\markboth{Acknowledgments}{Acknowledgments}",
        "\\cleardoublepage\n\\phantomsection\n\\chapter*{Acknowledgments}\n\\addcontentsline{toc}{chapter}{Acknowledgments}\n\\markboth{Acknowledgments}{Acknowledgments}"
    ),
]:
    if old in tex and "\\cleardoublepage\n\\phantomsection\n" + old[:20] not in tex:
        tex = tex.replace(old, new, 1)
        print(f"[4] Added phantomsection to: {old[:40]}")

# ── 5. Add phantomsection before References chapter ───────────────────────────
ref_old = "\\chapter{References}\\label{sec:references}"
ref_new = "\\cleardoublepage\n\\phantomsection\n\\chapter{References}\\label{sec:references}"
if ref_old in tex and "\\phantomsection\n\\chapter{References}" not in tex:
    tex = tex.replace(ref_old, ref_new, 1)
    print("[5] Added phantomsection before References")

# ── 6. Write ──────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: {len(tex):,} chars")
print(f"listoffigures: {tex.count(chr(92)+'listoffigures')}")
print(f"listoftables:  {tex.count(chr(92)+'listoftables')}")
print(f"phantomsection: {tex.count(chr(92)+'phantomsection')}")