"""
fix_heb_dup.py
--------------
Fixes duplicate Abstract (Hebrew) TOC entry and markdown in Hebrew abstract.
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ── 1. Remove duplicate \addcontentsline for Abstract (Hebrew) ────────────────
marker = r"\addcontentsline{toc}{chapter}{Abstract (Hebrew)}"
count = tex.count(marker)
print(f"[1] Abstract (Hebrew) TOC entries: {count}")

if count > 1:
    # Keep only the FIRST occurrence (in the template's \chapter* block)
    first = tex.find(marker)
    # Remove all subsequent occurrences
    rest = tex[first + len(marker):]
    rest = rest.replace(marker, "")
    tex = tex[:first + len(marker)] + rest
    print(f"    Removed {count - 1} duplicate(s)")

# ── 2. Fix nested \begin{otherlanguage}{hebrew} (double-wrapped) ──────────────
print("\n[2] Checking nested otherlanguage blocks ...")
# Find the Hebrew abstract section
heb_start = tex.find(r"\begin{otherlanguage}{hebrew}")
if heb_start >= 0:
    # Check if there's another \begin{otherlanguage}{hebrew} inside
    inner = tex.find(r"\begin{otherlanguage}{hebrew}", heb_start + 1)
    if inner >= 0:
        # Find the matching \end{otherlanguage} for the inner one
        inner_end = tex.find(r"\end{otherlanguage}", inner)
        if inner_end >= 0:
            # Remove the inner \begin{otherlanguage}{hebrew} and its \end{otherlanguage}
            tex = (tex[:inner] 
                   + tex[inner + len(r"\begin{otherlanguage}{hebrew}"):inner_end]
                   + tex[inner_end + len(r"\end{otherlanguage}"):])
            print("  Removed nested otherlanguage block")
        else:
            print("  Could not find matching end for inner block")
    else:
        print("  No nested otherlanguage found")

# ── 3. Fix markdown bold (**text**) in Hebrew abstract ────────────────────────
print("\n[3] Fixing markdown bold in Hebrew abstract ...")
# Find Hebrew abstract section
heb_idx = tex.find(r"\begin{otherlanguage}{hebrew}")
if heb_idx >= 0:
    heb_end = tex.find(r"\end{otherlanguage}", heb_idx)
    if heb_end >= 0:
        heb_section = tex[heb_idx:heb_end]
        # Fix **text** → \textbf{text}
        fixed, c = re.subn(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", heb_section)
        if c > 0:
            tex = tex[:heb_idx] + fixed + tex[heb_end:]
            print(f"  Fixed {c} bold markdown patterns")
        else:
            print("  No markdown bold found")

# ── 4. Fix \begin{flushright} wrapping \chapter* (causes spacing issues) ──────
print("\n[4] Fixing flushright around chapter* ...")
old = r"""\begin{flushright}
\chapter*{\texthebrew{תקציר}}
\addcontentsline{toc}{chapter}{Abstract (Hebrew)}
\markboth{Abstract (Hebrew)}{Abstract (Hebrew)}
\end{flushright}"""

new = r"""\chapter*{\texthebrew{תקציר}}
\addcontentsline{toc}{chapter}{Abstract (Hebrew)}
\markboth{Abstract (Hebrew)}{Abstract (Hebrew)}"""

if old in tex:
    tex = tex.replace(old, new)
    print("  Removed flushright wrapper from Hebrew chapter*")

# ── 5. Write ──────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: {len(tex):,} chars")
marker_count = tex.count(r"\addcontentsline{toc}{chapter}{Abstract (Hebrew)}")
print(f"Abstract (Hebrew) TOC entries: {marker_count}")