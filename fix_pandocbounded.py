"""
fix_pandocbounded.py
--------------------
Fixes remaining LaTeX errors in thesis_tau.tex:
  1. \pandocbounded undefined → define as passthrough
  2. \tag not allowed here → remove remaining \tag{} commands
  3. Find and report other undefined control sequences
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Define \pandocbounded (pandoc 3.x image bounding command)
# ─────────────────────────────────────────────────────────────────────────────
if r"\pandocbounded" in tex and r"\newcommand{\pandocbounded}" not in tex:
    pandoc_defs = r"""
%% ── Pandoc 3.x compatibility ─────────────────────────────────
\newcommand{\pandocbounded}[1]{#1}
"""
    tex = tex.replace(r"\begin{document}", pandoc_defs + "\n" + r"\begin{document}", 1)
    count = tex.count(r"\pandocbounded")
    print(f"[1] Defined \\pandocbounded ({count} usages in document)")
else:
    print("[1] \\pandocbounded already defined or not used")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Remove any remaining \tag{N} commands (should have been removed earlier)
# ─────────────────────────────────────────────────────────────────────────────
tex, c = re.subn(r"\\tag\{[^\}]+\}", "", tex)
print(f"[2] Removed {c} remaining \\tag{{}} commands")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check for other undefined commands by looking at the log
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Checking log for specific undefined sequences ...")
try:
    with open("xelatex_run7.log", "r", encoding="utf-8", errors="replace") as f:
        log = f.read()
    
    # Find all "Undefined control sequence" with context
    undef_pattern = re.compile(
        r"! Undefined control sequence\.\s*\n(.*?)\n(l\.\d+ .*?)(?=\n)",
        re.DOTALL
    )
    undefs = undef_pattern.findall(log)
    print(f"  Undefined sequences: {len(undefs)}")
    for ctx, line in undefs[:10]:
        print(f"    {line[:100]}")
    
    # Find \tag errors
    tag_errors = [l for l in log.split("\n") if "tag" in l.lower() and ("!" in l or "Error" in l)]
    print(f"  Tag errors: {len(tag_errors)}")
    for e in tag_errors[:5]:
        print(f"    {e[:100]}")
        
except FileNotFoundError:
    print("  Log file not found")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fix \pandocbounded wrapping \includegraphics
#    pandoc generates: \pandocbounded{\includegraphics[...]{...}}
#    After our fix, this becomes: \includegraphics[...]{...}
#    But we need to make sure the includegraphics inside pandocbounded
#    also has the right options
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Checking \\pandocbounded usage ...")
pb_usages = re.findall(r"\\pandocbounded\{[^\}]{0,100}\}", tex)
print(f"  \\pandocbounded usages: {len(pb_usages)}")
if pb_usages:
    print(f"  Sample: {pb_usages[0][:100]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")