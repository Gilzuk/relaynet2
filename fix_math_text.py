"""
fix_math_text.py
----------------
Fixes mixed math/text patterns where \$ is used with math commands.
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Fix patterns like \$\sim, \$\geq, \$\leq, \$\approx etc.
# These should be $\sim$, $\geq$, etc.
# ─────────────────────────────────────────────────────────────────────────────
fixes = [
    # \$\sim\( → $\sim$(
    (r"\\\$\\sim\\\(", r"$\\sim$("),
    # \)\times\$ → $)\\times$
    (r"\\\)\\times\\\$", r"$)\\times$"),
    # \$\sim\$ → $\sim$
    (r"\\\$\\sim\\\$", r"$\\sim$"),
    # \$\geq\$ → $\geq$
    (r"\\\$\\geq\\\$", r"$\\geq$"),
    # \$\leq\$ → $\leq$
    (r"\\\$\\leq\\\$", r"$\\leq$"),
    # \$\approx\$ → $\approx$
    (r"\\\$\\approx\\\$", r"$\\approx$"),
    # \$\times\$ → $\times$
    (r"\\\$\\times\\\$", r"$\\times$"),
    # Generic: \$\cmd\$ → $\cmd$
    (r"\\\$\\([a-zA-Z]+)\\\$", r"$\\\1$"),
    # Generic: \$\cmd( → $\cmd$(
    (r"\\\$\\([a-zA-Z]+)\\\(", r"$\\\1$("),
    # Generic: )\$\cmd → )$\cmd
    (r"\\\)\\([a-zA-Z]+)\\\$", r")$\\\1$"),
]

total = 0
for pattern, replacement in fixes:
    tex, c = re.subn(pattern, replacement, tex)
    if c > 0:
        print(f"  Fixed {c}: {pattern[:40]} → {replacement[:40]}")
        total += c

print(f"Total fixes: {total}")

# ─────────────────────────────────────────────────────────────────────────────
# Also fix \( and \) that are used as inline math delimiters
# but appear in text context causing issues
# Check for unmatched \( \) 
# ─────────────────────────────────────────────────────────────────────────────
# Count \( and \) 
open_count = tex.count(r"\(")
close_count = tex.count(r"\)")
print(f"\n\\( count: {open_count}, \\) count: {close_count}")
if open_count != close_count:
    print(f"  WARNING: Unmatched inline math delimiters!")

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")