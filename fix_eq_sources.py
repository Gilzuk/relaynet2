"""
fix_eq_sources.py
-----------------
Fixes the broken equation source citations in thesis_tau.tex.
The citations appear as \footnotesize\{\textit{Source: \1}\}\normalsize
(pandoc escaped the braces and \1 is a literal backreference that wasn't substituted).

Fix: extract the actual citation texts from thesis_restructured.md and
substitute them in order into thesis_tau.tex.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract citation texts from thesis_restructured.md (original source)
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    md = f.read()

# Find all *Source: ...* patterns in order
citations = re.findall(r"\*Source: ([^\n]+)\*", md)
print(f"Found {len(citations)} source citations in thesis_restructured.md")
for i, c in enumerate(citations[:5]):
    print(f"  [{i+1}] {c[:80]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Read thesis_tau.tex
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Count broken citations
broken_pattern = r"\\footnotesize\\?\{\\textit\{Source: \\1\}\\?\}\\normalsize"
broken_count = len(re.findall(broken_pattern, tex))
print(f"\nBroken citations in thesis_tau.tex: {broken_count}")

if broken_count != len(citations):
    print(f"WARNING: count mismatch ({broken_count} broken vs {len(citations)} in md)")
    # Use min to avoid index errors
    n = min(broken_count, len(citations))
else:
    n = len(citations)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Replace broken citations one by one with actual text
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nReplacing {n} citations ...")

citation_idx = [0]  # mutable counter for closure

def replace_citation(m):
    idx = citation_idx[0]
    if idx < len(citations):
        cit = citations[idx]
        citation_idx[0] += 1
        # Format as a proper LaTeX note below the equation
        return f"\\par{{\\small\\textit{{Source: {cit}}}}}"
    return m.group(0)

tex_new = re.sub(broken_pattern, replace_citation, tex)

replaced = citation_idx[0]
print(f"  Replaced {replaced} citations")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex_new)

print(f"\nOutput: thesis_tau.tex ({len(tex_new):,} chars)")

# Verify
with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    verify = f.read()

remaining_broken = len(re.findall(broken_pattern, verify))
fixed_citations = len(re.findall(r"\\par\{\\small\\textit\{Source:", verify))
print(f"\nVerification:")
print(f"  Remaining broken citations: {remaining_broken}")
print(f"  Fixed citations: {fixed_citations}")

# Show first 3 fixed citations
for m in re.finditer(r"\\par\{\\small\\textit\{Source: ([^\}]+)\}\}", verify):
    print(f"  Sample: {m.group(0)[:100]}")
    if len(re.findall(r"\\par\{\\small\\textit\{Source:", verify[:m.end()])) >= 3:
        break