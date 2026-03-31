"""
fix_split_main.py
-----------------
Fixes the split thesis_tau.tex:
  - Remove duplicate \frontmatter (already in front_setup)
  - Remove duplicate \cleardoublepage\phantomsection before ch00
  - Ensure correct structure
"""

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# The front_setup already contains \frontmatter, so remove the duplicate
# Current structure has:
#   ...dedication...\cleardoublepage\phantomsection
#   \frontmatter          ← DUPLICATE (already in front_setup)
#   \include{ch00_frontmatter}

# Fix: remove the duplicate \frontmatter line
old = """\cleardoublepage
\\phantomsection

\\frontmatter
\\include{chapters/ch00_frontmatter}"""

new = """\include{chapters/ch00_frontmatter}"""

if old in tex:
    tex = tex.replace(old, new)
    print("[1] Removed duplicate \\frontmatter")
else:
    # Try simpler replacement
    tex = tex.replace("\n\\frontmatter\n\\include{chapters/ch00_frontmatter}", 
                      "\n\\include{chapters/ch00_frontmatter}")
    print("[1] Removed duplicate \\frontmatter (simple)")

with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"Output: {len(tex):,} chars")

# Show the document body
bd = tex.find("\\begin{document}")
print("\nDocument body (from begin{document}+1400):")
print(tex[bd+1400:])