"""
fix_heb_final.py
----------------
Directly replaces the backmatter section with correct nesting.
"""

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

bm = tex.find("\\backmatter")
end_doc = tex.find("\\end{document}")

# Extract the Hebrew content (the actual text paragraphs)
# It starts after \begin{otherlanguage}{hebrew} (the inner one)
inner_heb = "\\begin{otherlanguage}{hebrew}\n\\textbf{"
inner_start = tex.find(inner_heb, bm)
# The content starts at \textbf{
content_start = tex.find("\\textbf{", inner_start)
# The content ends at \end{flushright}
content_end = tex.rfind("\\end{flushright}", bm, end_doc)

heb_content = tex[content_start:content_end].strip()
print(f"Extracted Hebrew content: {len(heb_content)} chars")
print(f"First 100: {heb_content[:100]}")

# Build the correct backmatter
new_backmatter = (
    "\\backmatter\n"
    "\\clearpage\n"
    "\\chapter*{\\texthebrew{תקציר}}\n"
    "\\addcontentsline{toc}{chapter}{Abstract (Hebrew)}\n"
    "\\markboth{Abstract (Hebrew)}{Abstract (Hebrew)}\n"
    "\n"
    "\\begin{otherlanguage}{hebrew}\n"
    "\\begin{flushright}\n"
    "\\label{sec:abstract-hebrew}\n"
    "\n"
    + heb_content + "\n"
    "\n"
    "\\end{flushright}\n"
    "\\end{otherlanguage}\n"
    "\n"
)

tex = tex[:bm] + new_backmatter + tex[end_doc:]

with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: {len(tex):,} chars")

# Verify
with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("\nLines 4020-4050:")
for i, l in enumerate(lines[4019:4055], start=4020):
    print(f"{i:5d}: {l[:120]}", end="")