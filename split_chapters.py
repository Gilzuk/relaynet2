"""
split_chapters.py
-----------------
Splits thesis_tau.tex into separate chapter files:
  chapters/ch00_frontmatter.tex   (preamble + title + abstract + ack)
  chapters/ch01_introduction.tex
  chapters/ch02_objectives.tex
  chapters/ch03_methods.tex
  chapters/ch04_experiments.tex
  chapters/ch05_discussion.tex
  chapters/ch06_equation_ref.tex
  chapters/ch07_references.tex
  chapters/ch08_appendices.tex
  chapters/ch09_hebrew_abstract.tex

Main file thesis_tau.tex becomes a thin driver that \include{}s each chapter.
"""

import re, os

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

os.makedirs("chapters", exist_ok=True)

# ── Find split points ─────────────────────────────────────────────────────────
# Split on \chapter (numbered and unnumbered), but keep \frontmatter/\mainmatter/\backmatter
chapter_pattern = re.compile(r"(?=\\chapter[\*{])")
splits = [m.start() for m in chapter_pattern.finditer(tex)]

# Also find \frontmatter, \mainmatter, \backmatter positions
fm = tex.find("\\frontmatter")
mm = tex.find("\\mainmatter")
bm = tex.find("\\backmatter")
end_doc = tex.find("\\end{document}")

print(f"frontmatter: {fm}, mainmatter: {mm}, backmatter: {bm}")
print(f"Chapter split points: {splits}")

# ── Extract sections ──────────────────────────────────────────────────────────
# Preamble: from start to \begin{document} + frontmatter setup
begin_doc = tex.find("\\begin{document}")
preamble = tex[:begin_doc + len("\\begin{document}")]

# Front matter: from \begin{document} to first \chapter*{Abstract}
abstract_start = tex.find("\\chapter*{Abstract}")
front_setup = tex[begin_doc + len("\\begin{document}"):abstract_start]

# Abstract chapter
ack_start = tex.find("\\chapter*{Acknowledgments}")
abstract_content = tex[abstract_start:ack_start]

# Acknowledgments
mm_start = tex.find("\\mainmatter")
ack_content = tex[ack_start:mm_start]

# Main chapters (between \mainmatter and \backmatter)
main_content = tex[mm_start:bm]

# Back matter
back_content = tex[bm:end_doc]

# Find chapter boundaries in main content
chapter_starts = []
for m in re.finditer(r"\\chapter[\*{]", main_content):
    chapter_starts.append(m.start())

print(f"\nMain content chapters: {len(chapter_starts)}")

# Split main content into chapters
chapters_content = []
for i, start in enumerate(chapter_starts):
    end = chapter_starts[i+1] if i+1 < len(chapter_starts) else len(main_content)
    content = main_content[start:end]
    # Get chapter title
    title_m = re.match(r"\\chapter[\*{][^\n]{0,80}", content)
    title = title_m.group(0) if title_m else f"Chapter {i+1}"
    chapters_content.append((title, content))
    print(f"  [{i}] {title[:70]} ({len(content):,} chars)")

# ── Write chapter files ───────────────────────────────────────────────────────
chapter_files = []

# ch00: abstract + acknowledgments (front matter body)
ch00 = abstract_content + ack_content
with open("chapters/ch00_frontmatter.tex", "w", encoding="utf-8") as f:
    f.write(ch00)
chapter_files.append("chapters/ch00_frontmatter")
print(f"\nWrote ch00_frontmatter.tex ({len(ch00):,} chars)")

# Main chapters
chapter_names = [
    "ch01_introduction",
    "ch02_objectives",
    "ch03_methods",
    "ch04_experiments",
    "ch05_discussion",
    "ch06_equation_ref",
    "ch07_references",
    "ch08_appendices",
]

for i, (title, content) in enumerate(chapters_content):
    if i < len(chapter_names):
        fname = f"chapters/{chapter_names[i]}"
    else:
        fname = f"chapters/ch{i+1:02d}_extra"
    with open(fname + ".tex", "w", encoding="utf-8") as f:
        f.write(content)
    chapter_files.append(fname)
    print(f"Wrote {fname}.tex ({len(content):,} chars)")

# ch_backmatter: Hebrew abstract
with open("chapters/ch09_hebrew_abstract.tex", "w", encoding="utf-8") as f:
    f.write(back_content)
chapter_files.append("chapters/ch09_hebrew_abstract")
print(f"Wrote ch09_hebrew_abstract.tex ({len(back_content):,} chars)")

# ── Build new main thesis_tau.tex ─────────────────────────────────────────────
includes = []
includes.append("\\frontmatter")
includes.append("\\include{chapters/ch00_frontmatter}")
includes.append("")
includes.append("\\tableofcontents")
includes.append("\\clearpage")
includes.append("")
includes.append("\\cleardoublepage")
includes.append("\\phantomsection")
includes.append("\\addcontentsline{toc}{chapter}{\\listfigurename}")
includes.append("\\listoffigures")
includes.append("\\clearpage")
includes.append("")
includes.append("\\cleardoublepage")
includes.append("\\phantomsection")
includes.append("\\addcontentsline{toc}{chapter}{\\listtablename}")
includes.append("\\listoftables")
includes.append("\\clearpage")
includes.append("")
includes.append("\\mainmatter")
for fname in chapter_files[1:-1]:  # skip ch00 and ch09
    includes.append(f"\\include{{{fname}}}")
includes.append("")
includes.append("\\backmatter")
includes.append("\\include{chapters/ch09_hebrew_abstract}")
includes.append("")
includes.append("\\end{document}")

new_main = preamble + "\n\n" + front_setup.strip() + "\n\n" + "\n".join(includes) + "\n"

with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(new_main)

print(f"\nNew thesis_tau.tex: {len(new_main):,} chars")
print("Chapter files:")
for f in chapter_files:
    print(f"  \\include{{{f}}}")