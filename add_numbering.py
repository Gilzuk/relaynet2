"""
add_numbering.py
----------------
Adds hierarchical section numbers to thesis_restructured.md headings.

Numbering scheme:
  ## Chapter N.  Title          (body chapters only, skip {.unnumbered})
  ### N.M  Sub-section
  #### N.M.K  Sub-sub-section

Skips:
  - Headings with {.unnumbered}
  - Headings that already start with a number (e.g. "### 4.1 ...")
  - References, Appendices (kept unnumbered)
"""

import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Identify which ## headings are numbered body chapters
# ─────────────────────────────────────────────────────────────────────────────
NUMBERED_CHAPTERS = [
    "Introduction and Literature Review",
    "Research Objectives",
    "Methods",
    "Experiments",
    "Discussion and Conclusions",
]

UNNUMBERED_SECTIONS = {
    "Table of Contents",
    "List of Figures",
    "List of Tables",
    "Equation Reference",
    "References",
    "Appendices",
    "Abstract (Hebrew)",
    "Abstract (English)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Process line by line
# ─────────────────────────────────────────────────────────────────────────────
lines = text.split("\n")
new_lines = []

# Counters: [chapter, section, subsection]
counters = [0, 0, 0]
current_chapter = 0  # 1-based index into NUMBERED_CHAPTERS

def is_unnumbered(title):
    """Check if heading should be skipped."""
    clean = re.sub(r"\{[^\}]*\}", "", title).strip()
    for u in UNNUMBERED_SECTIONS:
        if clean.startswith(u):
            return True
    return False

def already_numbered(title):
    """Check if heading already starts with a number like '4.1 ...'"""
    return bool(re.match(r"^\d+[\.\d]*\s", title.strip()))

for line in lines:
    # Match heading levels
    m2 = re.match(r"^(## )(.+)$", line)
    m3 = re.match(r"^(### )(.+)$", line)
    m4 = re.match(r"^(#### )(.+)$", line)

    if m2:
        prefix, title = m2.group(1), m2.group(2)
        if is_unnumbered(title) or already_numbered(title):
            # Reset counters so sub-sections of unnumbered chapters don't get numbered
            counters[0] = 0
            counters[1] = 0
            counters[2] = 0
            new_lines.append(line)
        else:
            # Find which chapter this is
            clean_title = re.sub(r"\{[^\}]*\}", "", title).strip()
            if clean_title in NUMBERED_CHAPTERS:
                idx = NUMBERED_CHAPTERS.index(clean_title)
                counters[0] = idx + 1
                counters[1] = 0
                counters[2] = 0
                new_lines.append(f"{prefix}{counters[0]}. {title}")
            else:
                new_lines.append(line)

    elif m3:
        prefix, title = m3.group(1), m3.group(2)
        if is_unnumbered(title) or already_numbered(title):
            new_lines.append(line)
        elif counters[0] == 0:
            new_lines.append(line)
        else:
            counters[1] += 1
            counters[2] = 0
            new_lines.append(f"{prefix}{counters[0]}.{counters[1]} {title}")

    elif m4:
        prefix, title = m4.group(1), m4.group(2)
        if is_unnumbered(title) or already_numbered(title):
            new_lines.append(line)
        elif counters[0] == 0 or counters[1] == 0:
            new_lines.append(line)
        else:
            counters[2] += 1
            new_lines.append(f"{prefix}{counters[0]}.{counters[1]}.{counters[2]} {title}")

    else:
        new_lines.append(line)

new_text = "\n".join(new_lines)

# ─────────────────────────────────────────────────────────────────────────────
# Update the TOC to match the new numbered headings
# ─────────────────────────────────────────────────────────────────────────────
# The TOC already has "1. [Introduction..." etc. — update sub-items
# Replace "   - 1.1 Cooperative..." with actual numbered sub-sections

# Build a map of old sub-section titles to new numbered titles
# by scanning the new headings
heading_map = {}
for line in new_lines:
    m3 = re.match(r"^### (\d+\.\d+) (.+)$", line)
    if m3:
        num = m3.group(1)
        title = m3.group(2)
        # Clean title (remove {.unnumbered} etc.)
        clean = re.sub(r"\{[^\}]*\}", "", title).strip()
        heading_map[clean] = f"{num} {clean}"

print(f"  Numbered {len(heading_map)} sub-sections")

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_text)

print(f"Output size: {len(new_text):,} chars")
print("Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────
print("\nVerification — ## headings:")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

for line in verify.split("\n"):
    if line.startswith("## ") or line.startswith("### "):
        if not line.startswith("#### "):
            print(f"  {line[:100]}")