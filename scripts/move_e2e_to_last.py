"""
Move E2E section from 7.14 to 7.16 (last experiment).
Renumber sections, figures, tables, and findings accordingly.

Current order: 7.14(E2E), 7.15(CSI), 7.16(Multi-CSI)
New order:     7.14(CSI),  7.15(Multi-CSI), 7.16(E2E)

Renumbering plan:
  Sections: 7.15 → 7.14, 7.16 → 7.15, 7.14(E2E) → 7.16
  Figures:  43-49 → 39-45 (CSI multi-arch), 39-42 → 46-49 (E2E)
  Tables:   19 → 18, 20→19, 21→20, 22→21, 23→22, 18→23 (E2E)
  Findings: 22 → 20, 23→21, 24→22, 25→23, 26→24, 20→25, 21→26
"""

import re
from pathlib import Path

thesis = Path(__file__).resolve().parent.parent / "thesis.md"
text = thesis.read_text(encoding="utf-8")

# --- Step 1: Use temporary placeholders to avoid collisions ---

# Section renumbering: use §TEMP_XX placeholders
# 7.14.x (E2E) → TEMP_16.x
# 7.15.x (CSI) → TEMP_14.x
# 7.16.x (Multi) → TEMP_15.x

# Handle subsections first (7.14.1, 7.14.2, etc.), then main sections
for sub in ["1", "2", "3", "4", "5", "6"]:
    text = text.replace(f"7.14.{sub}", f"§TEMP_16.{sub}")
    text = text.replace(f"7.15.{sub}", f"§TEMP_14.{sub}")
    text = text.replace(f"7.16.{sub}", f"§TEMP_15.{sub}")

# Main sections - use word boundary approach
text = re.sub(r'(?<!\.)7\.14(?!\.?\d)', '§TEMP_16', text)
text = re.sub(r'(?<!\.)7\.15(?!\.?\d)', '§TEMP_14', text)
text = re.sub(r'(?<!\.)7\.16(?!\.?\d)', '§TEMP_15', text)

# Figure renumbering: use FIGTEMP_XX placeholders
# 39-42 (E2E) → FIGTEMP_46..49
# 43-49 (Multi) → FIGTEMP_39..45
for old_fig in range(49, 42, -1):  # 49..43 → 39..45
    new_fig = old_fig - 4
    text = re.sub(rf'Figure {old_fig}(?!\d)', f'FIGTEMP_{new_fig}', text)

for old_fig in range(42, 38, -1):  # 42..39 → 46..49
    new_fig = old_fig + 7
    text = re.sub(rf'Figure {old_fig}(?!\d)', f'FIGTEMP_{new_fig}', text)

# Table renumbering: use TABTEMP_XX
# 18 (E2E) → TABTEMP_23
# 19 → TABTEMP_18, 20→19, 21→20, 22→21, 23→22
for old_tab in range(23, 18, -1):  # 23..19 → 22..18
    new_tab = old_tab - 1
    text = re.sub(rf'Table {old_tab}(?!\d)', f'TABTEMP_{new_tab}', text)

text = re.sub(r'Table 18(?!\d)', 'TABTEMP_23', text)

# Finding renumbering: use FINDTEMP_XX
# 20,21 (E2E) → FINDTEMP_25,26
# 22 → FINDTEMP_20
# 23→21, 24→22, 25→23, 26→24
for old_f in range(26, 22, -1):  # 26..23 → 24..21
    new_f = old_f - 2
    text = re.sub(rf'Finding {old_f}(?!\d)', f'FINDTEMP_{new_f}', text)

text = re.sub(r'Finding 22(?!\d)', 'FINDTEMP_20', text)
text = re.sub(r'Finding 21(?!\d)', 'FINDTEMP_26', text)
text = re.sub(r'Finding 20(?!\d)', 'FINDTEMP_25', text)

# --- Step 2: Replace all placeholders with final values ---
for i in range(50):
    text = text.replace(f'§TEMP_{i}.', f'7.{i}.')
    text = text.replace(f'§TEMP_{i}', f'7.{i}')
    text = text.replace(f'FIGTEMP_{i}', f'Figure {i}')
    text = text.replace(f'TABTEMP_{i}', f'Table {i}')
    text = text.replace(f'FINDTEMP_{i}', f'Finding {i}')

# --- Step 3: Physically move the E2E section ---
# Find the E2E section (now labeled 7.16) and move it after current 7.15 (multi-CSI)
# The E2E section starts with "### 7.16 Extension Experiment: End-to-End"
# and ends with "---" before "## 7.14" (now CSI section)

# Find E2E section boundaries
e2e_header = re.search(r'^### 7\.16 Extension Experiment: End-to-End.*$', text, re.MULTILINE)
if not e2e_header:
    print("ERROR: Could not find E2E section header")
    exit(1)

e2e_start = e2e_header.start()

# Find where the next section starts (## 7.14 which is the CSI section now)
next_section = re.search(r'^---\s*\n\n## 7\.14\b', text[e2e_start:], re.MULTILINE)
if not next_section:
    print("ERROR: Could not find section boundary after E2E")
    exit(1)

e2e_end = e2e_start + next_section.start()
e2e_content = text[e2e_start:e2e_end].rstrip()

# Remove E2E section from its current position
text = text[:e2e_start] + text[e2e_start + next_section.start() + len("---\n\n"):]

# Find where to insert (after "## 7.15" multi-CSI section, before "## 8.")
insert_marker = re.search(r'^---\s*\n\n## 8\. Discussion', text, re.MULTILINE)
if not insert_marker:
    print("ERROR: Could not find Section 8 marker")
    exit(1)

insert_pos = insert_marker.start()
text = text[:insert_pos] + e2e_content + "\n\n---\n\n" + text[insert_pos:]

thesis.write_text(text, encoding="utf-8")
print("Done — moved E2E to 7.16 and renumbered all references")
