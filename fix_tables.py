"""
fix_tables.py
-------------
For every markdown table in thesis_restructured.md:
  1. Ensures it has a Table N label (enumerate any unlabeled tables)
  2. Moves the caption BELOW the table (pandoc-crossref style: Table: caption {#tbl:N})
  3. Adds {#tbl:N} anchor for cross-referencing

Current format (before):
  Table N: Description text
  
  | col | col |
  |-----|-----|
  | ... | ... |

Target format (after):
  | col | col |
  |-----|-----|
  | ... | ... |
  
  Table: Description text {#tbl:tableN}
"""

import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find all table blocks
# A table block = optional label line + blank line + markdown table rows
# ─────────────────────────────────────────────────────────────────────────────

# Pattern: optional label, then markdown table
# Label formats seen in the document:
#   "Table N: description\n\n| ...\n|---|\n..."
#   "*Table N: description*\n\n| ...\n|---|\n..."
#   "Table N: description\n\nprose...\n\n| ...\n|---|\n..."  (label far from table)

# We'll process in two passes:
# Pass 1: Find all "Table N: ..." labels and their associated markdown tables
# Pass 2: Reformat each block

# Find all table labels with their positions
label_pattern = re.compile(
    r"(\*{0,2})(Table (\d+): [^\n]+)(\*{0,2})",
    re.MULTILINE
)

# Find all markdown table blocks (header row + separator row + data rows)
table_block_pattern = re.compile(
    r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)",
    re.MULTILINE
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build a list of (label_pos, label_text, table_num, table_pos, table_text)
# ─────────────────────────────────────────────────────────────────────────────

# Find all markdown tables
md_tables = list(table_block_pattern.finditer(text))
print(f"  Found {len(md_tables)} markdown table blocks")

# Find all Table N labels
labels = list(label_pattern.finditer(text))
print(f"  Found {len(labels)} Table N labels")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Associate each label with the nearest following markdown table
# ─────────────────────────────────────────────────────────────────────────────

# Build association: label → table
associations = []
used_tables = set()

for lbl in labels:
    lbl_pos = lbl.start()
    lbl_text = lbl.group(2)  # "Table N: description"
    tbl_num = int(lbl.group(3))

    # Find the nearest markdown table after this label (within 8000 chars)
    best_tbl = None
    best_dist = 8000
    for tbl in md_tables:
        if tbl.start() > lbl_pos and tbl.start() - lbl_pos < best_dist:
            if id(tbl) not in used_tables:
                best_tbl = tbl
                best_dist = tbl.start() - lbl_pos

    if best_tbl:
        used_tables.add(id(best_tbl))
        associations.append((lbl, lbl_text, tbl_num, best_tbl))
        print(f"  Table {tbl_num}: label at {lbl_pos}, table at {best_tbl.start()} (dist={best_dist})")
    else:
        print(f"  Table {tbl_num}: label at {lbl_pos}, NO TABLE FOUND")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Rebuild the document with reformatted table blocks
# Process in reverse order to preserve positions
# ─────────────────────────────────────────────────────────────────────────────

# Sort by table position (descending) for reverse processing
associations.sort(key=lambda x: x[3].start(), reverse=True)

new_text = text

for lbl, lbl_text, tbl_num, tbl_m in associations:
    # Extract the description (remove "Table N: " prefix and any asterisks)
    desc = re.sub(r"^\*{0,2}Table \d+:\s*", "", lbl_text).strip().rstrip("*").strip()

    # The pandoc-crossref caption format (below the table):
    # Table: Description {#tbl:tableN}
    caption_line = f"\nTable: {desc} {{#tbl:table{tbl_num}}}\n"

    # The table content (ensure it ends with newline)
    tbl_content = tbl_m.group(0)
    if not tbl_content.endswith("\n"):
        tbl_content += "\n"

    # New block: table + caption below
    new_block = tbl_content + caption_line

    # Replace the table in the text
    new_text = new_text[:tbl_m.start()] + new_block + new_text[tbl_m.end():]

# Now remove the old label lines (they've been moved below as captions)
# Remove "Table N: ..." lines (with optional asterisks) that precede tables
# These are now redundant since the caption is below
for lbl, lbl_text, tbl_num, tbl_m in associations:
    # Remove the label line from the text
    # The label might be: "Table N: ...\n" or "*Table N: ...*\n"
    escaped = re.escape(lbl.group(0))
    new_text = re.sub(r"\n?" + escaped + r"\n?", "\n", new_text, count=1)

print(f"\nOutput size: {len(new_text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Verify
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

captions_below = re.findall(r"^Table: .+ \{#tbl:", verify, re.MULTILINE)
print(f"\nTable captions below (pandoc-crossref): {len(captions_below)}")
for c in captions_below:
    print(f"  {c[:100]}")

old_labels = re.findall(r"^Table \d+: ", verify, re.MULTILINE)
print(f"\nOld 'Table N:' labels remaining: {len(old_labels)}")