"""
update_lot.py
-------------
Rebuilds the List of Tables section to include all 32 numbered tables.
"""

import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Extract all table captions in document order
captions_in_order = re.findall(
    r"^Table: (.+) \{#tbl:table(\d+)\}",
    text,
    re.MULTILINE
)

print(f"Found {len(captions_in_order)} table captions")

# Section map for all tables
tbl_section_map = {
    1: "§4.2", 2: "§4.2", 3: "§4.2",
    4: "§4.3", 5: "§4.3", 6: "§4.3",
    7: "§4.4", 8: "§4.4", 9: "§4.4",
    10: "§4.4", 11: "§4.4", 12: "§4.4", 13: "§4.4",
    14: "§4.5", 15: "§4.5", 16: "§4.6",
    17: "§1.7", 18: "§4.6", 19: "§4.6",
    20: "§4.6", 21: "§4.6", 22: "§4.6",
    23: "§4.8", 24: "§4.7", 25: "App. B",
    26: "§1.8", 27: "§3.1", 28: "§3.4",
    29: "§3.5", 30: "§3.5", 31: "§3.2",
    32: "§5.4", 33: "§5.1",
}

# Build new LoT
lot_lines = [
    "## List of Tables {.unnumbered}\n",
    "| Table | Description | Section |",
    "|-------|-------------|---------|",
]

# Sort by table number
sorted_captions = sorted(captions_in_order, key=lambda x: int(x[1]))

for cap, num in sorted_captions:
    n = int(num)
    # Truncate caption
    short = cap[:85].rstrip(",;")
    if len(cap) > 85:
        short += "..."
    sec = tbl_section_map.get(n, "§4")
    lot_lines.append(f"| {n} | {short} | {sec} |")

new_lot = "\n".join(lot_lines) + "\n"

# Replace the existing List of Tables section
text = re.sub(
    r"## List of Tables \{\.unnumbered\}.*?(?=\n## )",
    new_lot,
    text,
    flags=re.DOTALL
)

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Output size: {len(text):,} chars")
print("Saved.")

# Verify
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

lot_m = re.search(r"## List of Tables.*?(?=\n## )", verify, re.DOTALL)
if lot_m:
    rows = [l for l in lot_m.group(0).split("\n") if l.startswith("| ") and "---|" not in l and "Table" not in l[:8]]
    print(f"\nList of Tables: {len(rows)} entries")
    for r in rows:
        print(f"  {r[:110]}")