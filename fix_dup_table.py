import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# There are two Table 13 entries.
# The second one is "Context-length benchmark" in the Appendix.
# Renumber it as Table 25.

# Find both occurrences of {#tbl:table13}
positions = [m.start() for m in re.finditer(r"\{#tbl:table13\}", text)]
print(f"Found {len(positions)} occurrences of {{#tbl:table13}} at positions: {positions}")

if len(positions) == 2:
    # Replace only the SECOND occurrence
    second_pos = positions[1]
    # Replace the caption line containing the second table13
    # Find the full caption line
    line_start = text.rfind("\n", 0, second_pos) + 1
    line_end = text.find("\n", second_pos)
    old_line = text[line_start:line_end]
    new_line = old_line.replace("{#tbl:table13}", "{#tbl:table25}")
    text = text[:line_start] + new_line + text[line_end:]
    print(f"  Renamed second Table 13 → Table 25")
    print(f"  Caption: {new_line[:120]}")

# Also update the List of Tables if it references Table 13 twice
# The LoT should show Table 25 for the context-length benchmark
# Find the LoT section and update
lot_m = re.search(r"## List of Tables.*?(?=\n## )", text, re.DOTALL)
if lot_m:
    lot_text = lot_m.group(0)
    # Check if Table 13 appears twice in LoT
    t13_in_lot = lot_text.count("| 13 |")
    print(f"\nTable 13 entries in List of Tables: {t13_in_lot}")

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print("\nSaved.")

# Final verification
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

tbl_nums = re.findall(r"\{#tbl:table(\d+)\}", verify)
from collections import Counter
dupes = {k: v for k, v in Counter(tbl_nums).items() if v > 1}
print(f"Duplicate table numbers: {dupes}")
print(f"All table numbers: {sorted(set(int(n) for n in tbl_nums))}")
print(f"Document size: {len(verify):,} chars")