import re

with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Current table numbers in restructured
current = set(re.findall(r"\{#tbl:table(\d+)\}", text))
print(f"Current tables: {sorted(int(n) for n in current)}")

# Check for missing tables 15, 16, 17, 23, 24
missing = [15, 16, 17, 23, 24]
print(f"\nSearching for missing tables: {missing}")

for n in missing:
    # Search in backup with very wide range
    m = re.search(r"[Tt]able\s+" + str(n) + r"\b[^\n]*", backup)
    if m:
        start = m.start()
        after = backup[start:start+10000]
        # Find next markdown table
        tbl = re.search(r"\|[^\n]+\|\n\|[-|]+\|(?:\n\|[^\n]+\|)*", after)
        if tbl:
            print(f"\nTable {n}: label='{m.group(0)[:80]}'")
            print(f"  Table found at offset +{tbl.start()}")
            print(f"  First row: {tbl.group(0).split(chr(10))[0][:100]}")
        else:
            print(f"\nTable {n}: label='{m.group(0)[:80]}'")
            print(f"  NO MARKDOWN TABLE found in 10000 chars")
            # Show what's there
            print(f"  Context: {repr(backup[start:start+300])}")
    else:
        print(f"\nTable {n}: NOT FOUND in backup")

# Also check restructured for any markdown tables WITHOUT captions
print("\n\n=== Markdown tables WITHOUT captions in restructured ===")
all_md_tables = list(re.finditer(r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)", text))
print(f"Total markdown tables: {len(all_md_tables)}")
for i, m in enumerate(all_md_tables):
    # Check if followed by a Table: caption
    after = text[m.end():m.end()+200]
    has_caption = bool(re.match(r"\nTable: ", after))
    if not has_caption:
        first_row = m.group(0).split("\n")[0][:80]
        # Get context before
        before = text[max(0,m.start()-100):m.start()].strip().split("\n")[-1][:60]
        print(f"  [{i+1}] No caption | before: '{before}' | row: '{first_row}'")