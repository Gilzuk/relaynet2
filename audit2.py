import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

# 1. Find cross-refs incorrectly inserted into figure captions
print("=== Cross-refs in figure captions (BUGS) ===")
cap_with_ref = re.findall(r"\*Figure \d+[^\*]*\[@eq:[^\*]*\*", text)
for c in cap_with_ref:
    print(f"  {c[:150]}")

# 2. Find cross-refs incorrectly inserted into references
print("\n=== Cross-refs in bibliography entries (BUGS) ===")
refs_m = re.search(r"## References(.*?)(?=\n## |\Z)", text, re.DOTALL)
if refs_m:
    refs_text = refs_m.group(1)
    bad_refs = re.findall(r"\[\d+\][^\n]*\[@eq:[^\n]*", refs_text)
    for r in bad_refs:
        print(f"  {r[:150]}")

# 3. Full Hebrew abstract from backup
print("\n=== Full Hebrew Abstract ===")
heb_m = re.search(r'<div dir="rtl">(.*?)</div>', backup, re.DOTALL)
if heb_m:
    print(heb_m.group(0)[:2000])
else:
    # Try alternative
    heb_m2 = re.search(r"## Abstract \(Hebrew\)(.*?)(?=\n## )", backup, re.DOTALL)
    if heb_m2:
        print(heb_m2.group(1)[:2000])
    else:
        print("Not found with div, trying other patterns...")
        idx = backup.find("תקציר")
        if idx >= 0:
            print(repr(backup[max(0,idx-100):idx+2000]))

# 4. All figure captions (for List of Figures)
print("\n=== All Figure Captions ===")
captions = re.findall(r"\*Figure (\d+): ([^\*]+)\*", text)
for num, cap in captions[:60]:
    print(f"  Fig {num}: {cap[:100]}")
print(f"  Total: {len(captions)}")

# 5. All table labels (for List of Tables)
print("\n=== All Table Labels ===")
tables = re.findall(r"Table (\d+): ([^\n]+)", text)
for num, label in tables[:30]:
    print(f"  Table {num}: {label[:100]}")
print(f"  Total: {len(tables)}")