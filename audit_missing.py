import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

# 1. Check for Hebrew abstract
print("=== Hebrew Abstract ===")
heb_m = re.search(r"(תקציר|Abstract.*?Hebrew|Hebrew.*?Abstract|## Abstract.*?Hebrew)", backup, re.IGNORECASE | re.DOTALL)
if heb_m:
    start = heb_m.start()
    print(repr(backup[start:start+500]))
else:
    print("Not found in backup")

# Check in restructured
heb_r = re.search(r"(תקציר|Hebrew)", text, re.IGNORECASE)
print(f"\nHebrew in restructured: {bool(heb_r)}")

# 2. Check figure captions
print("\n=== Figure Captions (first 10) ===")
captions = re.findall(r"\*Figure \d+[^\*]+\*", text)
for c in captions[:10]:
    print(f"  {c[:120]}")
print(f"  Total captions: {len(captions)}")

# 3. Check references section
print("\n=== References (first 10) ===")
refs_m = re.search(r"## References(.*?)(?=\n## |\Z)", text, re.DOTALL)
if refs_m:
    refs_text = refs_m.group(1).strip()
    refs_lines = [l for l in refs_text.split("\n") if l.strip()]
    for l in refs_lines[:15]:
        print(f"  {l[:120]}")
    print(f"  Total ref lines: {len(refs_lines)}")

# 4. Check for List of Figures / List of Tables
print("\n=== List of Figures/Tables in restructured ===")
print(f"  'List of Figures': {'List of Figures' in text}")
print(f"  'List of Tables': {'List of Tables' in text}")

# 5. Check backup for Hebrew abstract
print("\n=== Hebrew abstract in backup ===")
heb_sections = re.findall(r"(?:תקציר|## Abstract.*?Hebrew|Hebrew Abstract)[^\n]*\n(.{0,1000})", backup, re.IGNORECASE | re.DOTALL)
for s in heb_sections[:2]:
    print(repr(s[:400]))