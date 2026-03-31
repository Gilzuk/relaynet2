import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Show sample equation blocks
print("=== Sample equation blocks ===")
samples = list(re.finditer(r"\$\$(.+?)\$\$ \{#eq:([^\}]+)\}", text, re.DOTALL))
print(f"Total labeled equations found: {len(samples)}")
for idx in [0, 6, 36, 60, 73]:
    if idx < len(samples):
        m = samples[idx]
        inner = m.group(1).strip().replace("\n", " ")[:80]
        label = m.group(2)
        print(f"  ({label}): {inner}...")

print()
print("=== Sample cross-references in prose ===")
crossrefs = list(re.finditer(r".{0,60}\(\[@eq:[^\]]+\]\).{0,30}", text))
for m in crossrefs[:8]:
    print(" ", repr(m.group(0).replace("\n", " ")))

print()
print("=== Equation Reference table (first 7 rows) ===")
tbl_m = re.search(r"## Equation Reference.*?(\|.+?)(?=\n##)", text, re.DOTALL)
if tbl_m:
    rows = tbl_m.group(1).strip().split("\n")
    for r in rows[:7]:
        print(" ", r)

print()
total_crossrefs = len(re.findall(r"\[@eq:", text))
total_tags = len(re.findall(r"\\tag\{", text))
total_labels = len(re.findall(r"\{#eq:", text))
print(f"Summary: {total_tags} tag{{N}}, {total_labels} {{#eq:}} labels, {total_crossrefs} [@eq:] refs")
print(f"Document size: {len(text):,} chars")