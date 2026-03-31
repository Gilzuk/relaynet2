import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Find all display equations ($$...$$)
display_eqs = list(re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL))
print(f"Display equations ($$...$$): {len(display_eqs)}")
for i, m in enumerate(display_eqs):
    start = m.start()
    before_lines = text[max(0, start - 200):start].strip().split("\n")
    before = before_lines[-1][-60:] if before_lines else ""
    eq_content = m.group(1).strip().replace("\n", " ")[:100]
    print(f"  [{i+1:02d}] context: ...{before}")
    print(f"        eq: {eq_content}")
    print()