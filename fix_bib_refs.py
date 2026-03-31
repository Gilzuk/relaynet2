import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Find all lines in References section with [@eq:]
refs_m = re.search(r"## References(.*?)(?=\n## )", text, re.DOTALL)
if refs_m:
    refs_text = refs_m.group(1)
    for line in refs_text.split("\n"):
        if "[@eq:" in line:
            print(repr(line[:250]))

print("\n--- Fixing ---")

# Remove all [@eq:...] from the References section
def clean_refs_section(m):
    return re.sub(r"\s*\(\[@eq:[^\]]+\]\)", "", m.group(0))

text_new = re.sub(
    r"## References.*?(?=\n## )",
    clean_refs_section,
    text,
    flags=re.DOTALL
)

# Also clean *Source: ...* lines that appear inside the References section
# (equation citations should not appear in the bibliography)
def clean_source_in_refs(m):
    return re.sub(r"\n\*Source:[^\n]+\*", "", m.group(0))

text_new = re.sub(
    r"## References.*?(?=\n## )",
    clean_source_in_refs,
    text_new,
    flags=re.DOTALL
)

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text_new)

print("Saved.")

# Verify
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

refs_m2 = re.search(r"## References(.*?)(?=\n## )", verify, re.DOTALL)
remaining = 0
if refs_m2:
    for line in refs_m2.group(1).split("\n"):
        if "[@eq:" in line:
            remaining += 1
            print(f"  Still has ref: {repr(line[:150])}")

print(f"Remaining [@eq:] in bibliography: {remaining}")
print(f"Document size: {len(verify):,} chars")