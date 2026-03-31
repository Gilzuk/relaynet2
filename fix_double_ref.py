import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Fix the double cross-reference:
# "([@eq: ([@eq:power-normalization])af-gain])" → "([@eq:af-gain])"
# This happened because the power-normalization rule fired first, then af-gain
# wrapped around it.

before = len(re.findall(r"\[@eq:", text))

# Pattern: nested/malformed cross-refs
# Fix: "([@eq: ([@eq:X])Y])" → "([@eq:Y])"
text = re.sub(
    r"\(\[@eq:\s*\(\[@eq:[^\]]+\]\)[^\]]*\]\)",
    lambda m: "([@eq:af-gain])",
    text
)

# Also fix any other double-refs: "([@eq:X]) ([@eq:X])" → "([@eq:X])"
text = re.sub(
    r"(\(\[@eq:[^\]]+\]\))\s*\1",
    r"\1",
    text
)

# Fix: "([@eq:X])([@eq:Y])" → "([@eq:X]) ([@eq:Y])"  (add space)
text = re.sub(
    r"(\(\[@eq:[^\]]+\]\))(\(\[@eq:)",
    r"\1 \2",
    text
)

after = len(re.findall(r"\[@eq:", text))
print(f"Cross-refs before: {before}, after: {after}")

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved.")

# Verify no malformed refs remain
malformed = re.findall(r"\[@eq:[^\]]*\[@eq:", text)
print(f"Malformed nested refs remaining: {len(malformed)}")
if malformed:
    for m in malformed:
        print(f"  {repr(m)}")