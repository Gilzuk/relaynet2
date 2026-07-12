import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Check heading structure in Literature Review
lit_m = re.search(r"## Introduction and Literature Review(.*?)## Research Objectives", text, re.DOTALL)
lit_text = lit_m.group(1) if lit_m else ""
lit_headings = re.findall(r"^###.+", lit_text, re.MULTILINE)
print("Literature Review sub-headings:")
for h in lit_headings:
    print(" ", h)

# Check Methods headings
meth_m = re.search(r"## Methods(.*?)## Experiments", text, re.DOTALL)
meth_text = meth_m.group(1) if meth_m else ""
meth_headings = re.findall(r"^###.+", meth_text, re.MULTILINE)
print("\nMethods sub-headings:")
for h in meth_headings:
    print(" ", h)

# Check Experiments headings
exp_m = re.search(r"## Experiments(.*?)## Discussion", text, re.DOTALL)
exp_text = exp_m.group(1) if exp_m else ""
exp_headings = re.findall(r"^###.+", exp_text, re.MULTILINE)
print("\nExperiments sub-headings:")
for h in exp_headings:
    print(" ", h)

# Final stats
total_figs = len(re.findall(r"!\[Figure", text))
total_tbls = len(re.findall(r"\|[-|]+\|", text))
print(f"\nDocument stats:")
print(f"  Total chars: {len(text):,}")
print(f"  Total figures: {total_figs}")
print(f"  Total table separators: {total_tbls}")

# Check no duplicate headings in Lit Review
dup_ch = text.count("### Channel Models")
dup_mimo = text.count("### MIMO Equalization Techniques")
print(f"\n  '### Channel Models' occurrences: {dup_ch} (should be 0)")
print(f"  '### MIMO Equalization Techniques' occurrences: {dup_mimo} (should be 0)")
print(f"  '### Theoretical Foundations: Channel Models' occurrences: {text.count('### Theoretical Foundations: Channel Models')}")
print(f"  '### Theoretical Foundations: MIMO Equalization' occurrences: {text.count('### Theoretical Foundations: MIMO Equalization')}")