import re

with open("thesis_preprocessed.md", "r", encoding="utf-8") as f:
    text = f.read()

# Find and fix duplicate {#fig:figN} labels
# Keep first occurrence, remove label from subsequent occurrences
from collections import Counter

figs = re.findall(r"\{#fig:fig(\d+)\}", text)
dupes = {k for k, v in Counter(figs).items() if v > 1}
print(f"Duplicate fig labels: {sorted(int(d) for d in dupes)}")

seen = set()
def fix_dup_label(m):
    label = m.group(1)
    if label not in seen:
        seen.add(label)
        return m.group(0)  # keep first occurrence
    else:
        # Remove the label from duplicate (keep image without anchor)
        return ""  # remove {#fig:figN} from duplicate

text = re.sub(r"\{#fig:fig(\d+)\}", fix_dup_label, text)

# Verify
figs_after = re.findall(r"\{#fig:fig(\d+)\}", text)
dupes_after = {k: v for k, v in Counter(figs_after).items() if v > 1}
print(f"Duplicates after fix: {dupes_after}")
print(f"Total fig labels: {len(figs_after)}")

with open("thesis_preprocessed.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved thesis_preprocessed.md")