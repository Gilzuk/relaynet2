import re
with open('thesis_restructured.md', 'r', encoding='utf-8') as f:
    text = f.read()
headings = re.findall(r'^## .+', text, re.MULTILINE)
for h in headings:
    print(repr(h))
print(f"\nTotal ## headings: {len(headings)}")
print(f"\nFile size: {len(text)} chars")
# Also check for Experiments
if 'Experiments' in text:
    idx = text.index('Experiments')
    print(f"\n'Experiments' found at index {idx}")
    print(f"Context: {repr(text[max(0,idx-5):idx+50])}")