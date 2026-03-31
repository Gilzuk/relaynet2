import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("thesis.md", encoding="utf-8") as f:
    content = f.read()

blocks = list(re.finditer(r'\$\$([\s\S]+?)\$\$', content))
print(f"Total display equations: {len(blocks)}")
print()
for i, m in enumerate(blocks):
    start = m.start()
    pre = content[max(0, start-400):start].strip().replace('\n', ' ')
    eq  = m.group(1).strip().replace('\n', ' ')[:150]
    print(f"EQ {i+1:3d} | {eq}")
    print(f"       CTX: ...{pre[-220:]}")
    print()
