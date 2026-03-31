import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
with open("thesis.md", encoding="utf-8") as f:
    content = f.read()
blocks = list(re.finditer(r"\$\$([\s\S]+?)\$\$", content))
print("Total: %d" % len(blocks))
for i, m in enumerate(blocks):
    start = m.start()
    pre = content[max(0,start-350):start].strip().replace("\n"," ")
    eq  = m.group(1).strip().replace("\n"," ")[:140]
    print("EQ%03d|%s" % (i+1, eq))
    print("  CTX|%s" % pre[-200:])
