import re, os

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Check figure captions
fig_blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL)
figs_with_cap = sum(1 for b in fig_blocks if "\\caption" in b)
print(f"Figures: {len(fig_blocks)} total, {figs_with_cap} with caption")
if fig_blocks:
    print("Sample figure block:")
    print(fig_blocks[0][:400])
    print()

# Check longtable captions
lt_blocks = re.findall(r"\\begin\{longtable\}.*?\\end\{longtable\}", tex, re.DOTALL)
lt_with_cap = sum(1 for b in lt_blocks if "\\caption" in b)
print(f"Longtables: {len(lt_blocks)} total, {lt_with_cap} with caption")
for b in lt_blocks[:3]:
    cap = re.search(r"\\caption\{[^\}]+\}", b)
    if cap:
        print(f"  Caption: {cap.group(0)[:80]}")

# Check listoffigures/listoftables
print(f"\nlistoffigures: {tex.count(chr(92)+'listoffigures')}")
print(f"listoftables: {tex.count(chr(92)+'listoftables')}")

# Check bibliography
print(f"thebibliography env: {tex.count('thebibliography')}")
print(f"bibliography cmd: {tex.count(chr(92)+'bibliography{')}")

# References chapter preview
ref_idx = tex.find("\\chapter{References}")
if ref_idx < 0:
    ref_idx = tex.find("\\chapter*{References}")
if ref_idx >= 0:
    print(f"\nReferences section preview:")
    print(tex[ref_idx:ref_idx+400])

# LOF file
for fname in ["thesis_tau_new.lof", "thesis_tau.lof"]:
    if os.path.exists(fname):
        with open(fname, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        print(f"\n{fname}: {len(content)} chars")
        print(content[:600])
        break

# LOT file
for fname in ["thesis_tau_new.lot", "thesis_tau.lot"]:
    if os.path.exists(fname):
        with open(fname, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        print(f"\n{fname}: {len(content)} chars")
        print(content[:600])
        break