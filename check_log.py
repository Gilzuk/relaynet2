import re

with open("xelatex_run3.log", "r", encoding="utf-8", errors="replace") as f:
    log = f.read()

print(f"Log size: {len(log):,} chars")

# Find all error lines
error_lines = [l for l in log.split("\n") if l.startswith("!") or "Error" in l or "fatal" in l.lower()]
print(f"\nError lines ({len(error_lines)}):")
for e in error_lines[:20]:
    print(" ", e[:150])

# Show last 50 lines
lines = log.split("\n")
print(f"\nLast 50 lines (total {len(lines)}):")
for l in lines[-50:]:
    print(l[:150])