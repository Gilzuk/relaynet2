import re

with open("xelatex_run9.log", "r", encoding="utf-8", errors="replace") as f:
    log = f.read()

print(f"Log: {len(log):,} chars")

# Find all error blocks with line numbers
lines = log.split("\n")
for i, line in enumerate(lines):
    if line.startswith("!"):
        # Print error + next 5 lines for context
        block = lines[i:i+6]
        print("\n".join(block[:6]))
        print("---")