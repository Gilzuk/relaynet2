import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Remove incorrect 5.8-5.11 numbers from Appendix sub-sections
text = re.sub(
    r"^(### )5\.\d+ (Appendix [A-Z]:)",
    r"\1\2",
    text,
    flags=re.MULTILINE
)

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed Appendix sub-section numbers.")

# Verify
for line in text.split("\n"):
    if "Appendix" in line and line.startswith("###"):
        print(f"  {line}")