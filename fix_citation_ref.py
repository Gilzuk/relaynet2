import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

# Fix the malformed citation reference:
# "[21, Eq. [@eq: ([@eq:rayleigh-ber])vae-elbo]-4-15]"
# → "[21, Eq. 14-4-15]"
# This was a book citation (Proakis, Eq. 14-4-15), not a thesis equation ref.

text = text.replace(
    "[21, Eq. [@eq: ([@eq:rayleigh-ber])vae-elbo]-4-15]",
    "[21, Eq. 14-4-15]"
)

# Also add the correct cross-reference after the phrase
text = text.replace(
    "yields the closed-form [21, Eq. 14-4-15]:",
    "yields the closed-form [21, Eq. 14-4-15] ([@eq:rayleigh-ber]):"
)

with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed.")

# Verify
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

malformed = re.findall(r"\[@eq:[^\]]*\[@eq:", verify)
print(f"Malformed nested refs remaining: {len(malformed)}")
total_crossrefs = len(re.findall(r"\[@eq:", verify))
print(f"Total [@eq:] cross-references: {total_crossrefs}")
print(f"Document size: {len(verify):,} chars")