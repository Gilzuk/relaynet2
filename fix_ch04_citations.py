"""
Fix ch04_methods.tex:
1. Replace \par{\small\textit{Source: [21, ...]}} with \cite{proakis2008digital}
2. Replace \par{\small\textit{Source: [19, ...]}} with \cite{tse2005fundamentals}
3. Replace \par{\small\textit{Source: [23, ...]}} with \cite{goodfellow2016deep}
4. Remove \par{\small\textit{Source: Standard statistics}} annotations
5. Remove \par{\small\textit{Source: [1, Eq. 1]}} annotation
6. Add sklar2001digital citation where appropriate
"""
import re

with open('chapters/ch04_methods.tex', 'r', encoding='utf-8') as f:
    text = f.read()

original = text

# Map numbered references to bib keys
# [21] = proakis2008digital (Proakis & Salehi "Digital Communications")
# [19] = tse2005fundamentals (Tse & Viswanath "Fundamentals of Wireless Communication")
# [23] = goodfellow2016deep (Goodfellow et al. "Deep Learning")
# [1, Eq. 1] = self-reference or proakis, remove
# Standard statistics = remove

replacements = [
    # [21, ...] -> \cite{proakis2008digital}
    (r'\\par\{\\small\\textit\{Source: \[21[^\}]*\}\}\}', r'\\cite{proakis2008digital}'),
    # [19, ...] -> \cite{tse2005fundamentals}
    (r'\\par\{\\small\\textit\{Source: \[19[^\}]*\}\}\}', r'\\cite{tse2005fundamentals}'),
    # [23, ...] -> \cite{goodfellow2016deep}
    (r'\\par\{\\small\\textit\{Source: \[23[^\}]*\}\}\}', r'\\cite{goodfellow2016deep}'),
    # Standard statistics -> remove
    (r'\\par\{\\small\\textit\{Source: Standard statistics\}\}', ''),
    # [1, Eq. 1] -> remove (self-reference)
    (r'\\par\{\\small\\textit\{Source: \[1[^\}]*\}\}\}', ''),
]

for pattern, replacement in replacements:
    new_text = re.sub(pattern, replacement, text)
    count = len(re.findall(pattern, text))
    if count > 0:
        print(f'Replaced {count} occurrences of pattern: {pattern[:50]}...')
    text = new_text

# Also add sklar2001digital citation in the modulation section
# Add it to the BPSK section where modulation is first introduced
# Find the BPSK section and add a citation
bpsk_pattern = r'(Binary Phase-Shift Keying maps a single bit to a real-valued symbol:)'
bpsk_replacement = r'\1~\\cite{sklar2001digital}'
new_text = re.sub(bpsk_pattern, bpsk_replacement, text)
if new_text != text:
    print('Added sklar2001digital citation to BPSK section')
    text = new_text

with open('chapters/ch04_methods.tex', 'w', encoding='utf-8') as f:
    f.write(text)

# Count changes
import re as re2
orig_sources = len(re2.findall(r'\\par\{\\small\\textit\{Source:', original))
new_sources = len(re2.findall(r'\\par\{\\small\\textit\{Source:', text))
new_cites = len(re2.findall(r'\\cite\{', text))
print(f'\nOriginal Source annotations: {orig_sources}')
print(f'Remaining Source annotations: {new_sources}')
print(f'Total \\cite{{}} in ch04: {new_cites}')
print('Done.')