"""Fix ch04_methods.tex Source annotations using simple string replacement."""

with open('chapters/ch04_methods.tex', 'r', encoding='utf-8') as f:
    text = f.read()

original = text

# The actual format in the file is:
# \par{\small\textit{Source: [21, Ch. 4, Eq. 4-3-30]}}
# (2 closing braces, not 3)

replacements = [
    # [21, ...] -> \cite{proakis2008digital}
    (r'\par{\small\textit{Source: [21, Ch. 14]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 4, Eq. 4-3-30]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 4, Eq. 4-3-31]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 5, Eq. 5-2-60]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 5, Eq. 5-2-79]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 5, Eq. 5-2-1]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 5]}}', r'\cite{proakis2008digital}'),
    (r'\par{\small\textit{Source: [21, Ch. 5, Eq. 5-2-22]}}', r'\cite{proakis2008digital}'),
    # [19, ...] -> \cite{tse2005fundamentals}
    (r'\par{\small\textit{Source: [19, Ch. 7.2]}}', r'\cite{tse2005fundamentals}'),
    # [23, ...] -> \cite{goodfellow2016deep}
    (r'\par{\small\textit{Source: [23, Ch. 6]}}', r'\cite{goodfellow2016deep}'),
    # [1, Eq. 1] -> remove (self-reference)
    (r'\par{\small\textit{Source: [1, Eq. 1]}}', ''),
]

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f'Replaced: {old[:60]}')
    else:
        print(f'NOT FOUND: {old[:60]}')

with open('chapters/ch04_methods.tex', 'w', encoding='utf-8') as f:
    f.write(text)

# Count remaining
remaining = text.count('Source:')
cites = text.count(r'\cite{')
print(f'\nRemaining Source: annotations: {remaining}')
print(f'Total \\cite{{}} in ch04: {cites}')
print('Done.')