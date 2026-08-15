"""Brace-matching stripper for \REV{...} annotations and revblock environments.

Handles multi-line annotations, which a line-based regex cannot: it scans for
\REV{ and walks the braces (respecting \{ \} escapes) to find the true end.
"""
import re

def strip_rev(text):
    text = re.sub(r'\\begin\{revblock\}.*?\\end\{revblock\}\n?', '', text, flags=re.S)
    out = []
    i = 0
    while True:
        j = text.find('\\REV{', i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len('\\REV{')
        depth = 1
        while k < len(text) and depth:
            c = text[k]
            if c == '\\':          # skip escaped char
                k += 2; continue
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            k += 1
        # swallow a trailing blank line left behind
        while k < len(text) and text[k] in ' \t':
            k += 1
        if k < len(text) and text[k] == '\n':
            k += 1
        i = k
    s = ''.join(out)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s
