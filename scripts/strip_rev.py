r"""Brace-matching stripper for \REV{...} annotations and revblock environments.

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
        # A \REV{} that sat on a line of its own leaves an empty line behind,
        # which should go with it. A \REV{} sitting *inline* inside a sentence
        # must keep the space that follows it: swallowing that space ran the
        # neighbouring sentences together in the clean bundle
        # ("...(Chapter 6).These additional...") while the annotated copy, where
        # \REV expands to nothing, spaced them correctly. Only collapse the
        # whitespace when the annotation occupied the whole line.
        prefix = ''.join(out)
        alone_at_start = prefix[prefix.rfind('\n') + 1:].strip() == ''
        eol = k
        while eol < len(text) and text[eol] in ' \t':
            eol += 1
        if alone_at_start and eol < len(text) and text[eol] == '\n':
            k = eol + 1
        i = k
    s = ''.join(out)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s
