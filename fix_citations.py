import glob, re

# Mapping from citation number to BibTeX key
cite_map = {
    1: 'laneman2004cooperative',
    2: 'nosratinia2004cooperative',
    3: 'cover1979capacity',
    4: 'elgamal2011network',
    5: 'nazer2011compute',
    6: 'rankov2007spectral',
    7: 'ye2018power',
    8: 'samuel2019learning',
    9: 'dorner2018deep',
    10: 'sun2018learning',
    11: 'kingma2014auto',
    12: 'mirza2014conditional',
    13: 'gulrajani2017improved',
    14: 'goodfellow2014generative',
    15: 'vaswani2017attention',
    16: 'gu2024mamba',
    17: 'gu2022efficiently',
    18: 'dao2024transformers',
    19: 'tse2005fundamentals',
    20: 'wolniansky1998vblast',
    21: 'proakis2008digital',
    22: 'simon2005digital',
    23: 'goodfellow2016deep',
    24: 'sklar2001digital',
    25: 'sutton2018reinforcement',
    26: 'telatar1999capacity',
    27: 'foschini1996layered',
    28: 'zheng2003diversity',
    29: 'tse1999linear',
    30: 'loyka2004performance',
    31: 'higgins2017beta',
    32: 'he2015delving',
}

def replace_citations(text):
    # Replace {[}N, M, ...{]} multi-citations
    def replace_multi(m):
        nums = re.findall(r'\d+', m.group(0))
        keys = [cite_map[int(n)] for n in nums if int(n) in cite_map]
        if keys:
            return r'\cite{' + ', '.join(keys) + '}'
        return m.group(0)
    
    # Replace {[}N{]} single citations
    def replace_single(m):
        n = int(m.group(1))
        if n in cite_map:
            return r'\cite{' + cite_map[n] + '}'
        return m.group(0)
    
    # First handle multi-citations like {[}7, 8{]} or {[}7{]}{[}8{]}
    text = re.sub(r'\{\\?\[?\}?\{?\\?\[?\}?\s*\{?\[?\}?(\d+(?:,\s*\d+)+)\{?\]?\}?\{?\\?\]?\}?', replace_multi, text)
    
    # Then handle single citations {[}N{]}
    text = re.sub(r'\{\\?\[?\}?\s*(\d+)\s*\{\\?\]?\}?', replace_single, text)
    
    return text

# Process all chapter files
for f in sorted(glob.glob('chapters/*.tex')):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace {[}N{]} patterns (Pandoc-generated inline citations)
    # Pattern: {[}N{]} or {[}N, M{]} (multi-citation)
    def replace_cite(m):
        nums_str = m.group(1)
        nums = [int(n.strip()) for n in nums_str.split(',')]
        keys = [cite_map[n] for n in nums if n in cite_map]
        if keys:
            return r'\cite{' + ', '.join(keys) + '}'
        return m.group(0)
    
    # Match {[}N{]} or {[}N, M, ...{]} (exact Pandoc pattern)
    new_content = re.sub(r'\{\\?\[\\?\}([\d,\s]+)\{\\?\]\\?\}', replace_cite, content)
    
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f'Updated citations in {f}')

print('Done fixing citations.')
