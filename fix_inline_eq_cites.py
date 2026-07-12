"""
Fix inline [N, ...] citations in \par{\small\textit{Source: ...}} annotations.
Converts them to proper \cite{key} format and removes duplicates.
"""
import re

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

def convert_inline_source(m):
    """Convert \par{\small\textit{Source: [N, ...], [M, ...]}} to \cite{key1, key2}"""
    full = m.group(0)
    inner = m.group(1)  # content inside Source: ...
    
    # Extract all reference numbers
    nums = re.findall(r'\[(\d+)', inner)
    if not nums:
        return full  # no numbers to convert
    
    keys = [cite_map[int(n)] for n in nums if int(n) in cite_map]
    if not keys:
        return full
    
    return r'\par{\small\textit{Source: \cite{' + ', '.join(keys) + '}}}'

for filename in ['chapters/ch01_introduction.tex', 'chapters/ch02_literature_review.tex']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Step 1: Remove the duplicate \cite{} annotation I added (where old inline already exists)
    # Pattern: \par{\small\textit{Source: \cite{...}}} \par{\small\textit{Source: [N, ...]}}
    content = re.sub(
        r'\\par\{\\small\\textit\{Source: \\cite\{[^}]+\}\}\}\s*'
        r'(\\par\{\\small\\textit\{Source: \[)',
        r'\1',
        content
    )
    
    # Step 2: Convert remaining inline [N, ...] citations to \cite{key}
    content = re.sub(
        r'\\par\{\\small\\textit\{Source: ([^}]+)\}\}',
        convert_inline_source,
        content
    )
    
    if content != original:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {filename}')

print('Done fixing inline equation citations.')