import re

with open('chapters/ch01_introduction.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all {[}N{]} patterns
matches = re.findall(r'\{\\?\[\\?\}[\d,\s]+\{\\?\]\\?\}', text)
print(f'Pattern 1 matches: {len(matches)}')
for m in matches[:5]:
    print(repr(m))

# Try another pattern
matches2 = re.findall(r'\{\\?\[\\?\}\d+\{\\?\]\\?\}', text)
print(f'\nPattern 2 matches: {len(matches2)}')
for m in matches2[:5]:
    print(repr(m))

# Search for any [N] style
matches3 = re.findall(r'\[(\d+)\]', text)
print(f'\nPattern 3 [N] matches: {len(matches3)}')
for m in matches3[:5]:
    print(repr(m))

# Search for the exact Pandoc pattern
matches4 = re.findall(r'\{\\?\[\\?\}(\d+)\{\\?\]\\?\}', text)
print(f'\nPattern 4 exact Pandoc: {len(matches4)}')
for m in matches4[:5]:
    print(repr(m))

# Show a snippet of the text around a citation
idx = text.find('[7]')
if idx != -1:
    print(f'\nContext around [7]: {repr(text[max(0,idx-20):idx+20])}')
    
idx2 = text.find('{[}7{]}')
if idx2 != -1:
    print(f'\nContext around {{[}}7{{]}}: {repr(text[max(0,idx2-20):idx2+20])}')