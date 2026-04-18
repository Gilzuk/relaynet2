import re

with open('thesis_tau.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all titlepage blocks
title_pages = re.findall(r'\\begin\{titlepage\}.*?\\end\{titlepage\}', text, re.DOTALL)
print(f'Found {len(title_pages)} title pages')

if len(title_pages) >= 1:
    with open('chapters/ch00_cover_page.tex', 'w', encoding='utf-8') as f:
        f.write(title_pages[0])
    print('Wrote chapters/ch00_cover_page.tex')

if len(title_pages) >= 2:
    with open('chapters/ch00_front_page.tex', 'w', encoding='utf-8') as f:
        f.write(title_pages[1])
    print('Wrote chapters/ch00_front_page.tex')
else:
    # If only 1 title page, maybe write a dummy or just use the first one twice?
    # I'll just write the first one as cover page.
    pass

# Replace in main file
if title_pages:
    replacement = '\\include{chapters/ch00_cover_page}\n\\include{chapters/ch00_front_page}'
    
    # We replace the first title page and remove the second if it exists
    new_text = text.replace(title_pages[0], replacement)
    if len(title_pages) >= 2:
        new_text = new_text.replace(title_pages[1], '')
        
    with open('thesis_tau.tex', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Updated thesis_tau.tex')