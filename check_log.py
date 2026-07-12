import re
import os

if os.path.exists('thesis_tau.log'):
    with open('thesis_tau.log', 'r', encoding='utf-8') as f:
        text = f.read()

    undef_refs = set(re.findall(r'LaTeX Warning: Reference `(.*?)\' on page \d+ undefined', text))
    print('Undefined References:', undef_refs if undef_refs else 'None')
else:
    print('thesis_tau.log not found.')