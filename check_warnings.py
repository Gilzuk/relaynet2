import re
import os

if not os.path.exists('thesis_tau.log'):
    print('Log file not found.')
    exit(1)

with open('thesis_tau.log', 'r', encoding='utf-8') as f:
    text = f.read()

warnings = re.findall(r'LaTeX Warning:.*?(?=\n\n|\n[A-Z]|\Z)', text, re.DOTALL)
overfull = re.findall(r'Overfull \\hbox.*?(?=\n\n|\Z)', text, re.DOTALL)

print(f'--- LaTeX Warnings ({len(warnings)}) ---')
for w in warnings:
    if 'polyglossia' not in w and 'Hebrew calendar' not in w and 'Label(s) may have changed' not in w:
        print(w.strip() + '\n')

print(f'--- Overfull hboxes ({len(overfull)}) ---')
for o in overfull:
    print(o.strip() + '\n')