with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    text = f.read()
import re
figs = re.findall(r'\\begin\{figure\}', text)
with open('ch05_state.txt', 'w', encoding='utf-8') as f:
    f.write(f'Figures in ch05: {len(figs)}\n')
    f.write(f'Lines: {len(text.splitlines())}\n')
    f.write('First 5 lines:\n')
    for line in text.splitlines()[:5]:
        f.write(line + '\n')