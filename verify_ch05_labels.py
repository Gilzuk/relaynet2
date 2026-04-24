import re
with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    text = f.read()
remaining = re.findall(r'\\label\{fig:fig\d+\}', text)
with open('ch05_labels.txt', 'w') as f:
    f.write(f'Remaining fig labels in ch05: {remaining}\n')
    f.write('Done\n')