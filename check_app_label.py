with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    text = f.read()
import re
labels = re.findall(r'\\label\{([^}]+)\}', text[:300])
with open('app_label_check.txt', 'w', encoding='utf-8') as f:
    f.write(f'First labels: {labels}\n')
    f.write(f'First 300 chars:\n{text[:300]}\n')
    # Check if chap:app-experiments is in the file
    if 'chap:app-experiments' in text:
        f.write('chap:app-experiments FOUND in file\n')
    else:
        f.write('chap:app-experiments NOT FOUND in file\n')