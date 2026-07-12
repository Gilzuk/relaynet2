import re

with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    text = f.read()

count_before = text.count(r'\label{fig:fig')
print(f'Labels before: {count_before}')

# Remove all \label{fig:figXX} patterns
new_text = re.sub(r'\\label\{fig:fig\d+\}', '', text)

count_after = new_text.count(r'\label{fig:fig')
print(f'Labels after: {count_after}')

with open('chapters/ch05_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Done')