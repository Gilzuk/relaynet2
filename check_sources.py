with open('chapters/ch04_methods.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all Source: occurrences
idx = 0
count = 0
while True:
    idx = text.find('Source:', idx)
    if idx == -1:
        break
    print(f'Position {idx}: {repr(text[max(0,idx-30):idx+80])}')
    print()
    idx += 1
    count += 1

print(f'Total: {count}')