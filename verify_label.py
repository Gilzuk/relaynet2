with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    text = f.read()
if 'chap:app-experiments' in text:
    print('Label added successfully')
else:
    print('ERROR: Label not added')