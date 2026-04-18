with open(r'C:\Users\gzukerma\.vscode\extensions\iamhyc.overleaf-workshop-0.15.8\README.md', 'r', encoding='utf-8') as f:
    text = f.read()

with open('overleaf_readme.txt', 'w', encoding='utf-8') as out:
    out.write(text)