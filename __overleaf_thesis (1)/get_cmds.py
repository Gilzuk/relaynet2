import json
with open(r'C:\Users\gzukerma\.vscode\extensions\iamhyc.overleaf-workshop-0.15.8\package.json', 'r', encoding='utf-8') as f:
    pkg = json.load(f)

with open('overleaf_cmds.txt', 'w', encoding='utf-8') as out:
    cmds = pkg.get('contributes', {}).get('commands', [])
    for cmd in cmds:
        out.write(f'{cmd.get("command")}: {cmd.get("title")}\n')