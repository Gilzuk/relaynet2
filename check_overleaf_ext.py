import json

with open(r'C:\Users\gzukerma\.vscode\extensions\iamhyc.overleaf-workshop-0.15.8\package.json', 'r', encoding='utf-8') as f:
    pkg = json.load(f)

# Get configuration properties
config = pkg.get('contributes', {}).get('configuration', {})
if isinstance(config, list):
    for c in config:
        props = c.get('properties', {})
        for k, v in props.items():
            print(f'{k}: {v.get("description", "")[:100]}')
elif isinstance(config, dict):
    props = config.get('properties', {})
    for k, v in props.items():
        print(f'{k}: {v.get("description", "")[:100]}')