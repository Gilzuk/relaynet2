import os, json

for root, dirs, files in os.walk('results'):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            with open(path, 'r') as fh:
                try:
                    d = json.load(fh)
                    relays = list(d.get('results', {}).keys())[:3]
                    n = len(d.get('results', {}))
                    print(f"{path}: {n} variants, e.g. {relays}")
                except Exception as e:
                    print(f"{path}: parse error {e}")