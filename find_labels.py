import glob, re

labels = set()
for file in glob.glob('chapters/*.tex'):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        labels.update(re.findall(r'\\label\{([^}]+)\}', content))

undefs = [
    'fig:fig4', 'sec:experiments', 'sec:channel-model-validation',
    'sec:theoretical-basis-universal-approximation-and-denoising',
    'sec:higher-order-modulation-scalability-constellation-aware-training',
    'sec:input-normalization-and-csi-injection', 'sec:modulation-schemes',
    'sec:introduction', 'fig:relay-system-detailed', 'sec:awgn-channel-theoretical-analysis',
    'fig:fig8', 'sec:iq-splitting-for-ai-relay-processing-of-complex-constellations',
    'sec:class-2d-classification-for-qam16', 'chap:app-validation', 'chap:app-tables',
    'tbl:table15', 'tbl:table16', 'fig:fig21', 'chap:app-csi', 'tbl:table24',
    'sec:parameter-normalization-complexity-trade-off', 'tbl:table13',
    'sec:channel-robustness', 'sec:context-length-benchmark-validating-the-crossover-hypothesis',
    'sec:research-gap-and-motivation', 'sec:end-to-end-joint-optimization'
]

print('Missing labels in current .tex files:')
for u in undefs:
    if u not in labels:
        print(f'  {u}')

print(f'\nTotal defined labels: {len(labels)}')