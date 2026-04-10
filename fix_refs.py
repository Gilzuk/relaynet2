import glob, re, os

# Mapping of hardcoded strings to latex \ref strings
replacements = {
    r'\bSection 4\.10\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.11\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.12\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.13\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.14\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.15\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.16\b': r'Section~\\ref{sec:end-to-end-joint-optimization}',
    r'\bSection 4\.17\b': r'Section~\\ref{sec:class-2d-classification-for-qam16}',
    r'\bSection 4\.8\b': r'Section~\\ref{sec:parameter-normalization-complexity-trade-off}',
    r'\bSection 4\b': r'Chapter~\\ref{sec:experiments}',
    r'\bSection 1\.3\.1\b': r'Section~\\ref{sec:theoretical-basis-universal-approximation-and-denoising}',
    r'\bSection 1\.3\b': r'Section~\\ref{sec:machine-learning-in-wireless-communication}',
    r'\bSection 1\.7\b': r'Section~\\ref{sec:research-gap-and-motivation}',
    r'\bSection 3\.2\.1\b': r'Section~\\ref{sec:awgn-channel-theoretical-analysis}',
    r'\bSection 3\.2\b': r'Section~\\ref{sec:theoretical-foundations-channel-models}',
    r'\bSection 3\.7\.6\b': r'Section~\\ref{sec:iq-splitting-for-ai-relay-processing-of-complex-constellations}',
    r'\bSection 3\.7\b': r'Section~\\ref{sec:modulation-schemes}',
    r'\bSection 5\.1\b': r'Section~\\ref{sec:interpretation-of-results}',
    r'\bSection 5\.3\b': r'Section~\\ref{sec:state-space-vs.-attention-for-signal-processing}',
    r'\bSection 7\b': r'Chapter~\\ref{sec:experiments}',
    r'\bFigure 1\b': r'Figure~\\ref{fig:fig1-system-detailed}',
    r'\bFigure 21\b': r'Figure~\\ref{fig:fig21}',
    r'\bTable 13\b': r'Table~\\ref{tbl:table13}',
    r'\bTable 15\b': r'Table~\\ref{tbl:table15}',
    r'\bTable 16\b': r'Table~\\ref{tbl:table16}',
    r'\bTable 24\b': r'Table~\\ref{tbl:table24}',
}

files = glob.glob('chapters/*.tex')
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    # We apply replacements
    for k, v in replacements.items():
        new_content = re.sub(k, v, new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")

print("Done fixing references.")