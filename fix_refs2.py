import glob, re, os

replacements = {
    # ch02
    r'\bSection 4\b(?!\.\d)': r'Chapter~\\ref{sec:experiments}',
    r'\bSection 1\.3\.1\b': r'Section~\\ref{sec:theoretical-basis-universal-approximation-and-denoising}',
    r'\bSection 1\.3\b': r'Section~\\ref{sec:machine-learning-in-wireless-communication}',
    r'\bSection 1\.7\b': r'Section~\\ref{sec:research-gap-and-motivation}',

    # ch03
    r'\bSection 3\.7\.6\b': r'Section~\\ref{sec:iq-splitting-for-ai-relay-processing-of-complex-constellations}',
    r'\bSection 3\.7\b': r'Section~\\ref{sec:modulation-schemes}',
    r'\bFigure 1\b': r'Figure~\\ref{fig:relay-system-detailed}',
    r'\bSection 7\b': r'Chapter~\\ref{sec:experiments}',
    r'\bSection 3\.2\.1\b': r'Section~\\ref{sec:awgn-channel-theoretical-analysis}',
    r'\bSection 3\.2\b': r'Section~\\ref{sec:theoretical-foundations-channel-models}',

    # Sections inside ch04
    r'\bSection 4\.8\b': r'Section~\\ref{sec:parameter-normalization-complexity-trade-off}',
    r'\bSection 4\.10\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.11\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.12\b': r'Section~\\ref{sec:higher-order-modulation-scalability-constellation-aware-training}',
    r'\bSection 4\.13\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.14\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.15\b': r'Section~\\ref{sec:input-normalization-and-csi-injection}',
    r'\bSection 4\.16\b': r'Section~\\ref{sec:end-to-end-joint-optimization}',
    r'\bSection 4\.17\b': r'Section~\\ref{sec:class-2d-classification-for-qam16}',

    # ch05
    r'\bSection 5\.1\.3\b': r'Section~\\ref{sec:channel-robustness}',
    r'\bSection 5\.1\b': r'Section~\\ref{sec:interpretation-of-results}',
    r'\bSection 5\.3\.1\b': r'Section~\\ref{sec:context-length-benchmark-validating-the-crossover-hypothesis}',
    r'\bSection 5\.3\b': r'Section~\\ref{sec:state-space-vs.-attention-for-signal-processing}',

    # Tables/Figures
    r'\bFigure 21\b': r'Figure~\\ref{fig:fig21}',
    r'\bTable 13\b': r'Table~\\ref{tbl:table13}',
    r'\bTable 15\b': r'Table~\\ref{tbl:table15}',
    r'\bTable 16\b': r'Table~\\ref{tbl:table16}',
    r'\bTable 24\b': r'Table~\\ref{tbl:table24}',
}

# Fix hardcoded "Figure 1: " in ch04 captions.
def clean_ch04_captions(content):
    content = re.sub(r'\\caption\{Figure 1:\s*', r'\\caption{', content)
    content = re.sub(r'\\emph\{Figure 1 --- End-to-end.*?\}', '', content)
    return content

files = ['chapters/ch02_objectives.tex', 'chapters/ch03_methods.tex', 'chapters/ch04_experiments.tex', 'chapters/ch05_discussion.tex']
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    if 'ch04_experiments.tex' in file_path:
        new_content = clean_ch04_captions(new_content)
        # For Figure 1 inside ch04, let's not apply the "Figure~\ref{fig:relay-system-detailed}" blindly because it might be the AWGN theory plot
        # The AWGN theory plot is actually fig:fig1 in ch01, but the script `fix_figures_split.py` might have left "Figure 1".
        # We will manually replace "\bFigure 1\b" in ch04 with "\ref{fig:fig1}" if it refers to the theory plot.
        # Actually, in ch04, Figure 1 refers to nothing since theoretical plots are in ch01. But let's apply the replacements carefully.
        
    for k, v in replacements.items():
        # Avoid replacing Figure 1 in ch04 if we shouldn't. Wait, the AWGN figure is not in ch04 anymore.
        # So any "Figure 1" in ch04 is likely a leftover. We already stripped "\emph{Figure 1...}"
        if 'ch04' in file_path and k == r'\bFigure 1\b':
            continue
        new_content = re.sub(k, v, new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")

print("Done fixing references.")