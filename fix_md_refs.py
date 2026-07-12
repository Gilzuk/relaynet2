import re

# Mapping of hardcoded strings to markdown anchor links
replacements = {
    # ch02 equivalents
    r'\bSection 4\b(?!\.\d)': r'[Chapter 4](#sec:experiments)',
    r'\bSection 1\.3\.1\b': r'[Section 1.3.1](#sec:theoretical-basis-universal-approximation-and-denoising)',
    r'\bSection 1\.3\b': r'[Section 1.3](#sec:machine-learning-in-wireless-communication)',
    r'\bSection 1\.7\b': r'[Section 1.7](#sec:research-gap-and-motivation)',

    # ch03 equivalents
    r'\bSection 3\.7\.6\b': r'[Section 3.7.6](#sec:iq-splitting-for-ai-relay-processing-of-complex-constellations)',
    r'\bSection 3\.7\b': r'[Section 3.7](#sec:modulation-schemes)',
    r'\bFigure 1\b': r'[Figure 1](#fig:relay-system-detailed)',
    r'\bSection 7\b': r'[Chapter 7](#sec:experiments)',
    r'\bSection 3\.2\.1\b': r'[Section 3.2.1](#sec:awgn-channel-theoretical-analysis)',
    r'\bSection 3\.2\b': r'[Section 3.2](#sec:theoretical-foundations-channel-models)',

    # Sections inside ch04
    r'\bSection 4\.8\b': r'[Section 4.8](#sec:parameter-normalization-complexity-trade-off)',
    r'\bSection 4\.10\b': r'[Section 4.10](#sec:higher-order-modulation-scalability-constellation-aware-training)',
    r'\bSection 4\.11\b': r'[Section 4.11](#sec:higher-order-modulation-scalability-constellation-aware-training)',
    r'\bSection 4\.12\b': r'[Section 4.12](#sec:higher-order-modulation-scalability-constellation-aware-training)',
    r'\bSection 4\.13\b': r'[Section 4.13](#sec:input-normalization-and-csi-injection)',
    r'\bSection 4\.14\b': r'[Section 4.14](#sec:input-normalization-and-csi-injection)',
    r'\bSection 4\.15\b': r'[Section 4.15](#sec:input-normalization-and-csi-injection)',
    r'\bSection 4\.16\b': r'[Section 4.16](#sec:end-to-end-joint-optimization)',
    r'\bSection 4\.17\b': r'[Section 4.17](#sec:class-2d-classification-for-qam16)',

    # ch05
    r'\bSection 5\.1\.3\b': r'[Section 5.1.3](#sec:channel-robustness)',
    r'\bSection 5\.1\b': r'[Section 5.1](#sec:interpretation-of-results)',
    r'\bSection 5\.3\.1\b': r'[Section 5.3.1](#sec:context-length-benchmark-validating-the-crossover-hypothesis)',
    r'\bSection 5\.3\b': r'[Section 5.3](#sec:state-space-vs.-attention-for-signal-processing)',

    # Tables/Figures
    r'\bFigure 21\b': r'[Figure 21](#fig:fig21)',
    r'\bTable 13\b': r'[Table 13](#tbl:table13)',
    r'\bTable 15\b': r'[Table 15](#tbl:table15)',
    r'\bTable 16\b': r'[Table 16](#tbl:table16)',
    r'\bTable 24\b': r'[Table 24](#tbl:table24)',
}

with open('thesis.md', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content
for k, v in replacements.items():
    new_content = re.sub(k, v, new_content)

# Clean up any "Figure 1:" captions that might be there, just like we did in LaTeX
new_content = re.sub(r'\\caption\{Figure 1:\s*', r'\\caption{', new_content)
new_content = re.sub(r'\bFigure 1 --- End-to-end', 'End-to-end', new_content)

with open('thesis.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Done fixing references in thesis.md.")