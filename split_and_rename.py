
# 1. Read original ch01
with open('chapters/ch01_introduction.tex', 'r', encoding='utf-8') as f:
    text = f.read()

sections_list = re.split(r'(\\section\{[^}]+\}(?:\\label\{[^}]+\})?)', text)

ch1_parts = [sections_list[0].replace(r'\chapter{Introduction and Literature Review}\label{sec:introduction-and-literature-review}', r'\chapter{Introduction}\label{sec:introduction}')]
ch2_parts = [r'\chapter{Literature Review}\label{sec:literature-review}' + '\n\n']

# For ch2, we want to add an expanded section on DNNs.
dnn_expansion = r"""
\subsection{Deep Neural Networks for Physical Layer Design}

In addition to specific applications like channel estimation and detection, generic Deep Neural Networks (DNNs) have been thoroughly investigated for their ability to replace traditional block-based receiver pipelines. A DNN architecture---typically comprising multiple fully connected layers with non-linear activation functions (e.g., ReLU or $\tanh$)---is exceptionally well-suited for modeling complex, non-linear functional mappings. For instance, in relay networks suffering from non-Gaussian interference, hardware impairments, or severe multipath fading, linear and classical methods (such as AF or DF) often fall short because they assume simplified noise and channel models.

Recent literature highlights that DNNs can act as universal function approximators, implicitly learning the optimal detection boundary from large datasets without requiring explicit mathematical formulation of the channel state. O'Shea and Hoydis [9] demonstrated that end-to-end communication systems modeled entirely as neural networks could discover novel, efficient representations that conventional coding schemes do not account for. For relay networks specifically, feed-forward DNNs have been deployed to learn adaptive amplification factors and decoding thresholds simultaneously, effectively bypassing the rigid, hard-decision boundaries of Decode-and-Forward and the noise-amplifying characteristics of Amplify-and-Forward. By focusing on DNNs for these specific sub-tasks, researchers have proven that multi-layer neural architectures provide a flexible and highly robust alternative to modular, manually engineered digital communication pipelines.
"""

for i in range(1, len(sections_list), 2):
    header = sections_list[i]
    content = sections_list[i+1]
    
    if 'Machine Learning in Wireless' in header or 'Generative Models for' in header or 'Sequence Models' in header:
        if 'Machine Learning in Wireless' in header:
            # We append the expansion right after the Prior Work section
            content = content.replace(r'\subsection{Neural Network Relay Processing}', dnn_expansion + '\n\n' + r'\subsection{Neural Network Relay Processing}')
        ch2_parts.append(header + content)
    else:
        ch1_parts.append(header + content)

ch1_text = ''.join(ch1_parts)
ch2_text = ''.join(ch2_parts)

with open('chapters/ch01_introduction.tex', 'w', encoding='utf-8') as f:
    f.write(ch1_text)
    
with open('chapters/ch02_literature_review.tex', 'w', encoding='utf-8') as f:
    f.write(ch2_text)

print("Split ch01 into ch01 and ch02 successfully.")

# 2. Rename subsequent chapters
rename_map = {
    'chapters/ch02_objectives.tex': 'chapters/ch03_objectives.tex',
    'chapters/ch03_methods.tex': 'chapters/ch04_methods.tex',
    'chapters/ch04_experiments.tex': 'chapters/ch05_experiments.tex',
    'chapters/ch05_discussion.tex': 'chapters/ch06_discussion.tex',
    'chapters/ch06_equation_ref.tex': 'chapters/ch07_equation_ref.tex',
    'chapters/ch07_references.tex': 'chapters/ch08_references.tex',
    'chapters/ch08_appendices.tex': 'chapters/ch09_appendices.tex',
    'chapters/ch09_hebrew_abstract.tex': 'chapters/ch10_hebrew_abstract.tex'
}

for old_name, new_name in reversed(list(rename_map.items())):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"Renamed {old_name} to {new_name}")

# 3. Update main thesis_tau.tex
with open('thesis_tau.tex', 'r', encoding='utf-8') as f:
    main_tex = f.read()

main_tex = re.sub(r'\\include\{chapters/ch01_introduction\}', r'\\include{chapters/ch01_introduction}' + '\n' + r'\\include{chapters/ch02_literature_review}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch02_objectives\}', r'\\include{chapters/ch03_objectives}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch03_methods\}', r'\\include{chapters/ch04_methods}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch04_experiments\}', r'\\include{chapters/ch05_experiments}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch05_discussion\}', r'\\include{chapters/ch06_discussion}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch06_equation_ref\}', r'\\include{chapters/ch07_equation_ref}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch07_references\}', r'\\include{chapters/ch08_references}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch08_appendices\}', r'\\include{chapters/ch09_appendices}', main_tex)
main_tex = re.sub(r'\\include\{chapters/ch09_hebrew_abstract\}', r'\\include{chapters/ch10_hebrew_abstract}', main_tex)

with open('thesis_tau.tex', 'w', encoding='utf-8') as f:
    f.write(main_tex)
print("Updated thesis_tau.tex includes.")