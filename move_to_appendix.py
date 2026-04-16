import re

with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    ch05 = f.read()

# 1. Extract Channel Model Validation
# Find the start of the section and the start of the next section
match_val = re.search(r'(\\section\{Channel Model Validation\}.*?)(?=\\section\{)', ch05, re.DOTALL)
if match_val:
    val_content = match_val.group(1)
    ch05 = ch05.replace(val_content, "\\section{Channel Model Validation}\\label{sec:channel-model-validation}\n\nThe detailed experimental validation of the channel models, including comparison against closed-form theoretical BER expressions (AWGN, Rayleigh, Rician, and MIMO configurations), is provided in \\textbf{Appendix~\\ref{chap:app-validation}}.\n\n")
    print("Extracted Channel Model Validation.")

# 2. Extract Input Normalization and CSI Injection (48 variants)
# This is a large section. We'll leave a summary and move the bulk to appendix.
match_csi = re.search(r'(\\section\{Input Normalization and CSI Injection\}.*?)(?=\\section\{)', ch05, re.DOTALL)
if match_csi:
    csi_content = match_csi.group(1)
    summary = r"""\section{Input Normalization and CSI Injection}\label{sec:input-normalization-and-csi-injection}

To evaluate the impact of explicit Channel State Information (CSI) injection and Input Layer Normalization across 48 distinct neural relay configurations (for 16-QAM and 16-PSK), an exhaustive hyperparameter sweep was conducted. Due to the volume of data, the full experimental methodology, BER vs. SNR charts, and detailed performance tables for all 48 variants have been relegated to \textbf{Appendix~\ref{chap:app-csi}}.

The key finding from this extensive experiment is a clear dichotomy: explicit CSI injection significantly improves performance for phase-only modulations (16-PSK) by resolving phase ambiguity, but actively degrades performance for amplitude-dependent modulations (16-QAM) by disrupting the delicate amplitude-decision boundaries. Input Layer Normalization proved universally beneficial.
"""
    ch05 = ch05.replace(csi_content, summary + "\n")
    print("Extracted Input Normalization and CSI Injection.")

# 3. Extract Long Tables
# We'll find all \begin{longtable}...\end{longtable} that contain "BER" or similar.
# Actually, the user asked to move "the long tables" to the appendix. We can just yank ALL \begin{longtable} blocks from Ch05 and put them in an Appendix.
longtables = re.findall(r'(\\begin\{longtable\}.*?\\end\{longtable\})', ch05, re.DOTALL)
extracted_tables = []
for idx, tbl in enumerate(longtables):
    extracted_tables.append(tbl)
    ch05 = ch05.replace(tbl, f"\\textit{{[Detailed tabular results have been moved to Appendix~\\ref{{chap:app-tables}}]}}\n")
print(f"Extracted {len(longtables)} long tables.")

with open('chapters/ch05_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(ch05)

# 4. Append to ch09_appendices.tex
with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    ch09 = f.read()

new_appendices = ""

if match_val:
    new_appendices += "\n\n\\chapter{Channel Model Validation Charts}\\label{chap:app-validation}\n"
    # remove the original \section command from the extracted text so it doesn't double-up or we can keep it as section
    # Let's change \section{Channel Model Validation} to just text since the chapter is the title.
    clean_val = re.sub(r'\\section\{Channel Model Validation\}\\label\{[^\}]+\}', '', val_content).strip()
    new_appendices += clean_val + "\n"

if extracted_tables:
    new_appendices += "\n\n\\chapter{Detailed Experimental Tables}\\label{chap:app-tables}\n"
    new_appendices += "This appendix contains the exhaustive numerical BER results for the experiments discussed in Chapter~\\ref{sec:experiments}.\n\n"
    for tbl in extracted_tables:
        new_appendices += tbl + "\n\n"

if match_csi:
    new_appendices += "\n\n\\chapter{48-Variant Neural Relay Experiments (CSI \& Normalization)}\\label{chap:app-csi}\n"
    clean_csi = re.sub(r'\\section\{Input Normalization and CSI Injection\}\\label\{[^\}]+\}', '', csi_content).strip()
    # Downgrade subsections to sections inside the appendix? 
    # \subsection{...} -> \section{...}
    clean_csi = re.sub(r'\\subsection', r'\\section', clean_csi)
    clean_csi = re.sub(r'\\subsubsection', r'\\subsection', clean_csi)
    new_appendices += clean_csi + "\n"

ch09 += new_appendices

with open('chapters/ch09_appendices.tex', 'w', encoding='utf-8') as f:
    f.write(ch09)

print("Appendices updated.")