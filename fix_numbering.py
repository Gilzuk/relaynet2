"""
Fix thesis.md numbering:
  - Pre-matter sections get {.unnumbered} so pandoc skips them in Arabic counter
  - Body chapters have old N. prefix stripped; pandoc auto-numbers from 1
  - Hand-written ToC rewritten: Roman numerals for pre-matter, Arabic 1-7 for body
  - Section cross-references in body text remapped (old ch4->1, ch5->2, etc.)
"""
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("thesis.md", encoding="utf-8") as f:
    content = f.read()

# 1. Strip "N. " prefix from ## headings (e.g. "## 4. Introduction")
content = re.sub(r'^(#{1,4}) \d+\. ', r'\1 ', content, flags=re.MULTILINE)

# 2. Strip "N.M " prefix from ### / #### headings (e.g. "### 4.1 Coop")
content = re.sub(r'^(#{2,4}) \d+\.\d+(?:\.\d+)? ', r'\1 ', content, flags=re.MULTILINE)

# 3. Add {.unnumbered} to pre-matter ## headings
for pat in [
    r"Table of Contents",
    r"List of Figures",
    r"List of Tables",
    r"List of Abbreviations",
    r"Abstract \(Hebrew\)",
    r"Keywords",
    r"Abstract \(English\)",
]:
    content = re.sub(
        r'^(## ' + pat + r')$',
        r'\1 {.unnumbered}',
        content,
        flags=re.MULTILINE,
    )

# 4. Replace the hand-written ToC block
NEW_TOC = (
    "## Table of Contents {.unnumbered}\n"
    "\n"
    "**Front Matter**\n"
    "\n"
    "| | |\n"
    "|---|---|\n"
    "| I | List of Abbreviations |\n"
    "| II | Abstract (Hebrew) |\n"
    "| III | Keywords |\n"
    "\n"
    "**Body**\n"
    "\n"
    "1. [Introduction and Literature Review](#introduction-and-literature-review)\n"
    "   - 1.1 Cooperative Relay Communication\n"
    "   - 1.2 Classical Relay Strategies\n"
    "   - 1.3 Machine Learning in Wireless Communication\n"
    "   - 1.4 Generative Models for Signal Processing\n"
    "   - 1.5 Sequence Models: Transformers and State Space Models\n"
    "   - 1.6 MIMO Systems and Equalization\n"
    "   - 1.7 Research Gap and Motivation\n"
    "2. [Research Objectives](#research-objectives)\n"
    "   - 2.1 Main Objective\n"
    "   - 2.2 Research Hypotheses\n"
    "   - 2.3 Specific Objectives\n"
    "   - 2.4 Scope and Delimitations\n"
    "3. [Methods](#methods)\n"
    "   - 3.1 System Model\n"
    "     - 3.1.1 MIMO Topology with Neural Network Relay and Equalization\n"
    "   - 3.2 Channel Models\n"
    "     - 3.2.1 AWGN Channel\n"
    "     - 3.2.2 Rayleigh Fading Channel\n"
    "     - 3.2.3 Rician Fading Channel\n"
    "     - 3.2.4 Fading Coefficient Distributions\n"
    "     - 3.2.5 2x2 MIMO Channel\n"
    "     - 3.2.6 Channel Model Validation\n"
    "   - 3.3 Relay Strategies\n"
    "   - 3.4 MIMO Equalization Techniques\n"
    "   - 3.5 Simulation Framework\n"
    "   - 3.6 Normalized Parameter Comparison\n"
    "   - 3.7 Modulation Schemes\n"
    "4. [Results](#results)\n"
    "   - 4.1 Channel Model Validation\n"
    "   - 4.2 AWGN Channel\n"
    "   - 4.3 Rayleigh Fading Channel\n"
    "   - 4.4 Rician Fading Channel (K=3)\n"
    "   - 4.5 2x2 MIMO with ZF Equalization\n"
    "   - 4.6 2x2 MIMO with MMSE Equalization\n"
    "   - 4.7 2x2 MIMO with SIC Equalization\n"
    "   - 4.8 Normalized 3K-Parameter Comparison\n"
    "   - 4.9 Complexity-Performance Trade-off\n"
    "   - 4.10 Modulation Comparison\n"
    "   - 4.11 16-QAM Activation Experiment\n"
    "   - 4.12 Constellation-Aware Activation Study\n"
    "   - 4.13 Input Layer Normalization and Scaled Tanh\n"
    "   - 4.14 Structural CSI Injection\n"
    "   - 4.15 Comprehensive Multi-Architecture CSI Experiment\n"
    "   - 4.16 End-to-End Joint Optimization\n"
    "   - 4.17 16-Class 2D Classification for QAM16\n"
    "5. [Discussion and Conclusions](#discussion-and-conclusions)\n"
    "   - 5.1 Interpretation of Results\n"
    "   - 5.2 The Less is More Principle\n"
    "   - 5.3 State Space vs. Attention for Signal Processing\n"
    "   - 5.4 Practical Deployment Recommendations\n"
    "   - 5.5 Limitations\n"
    "   - 5.6 Future Work\n"
    "   - 5.7 Conclusions\n"
    "6. [References](#references)\n"
    "7. [Appendices](#appendices)\n"
    "   - Appendix A: Mathematical Notation\n"
    "   - Appendix B: Model Architectures and Hyperparameters\n"
    "   - Appendix C: Software Architecture\n"
    "   - Appendix D: Normalized 3K-Parameter Configurations\n"
    "\n"
    "**Back Matter**\n"
    "\n"
    "| | |\n"
    "|---|---|\n"
    "| VIII | Abstract (English) |\n"
)

toc_start = content.find("## Table of Contents")
lof_start = content.find("\n## List of Figures")
if toc_start != -1 and lof_start != -1:
    content = content[:toc_start] + NEW_TOC + "\n" + content[lof_start + 1:]
    print("ToC block replaced OK")
else:
    print("WARNING: could not locate ToC block  toc=", toc_start, " lof=", lof_start)

# 5. Remap "Section X.Y" cross-references in body text
# Old: Intro=4, Objectives=5, Methods=6, Results=7, Discussion=8
# New: Intro=1, Objectives=2, Methods=3, Results=4, Discussion=5
# Strategy: only remap the explicit "Section N.M" pattern to avoid
# corrupting numeric values like BER numbers, SNR values, etc.
CHAPTER_MAP = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5}

def remap_section_ref(m):
    old_ch = int(m.group(1))
    rest   = m.group(2)   # e.g. ".10", ".15.3"
    new_ch = CHAPTER_MAP.get(old_ch, old_ch)
    return "Section " + str(new_ch) + rest

# Only remap the word "Section" followed by a chapter number in 4-8
# Use a negative lookahead to avoid matching things like "Section 4.3.1.2"
content = re.sub(
    r'Section ([4-8])(\.[0-9]+(?:\.[0-9]+)?)(?=[^0-9]|$)',
    remap_section_ref,
    content,
)

# 6. Save
with open("thesis.md", "w", encoding="utf-8") as f:
    f.write(content)

# 7. Report
headings = re.findall(r'^#{1,3} .+', content, re.MULTILINE)
print(f"\nTop-level headings after fix ({len(headings)} total):")
for h in headings[:65]:
    print(" ", h)
