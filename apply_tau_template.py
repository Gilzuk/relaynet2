"""
apply_tau_template.py
---------------------
Takes the pandoc-generated thesis.tex and wraps it in the TAU thesis template.
Extracts the body content and inserts it into the TAU template structure.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Read pandoc-generated thesis.tex
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis.tex", "r", encoding="utf-8") as f:
    pandoc_tex = f.read()

print(f"Input thesis.tex: {len(pandoc_tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Extract body content (between \begin{document} and \end{document})
# ─────────────────────────────────────────────────────────────────────────────
body_m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", pandoc_tex, re.DOTALL)
if not body_m:
    print("ERROR: Could not find \\begin{document}...\\end{document}")
    exit(1)

body = body_m.group(1).strip()
print(f"Extracted body: {len(body):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Remove pandoc's auto-generated title block (maketitle etc.)
# ─────────────────────────────────────────────────────────────────────────────
# Remove \maketitle if present
body = re.sub(r"\\maketitle\s*\n?", "", body)

# ─────────────────────────────────────────────────────────────────────────────
# Extract the English abstract section from the body
# (it's in ## Abstract (English) {.unnumbered} → \chapter*{Abstract (English)})
# ─────────────────────────────────────────────────────────────────────────────
eng_abs_m = re.search(
    r"\\chapter\*\{Abstract \(English\)\}(.*?)(?=\\chapter|\Z)",
    body, re.DOTALL
)
eng_abstract = ""
if eng_abs_m:
    eng_abstract = eng_abs_m.group(1).strip()
    # Remove from body
    body = body[:eng_abs_m.start()] + body[eng_abs_m.end():]
    print(f"Extracted English abstract: {len(eng_abstract)} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Extract the Hebrew abstract section from the body
# ─────────────────────────────────────────────────────────────────────────────
heb_abs_m = re.search(
    r"\\chapter\*\{Abstract \(Hebrew\)\}(.*?)(?=\\chapter|\Z)",
    body, re.DOTALL
)
heb_abstract = ""
if heb_abs_m:
    heb_abstract = heb_abs_m.group(1).strip()
    body = body[:heb_abs_m.start()] + body[heb_abs_m.end():]
    print(f"Extracted Hebrew abstract: {len(heb_abstract)} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Remove the pandoc-generated TOC/LOF/LOT sections from body
# (we'll use LaTeX's \tableofcontents etc. in the template)
# ─────────────────────────────────────────────────────────────────────────────
# Remove \chapter*{Table of Contents} and its content (pandoc generates a manual TOC)
body = re.sub(
    r"\\chapter\*\{Table of Contents\}.*?(?=\\chapter|\Z)",
    "", body, flags=re.DOTALL
)
# Remove \chapter*{List of Figures} (pandoc manual list)
body = re.sub(
    r"\\chapter\*\{List of Figures\}.*?(?=\\chapter|\Z)",
    "", body, flags=re.DOTALL
)
# Remove \chapter*{List of Tables} (pandoc manual list)
body = re.sub(
    r"\\chapter\*\{List of Tables\}.*?(?=\\chapter|\Z)",
    "", body, flags=re.DOTALL
)
# Remove \chapter*{Equation Reference} section (keep in appendix)
# Actually keep it - it's useful

print(f"Body after cleanup: {len(body):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Extract pandoc-generated packages that we need to keep
# (longtable, etc.)
# ─────────────────────────────────────────────────────────────────────────────
# Get the pandoc preamble packages we need
preamble_m = re.search(r"\\documentclass.*?\\begin\{document\}", pandoc_tex, re.DOTALL)
preamble = preamble_m.group(0) if preamble_m else ""

# Extract specific packages from pandoc preamble that we need
needed_packages = []
for pkg in ["longtable", "booktabs", "array", "calc", "unicode-math",
            "upquote", "microtype", "selnolig", "bookmark"]:
    if f"\\usepackage{{{pkg}}}" in preamble or f"\\usepackage[" in preamble:
        pass  # we'll include them in our template

# Extract CSL references environment definition from pandoc output
csl_m = re.search(r"(\\newlength\{\\cslhangindent\}.*?\\end\{CSLReferences\}\})",
                  pandoc_tex, re.DOTALL)
csl_defs = csl_m.group(1) if csl_m else ""

# ─────────────────────────────────────────────────────────────────────────────
# Build the complete TAU thesis LaTeX file
# ─────────────────────────────────────────────────────────────────────────────

tau_tex = r"""%% ============================================================
%% TAU Thesis Template — Tel Aviv University
%% Deep Learning Architectures for Two-Hop Relay Communication
%% Gil Zukerma, M.Sc. Thesis, 2026
%% ============================================================
\documentclass[12pt,a4paper,oneside]{report}

%% ── Encoding & Fonts ─────────────────────────────────────────
\usepackage{iftex}
\ifXeTeX
  \usepackage{fontspec}
  \setmainfont{Times New Roman}
  \setsansfont{Arial}
  \setmonofont[Scale=0.9]{Courier New}
\else
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
\fi

%% ── Page Layout ──────────────────────────────────────────────
\usepackage[top=2.5cm, bottom=2.5cm, left=3.5cm, right=2.5cm,
            headheight=15pt]{geometry}
\usepackage{setspace}
\onehalfspacing

%% ── Language & Hebrew ────────────────────────────────────────
\usepackage{polyglossia}
\setmainlanguage{english}
\setotherlanguage{hebrew}
\ifXeTeX
  \newfontfamily\hebrewfont[Script=Hebrew]{Arial}
  \newfontfamily\hebrewfonttt[Script=Hebrew]{Courier New}
\fi

%% ── Mathematics ──────────────────────────────────────────────
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{bm}

%% ── Tables ───────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{makecell}

%% ── Figures ──────────────────────────────────────────────────
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\captionsetup{font=small, labelfont=bf, margin=10pt}
\captionsetup[table]{position=below}
\captionsetup[figure]{position=below}

%% ── Colors ───────────────────────────────────────────────────
\usepackage{xcolor}
\definecolor{tauBlue}{RGB}{0,56,101}
\definecolor{tauGray}{RGB}{100,100,100}
\definecolor{linkBlue}{RGB}{0,70,140}

%% ── Headers & Footers ────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{tauGray}{\nouppercase{\leftmark}}}
\fancyhead[R]{\small\textcolor{tauGray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{%
  \color{tauBlue}\leaders\hrule height \headrulewidth\hfill}}
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[C]{\small\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

%% ── Chapter & Section Headings ───────────────────────────────
\usepackage{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\color{tauBlue}}
  {\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-20pt}{40pt}

\titleformat{\section}
  {\normalfont\Large\bfseries\color{tauBlue}}
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries}
  {\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries}
  {\thesubsubsection}{1em}{}

%% ── TOC Formatting ───────────────────────────────────────────
\usepackage{tocloft}
\renewcommand{\cftchapfont}{\bfseries\color{tauBlue}}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftsecfont}{\small}
\renewcommand{\cftsubsecfont}{\small}
\setlength{\cftbeforechapskip}{6pt}
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

%% ── Hyperlinks ───────────────────────────────────────────────
\usepackage[colorlinks=true,
            linkcolor=linkBlue,
            citecolor=linkBlue,
            urlcolor=linkBlue,
            bookmarks=true,
            bookmarksnumbered=true,
            pdftitle={Deep Learning Architectures for Two-Hop Relay Communication},
            pdfauthor={Gil Zukerma},
            pdfsubject={M.Sc. Thesis, Tel Aviv University, 2026}]{hyperref}
\usepackage{cleveref}

%% ── Bibliography ─────────────────────────────────────────────
\usepackage[numbers,sort&compress]{natbib}

%% ── Code Listings ────────────────────────────────────────────
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny\color{tauGray},
  keywordstyle=\color{tauBlue}\bfseries,
  commentstyle=\color{tauGray}\itshape,
  stringstyle=\color{red!70!black},
}

%% ── Miscellaneous ────────────────────────────────────────────
\usepackage{microtype}
\usepackage{parskip}
\setlength{\parskip}{6pt}
\usepackage{enumitem}
\usepackage{csquotes}
\usepackage{url}

%% ── pandoc compatibility ─────────────────────────────────────
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

""" + csl_defs + r"""

%% ── Theorem environments ─────────────────────────────────────
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

%% ============================================================
\begin{document}

%% ── Title Page ───────────────────────────────────────────────
\begin{titlepage}
\begin{center}
\vspace*{0.5cm}

{\LARGE\textbf{TEL AVIV UNIVERSITY}}\\[6pt]
{\large The Iby and Aladar Fleischman Faculty of Engineering}\\[4pt]
{\large Department of Electrical Engineering}\\[1.5cm]

\noindent\rule{\textwidth}{2pt}\\[0.4cm]
{\LARGE\bfseries\color{tauBlue}
Deep Learning Architectures for\\[6pt]
Two-Hop Relay Communication:\\[6pt]
A Comparative Study of Classical and\\[6pt]
Neural Network Relay Strategies}\\[0.4cm]
\noindent\rule{\textwidth}{2pt}\\[1.2cm]

{\large\textit{A thesis submitted toward the degree of}}\\[4pt]
{\large\textbf{Master of Science in Electrical Engineering}}\\[1.5cm]

{\large by}\\[6pt]
{\Large\bfseries Gil Zukerma}\\[1.5cm]

{\large Submitted to the Senate of Tel Aviv University}\\[4pt]
{\large\textbf{2026}}\\[1.5cm]

\vfill
\begin{tabular}{ll}
\textbf{Thesis Advisor:} & [Supervisor Name]\\[4pt]
\textbf{Department:} & Electrical Engineering\\[4pt]
\textbf{Faculty:} & The Iby and Aladar Fleischman Faculty of Engineering\\
\end{tabular}
\end{center}
\end{titlepage}

%% ── Front Matter ─────────────────────────────────────────────
\frontmatter
\pagenumbering{roman}

%% ── Dedication ───────────────────────────────────────────────
\clearpage
\thispagestyle{empty}
\vspace*{5cm}
\begin{center}
\textit{Dedicated to the pursuit of knowledge\\
at the intersection of communications and machine learning.}
\end{center}
\clearpage

%% ── English Abstract ─────────────────────────────────────────
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract (English)}
\markboth{Abstract}{Abstract}

""" + eng_abstract + r"""

\clearpage

%% ── Acknowledgments ──────────────────────────────────────────
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments}
\markboth{Acknowledgments}{Acknowledgments}

I would like to thank my supervisors and colleagues at Tel Aviv University for their guidance and support throughout this research. This work was conducted in the Department of Electrical Engineering, Tel Aviv University.

\clearpage

%% ── Table of Contents ────────────────────────────────────────
\tableofcontents
\clearpage

%% ── List of Figures ──────────────────────────────────────────
\listoffigures
\addcontentsline{toc}{chapter}{List of Figures}
\clearpage

%% ── List of Tables ───────────────────────────────────────────
\listoftables
\addcontentsline{toc}{chapter}{List of Tables}
\clearpage

%% ── Main Body ────────────────────────────────────────────────
\mainmatter
\pagenumbering{arabic}

""" + body.strip() + r"""

%% ── Hebrew Abstract (Back Matter) ───────────────────────────
\backmatter
\clearpage
\begin{flushright}
\chapter*{\texthebrew{תקציר}}
\addcontentsline{toc}{chapter}{Abstract (Hebrew)}
\markboth{Abstract (Hebrew)}{Abstract (Hebrew)}
\end{flushright}

\begin{otherlanguage}{hebrew}
\begin{flushright}
""" + heb_abstract + r"""
\end{flushright}
\end{otherlanguage}

\end{document}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tau_tex)

print(f"\nOutput: thesis_tau.tex ({len(tau_tex):,} chars)")

# Verify structure
import re as re2
chapters = re2.findall(r"\\chapter\*?\{[^\}]+\}", tau_tex)
print(f"\nChapters ({len(chapters)}):")
for c in chapters[:20]:
    print(f"  {c[:80]}")

sections = re2.findall(r"\\section\*?\{[^\}]+\}", tau_tex)
print(f"\nSections: {len(sections)}")
print(f"Figures: {tau_tex.count(chr(92)+'begin{figure}')}")
print(f"Longtables: {tau_tex.count('longtable')}")
print(f"Equations: {tau_tex.count(chr(92)+'begin{equation}')}")