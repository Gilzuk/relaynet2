import re

with open('thesis_tau.log', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Find the LAST occurrence of "Output written"
pdf_idx = text.rfind('Output written on thesis_tau.pdf')
xdv_idx = text.rfind('Output written on thesis_tau.xdv')

with open('final_refs.txt', 'w', encoding='utf-8') as f:
    f.write(f'PDF idx: {pdf_idx}\n')
    f.write(f'XDV idx: {xdv_idx}\n')
    f.write(f'Log length: {len(text)}\n\n')

    # The final pass is the LAST xelatex run
    # Find all "Run number X of rule 'xelatex'" markers
    runs = [(m.start(), m.group(1)) for m in re.finditer(r"Run number (\d+) of rule 'xelatex'", text)]
    f.write(f'XeLaTeX runs: {runs}\n\n')

    if runs:
        # Get the last run
        last_run_start = runs[-1][0]
        last_run_text = text[last_run_start:]
        undef = set(re.findall(r"LaTeX Warning: Reference `([^']+)' on page", last_run_text))
        f.write(f'Undefined refs in LAST xelatex run: {len(undef)}\n')
        for r in sorted(undef):
            f.write(f'  {r}\n')
    else:
        undef = set(re.findall(r"LaTeX Warning: Reference `([^']+)' on page", text))
        f.write(f'Undefined refs (all): {len(undef)}\n')
        for r in sorted(undef):
            f.write(f'  {r}\n')