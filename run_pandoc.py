import subprocess
import os

PANDOC = r"C:\Users\gzukerma\AppData\Local\Pandoc\pandoc.exe"
PANDOC_CROSSREF = r"C:\Users\gzukerma\AppData\Local\Pandoc\pandoc-crossref.exe"

cmd = [
    PANDOC,
    "thesis_preprocessed.md",
    "--from", "markdown+raw_tex+tex_math_dollars+fenced_code_blocks",
    "--to", "latex",
    "--standalone",
    "--filter", PANDOC_CROSSREF,
    "--metadata-file", "thesis_meta.yaml",
    "--output", "thesis.tex",
    "--wrap", "none",
    "--top-level-division=chapter",
    "--shift-heading-level-by=-1",
    "-V", "mainfont=Times New Roman",
    "-V", "sansfont=Arial",
    "-V", "monofont=Courier New",
]

result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
print(f"Exit code: {result.returncode}")
if result.stderr:
    print("STDERR:", result.stderr[:3000])
if result.returncode == 0:
    size = os.path.getsize("thesis.tex")
    print(f"thesis.tex: {size:,} bytes")
    with open("thesis.tex", "r", encoding="utf-8") as f:
        tex = f.read()
    bs = chr(92)
    print(f"chapters:  {tex.count(bs + 'chapter')}")
    print(f"sections:  {tex.count(bs + 'section')}")
    print(f"figures:   {tex.count(bs + 'begin{figure}')}")
    print(f"tables:    {tex.count(bs + 'begin{table}')}")
    print(f"equations: {tex.count(bs + 'begin{equation}')}")
    print(f"labels:    {tex.count(bs + 'label')}")
else:
    print("FAILED")