"""
Convert thesis_doc.md -> thesis.docx using pandoc + pandoc-crossref.
Equation numbering is handled by pandoc-crossref ({#eq:eqN} labels).
"""
import os, sys, io, subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PANDOC_DIR   = r"C:\Users\gzukerma\AppData\Local\pandoc"
PANDOC_EXE   = os.path.join(PANDOC_DIR, "pandoc.exe")
CROSSREF_EXE = os.path.join(PANDOC_DIR, "pandoc-crossref.exe")

# Verify both binaries exist
for exe in (PANDOC_EXE, CROSSREF_EXE):
    if not os.path.exists(exe):
        sys.exit(f"ERROR: not found: {exe}")

cmd = [
    PANDOC_EXE,
    "thesis_doc.md",
    "--output", "thesis.docx",
    "--standalone",
    "--from", "markdown+tex_math_dollars+raw_tex+pipe_tables",
    "--reference-doc", "reference.docx",
    "--wrap=none",
    "--toc",
    "--toc-depth=3",
    "--number-sections",
    "--number-offset=0",
    "--mathml",
]

print("Running pandoc with pandoc-crossref...")
print(" ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(f"pandoc failed with code {result.returncode}")

if result.stderr:
    print("Warnings:", result.stderr[:500])

size = os.path.getsize("thesis.docx")
print(f"\nthesis.docx created — {size:,} bytes ({size // 1024} KB)")
