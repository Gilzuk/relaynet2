"""
Create a reference.docx with TAU-compliant styles, then regenerate thesis.docx.
Uses python-docx to build the reference template.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import copy
except ImportError:
    os.system(r"C:\Users\gzukerma\source\repos\venv\Scripts\pip.exe install python-docx --quiet")
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# ── Page layout: A4, margins per TAU guidelines ──────────────────────────────
section = doc.sections[0]
section.page_height = Cm(29.7)
section.page_width  = Cm(21.0)
section.left_margin   = Cm(3.0)   # binding side
section.right_margin  = Cm(2.5)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)

styles = doc.styles

def set_para_style(style_name, font_name="Times New Roman", font_size=12,
                   bold=False, space_before=0, space_after=6,
                   line_spacing=None, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    try:
        s = styles[style_name]
    except KeyError:
        s = styles.add_style(style_name, 1)  # 1 = paragraph
    pf = s.paragraph_format
    pf.alignment = alignment
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if line_spacing:
        from docx.shared import Pt as _Pt
        pf.line_spacing = _Pt(line_spacing)
    rf = s.font
    rf.name      = font_name
    rf.size      = Pt(font_size)
    rf.bold      = bold
    rf.color.rgb = RGBColor(0, 0, 0)
    return s

# Normal body text — Times New Roman 12pt, 1.5 line spacing, justified
set_para_style("Normal", font_size=12, space_after=6, line_spacing=18)

# Headings
set_para_style("Heading 1", font_size=16, bold=True, space_before=18, space_after=6,
               alignment=WD_ALIGN_PARAGRAPH.LEFT)
set_para_style("Heading 2", font_size=14, bold=True, space_before=12, space_after=4,
               alignment=WD_ALIGN_PARAGRAPH.LEFT)
set_para_style("Heading 3", font_size=12, bold=True, space_before=10, space_after=4,
               alignment=WD_ALIGN_PARAGRAPH.LEFT)
set_para_style("Heading 4", font_size=12, bold=False, space_before=8, space_after=2,
               alignment=WD_ALIGN_PARAGRAPH.LEFT)

# Caption style
set_para_style("Caption", font_size=10, space_before=4, space_after=8,
               alignment=WD_ALIGN_PARAGRAPH.CENTER)

# Table text
set_para_style("Table Paragraph", font_size=10, space_before=2, space_after=2,
               alignment=WD_ALIGN_PARAGRAPH.LEFT)

# Abstract
set_para_style("Abstract", font_size=12, space_before=0, space_after=6, line_spacing=18)

# Block quote / code
set_para_style("Verbatim Char", font_name="Courier New", font_size=10,
               space_before=4, space_after=4, alignment=WD_ALIGN_PARAGRAPH.LEFT)

# Add a placeholder paragraph so the file is valid
doc.add_paragraph("Reference document for thesis formatting.", style="Normal")

doc.save("reference.docx")
print("reference.docx created")
