"""strip_rev must not change what the document renders, only what it says.

The clean bundle (thesis_overleaf_clean.zip) is the submission copy. It is the
annotated sources with \\REV{...} annotations removed, and since main.tex
defines \\REV to discard its argument, the two must render identically. They did
not: the stripper swallowed the whitespace following every annotation, so an
annotation used mid-sentence closed the gap between two sentences
("...(Chapter 6).These additional...") in the submitted document only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from strip_rev import strip_rev


def test_inline_annotation_keeps_the_following_space():
    src = "ends here.\\REV{ a note} Next sentence starts.\n"
    assert strip_rev(src) == "ends here. Next sentence starts.\n"


def test_inline_annotation_keeps_a_following_newline():
    src = "ends here.\\REV{ a note}\nNext line.\n"
    assert strip_rev(src) == "ends here.\nNext line.\n"


def test_annotation_alone_on_its_line_takes_the_line_with_it():
    src = "para one.\n\\REV{a whole-line note}\npara two.\n"
    assert strip_rev(src) == "para one.\npara two.\n"


def test_paragraph_break_survives():
    src = "para one.\\REV{note}\n\npara two.\n"
    assert "\n\n" in strip_rev(src)


def test_nested_braces_are_matched():
    src = "before \\REV{outer \\emph{inner} rest} after\n"
    assert strip_rev(src) == "before  after\n"


def test_real_chapter_text_keeps_its_sentence_spacing():
    """The exact construction that produced the defect, from ch02."""
    src = ("outperform DF (Chapter~\\ref{sec:unknown-channels}).\\REV{ Rephrased "
           "``provide clear lower and upper bounds'' $\\to$ representative "
           "baselines.} These additional relay strategies are reviewed.\n")
    out = strip_rev(src)
    assert ").These" not in out
    assert "). These additional" in out
