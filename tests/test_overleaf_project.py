"""The Overleaf project is the closure of main.tex -- not the thesis/ directory.

thesis/ carries the working trail alongside the document: superseded chapter
drafts (chapters/_*.tex), the review-response appendices main.tex no longer
includes, LaTeX build artefacts, changelogs, the supervisor-comment log. None
of it compiles into the thesis and none of it belongs on Overleaf. These tests
pin that boundary so a future change cannot quietly widen it.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pytest

from overleaf_project import THESIS, manifest, stage


@pytest.fixture(scope="module")
def man():
    return manifest()


@pytest.fixture(scope="module")
def staged(tmp_path_factory, man):
    d = tmp_path_factory.mktemp("overleaf_clean")
    stage(str(d), "clean", man)
    return d


def _relpaths(root):
    out = []
    for base, _, files in os.walk(root):
        for f in files:
            out.append(os.path.relpath(os.path.join(base, f), root))
    return sorted(out)


def test_main_tex_is_at_the_project_root(staged):
    """Overleaf compiles from the root; a project rooted at thesis/ would not."""
    assert os.path.exists(os.path.join(staged, "main.tex"))


def test_no_draft_chapters(staged):
    """chapters/_ch01_introduction.tex and friends are drafting history."""
    assert [p for p in _relpaths(staged)
            if os.path.basename(p).startswith("_")] == []


def test_no_build_artefacts(staged):
    bad = [p for p in _relpaths(staged)
           if p.endswith((".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk",
                          ".bbl", ".blg", ".lof", ".lot", ".xdv", ".synctex.gz"))]
    assert bad == []


def test_no_working_trail_files(staged):
    names = {os.path.basename(p) for p in _relpaths(staged)}
    for trail in ("CHANGELOG.md", "RERUN_CHANGELOG.md", "ak_comments.json",
                  "results_reference.html", "cover_letter.md", "main.pdf",
                  "OVERLEAF_SYNC.md"):
        assert trail not in names, f"{trail} is working trail, not the document"


def test_excluded_appendices_do_not_travel(staged):
    """main.tex deliberately comments these out; the closure must respect that."""
    names = {os.path.basename(p) for p in _relpaths(staged)}
    assert "ak_response_appendix.tex" not in names
    assert "appendix_f_review.tex" not in names


def test_clean_mode_carries_no_annotations(staged):
    for p in _relpaths(staged):
        if p.endswith(".tex"):
            body = open(os.path.join(staged, p)).read()
            assert "\\REV{" not in body, f"{p} still carries fix records"


def test_every_local_style_travels(staged, man):
    """A .sty that main.tex loads but the project omits = a build that fails."""
    assert man["styles"], "expected at least one local .sty (hebcal.sty)"
    for sty in man["styles"]:
        assert os.path.exists(os.path.join(staged, sty))


def test_local_styles_are_discovered_not_hardcoded(man):
    """hebrewcal.sty exists in thesis/ but nothing loads it; it must not travel."""
    assert "hebcal.sty" in man["styles"]
    assert "hebrewcal.sty" not in man["styles"]
    assert os.path.exists(os.path.join(THESIS, "hebrewcal.sty")), \
        "precondition: the unused duplicate is still present in thesis/"


def test_every_referenced_figure_travels(staged, man):
    for fig in man["figures"]:
        assert os.path.exists(os.path.join(staged, fig)), f"missing figure {fig}"


def test_annotated_mode_keeps_the_annotations(tmp_path, man):
    stage(str(tmp_path), "annotated", man)
    total = sum(open(os.path.join(base, f)).read().count("\\REV{")
                for base, _, fs in os.walk(tmp_path)
                for f in fs if f.endswith(".tex"))
    assert total > 0


def test_unknown_mode_is_rejected(tmp_path, man):
    with pytest.raises(ValueError):
        stage(str(tmp_path), "sanitised", man)
