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

from overleaf_project import THESIS, local_styles, manifest, stage


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
    for sty in man["styles"]:
        assert os.path.exists(os.path.join(staged, sty))


def test_only_loaded_styles_travel(staged, man):
    """A .sty sitting in thesis/ that nothing loads must not be shipped."""
    shipped = {os.path.basename(p) for p in _relpaths(staged) if p.endswith(".sty")}
    assert shipped == set(man["styles"])


# The discovery rule is tested against synthetic sources rather than whatever
# main.tex happens to load today. It used to assert "hebcal.sty is discovered",
# which broke the moment \usepackage{hebcal} was commented out -- the test was
# pinning a fact about the document, not the behaviour it meant to protect.
def test_a_loaded_local_style_is_discovered():
    assert local_styles([r"\usepackage{hebcal}"]) == ["hebcal.sty"]


def test_a_ctan_package_is_not_treated_as_local():
    """amsmath has no thesis/amsmath.sty, so it must not be shipped."""
    assert local_styles([r"\usepackage{amsmath}"]) == []


def test_a_commented_out_package_is_not_discovered():
    """The exact case that arose: hebcal commented out must stop travelling."""
    assert local_styles([r"%\usepackage{hebcal}"]) == []
    assert local_styles([r"% \usepackage{hebcal}"]) == []


def test_discovery_reads_options_and_grouped_names():
    assert local_styles([r"\usepackage[utf8]{hebcal}"]) == ["hebcal.sty"]
    assert local_styles([r"\usepackage{amsmath,hebcal}"]) == ["hebcal.sty"]


def test_the_unused_duplicate_never_travels():
    """hebrewcal.sty is a byte-identical leftover; nothing loads it."""
    assert os.path.exists(os.path.join(THESIS, "hebrewcal.sty")), \
        "precondition: the unused duplicate is still present in thesis/"
    assert "hebrewcal.sty" not in manifest()["styles"]


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


def test_the_built_pdf_travels(staged):
    """The published repository is read as well as compiled, so the PDF ships."""
    assert os.path.exists(os.path.join(staged, "thesis.pdf"))


def test_the_pdf_does_not_shadow_overleafs_build_output(staged):
    """Overleaf writes main.pdf when it compiles main.tex. A source file of
    that name in the project root collides with it, so the shipped PDF is
    called thesis.pdf and main.pdf must not appear in the staged tree."""
    assert not os.path.exists(os.path.join(staged, "main.pdf"))


# --- publish lineage -------------------------------------------------------
# A rebuilt container has no local overleaf-dist. If ensure_branch starts a
# fresh orphan there, its lineage is unrelated to what is already published,
# every subsequent push reports the whole remote history as missing, and the
# overwrite guard can only be satisfied with --force. The guard exists to stop
# edits made in Overleaf being destroyed, so that path must not be the default.

import subprocess as _sp
import sys as _sys
_sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _git(d, *a):
    return _sp.run(["git", "-C", d, *a], capture_output=True, text=True,
                   check=True).stdout.strip()


def test_ensure_branch_adopts_the_published_branch(tmp_path, monkeypatch):
    published = tmp_path / "published"
    published.mkdir()
    _git(str(published), "init", "-q", "--initial-branch=main")
    (published / "main.tex").write_text("x")
    _git(str(published), "add", "-A")
    _git(str(published), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "published state")
    published_head = _git(str(published), "rev-parse", "HEAD")

    local = tmp_path / "local"
    local.mkdir()
    _git(str(local), "init", "-q")
    _git(str(local), "remote", "add", "thesis-repo", str(published))

    import overleaf_sync as osync
    monkeypatch.setattr(osync, "ROOT", str(local))
    created = osync.ensure_branch("overleaf-dist")

    assert created
    assert _git(str(local), "rev-parse", "overleaf-dist") == published_head, \
        "ensure_branch started a new orphan instead of continuing the " \
        "published lineage; every later push would need --force"


def test_ensure_branch_still_makes_a_root_when_nothing_is_published(tmp_path,
                                                                    monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    _git(str(local), "init", "-q")
    import overleaf_sync as osync
    monkeypatch.setattr(osync, "ROOT", str(local))
    assert osync.ensure_branch("overleaf-dist")
    assert _git(str(local), "rev-list", "--count", "overleaf-dist") == "1"
