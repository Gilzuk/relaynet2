"""The citation checker must catch a bad reference, not just pass good ones.

Two CI runs verified 7 of 7 candidates. That says nothing about whether the
checker can catch a fabrication: a checker that returned VERIFIED
unconditionally would have scored identically. FAIL has been exercised for
real (every lookup 403s from the development container), but MISMATCH -- the
record exists and the title does not match, which is the signature of a
mashed-up reference -- had never fired against anything. These tests fire it.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import verify_citations as vc


def _record(title):
    return {"title": title, "authors": ["A. Author"],
            "published": "2024-01-01T00:00:00Z", "updated": "2024-01-01T00:00:00Z",
            "abstract": "irrelevant", "id": "http://arxiv.org/abs/0000.00000"}


def _stub(monkeypatch, record):
    monkeypatch.setattr(vc, "fetch", lambda _id, retries=3: b"<xml/>")
    monkeypatch.setattr(vc, "parse", lambda _b: record)


REAL = "Neural Network-Aided BCJR Algorithm for Joint Symbol Detection and Channel Decoding"


def test_exact_match_verifies(monkeypatch):
    _stub(monkeypatch, _record(REAL))
    assert vc.verify({"arxiv_id": "x", "reported_title": REAL})["verdict"] == "VERIFIED"


def test_wrong_title_on_a_real_record_is_a_mismatch(monkeypatch):
    """The mashup case: the id resolves, but to a different paper."""
    _stub(monkeypatch, _record("Deep Learning for Symbol Detection in Optical Fibre"))
    r = vc.verify({"arxiv_id": "x", "reported_title": REAL})
    assert r["verdict"] == "MISMATCH"
    assert r["actual_title"] != r["reported_title"]


def test_plausible_but_altered_title_is_a_mismatch(monkeypatch):
    """A single swapped word is exactly what a fabricated citation looks like."""
    _stub(monkeypatch, _record(REAL.replace("BCJR", "Viterbi")))
    assert vc.verify({"arxiv_id": "x", "reported_title": REAL})["verdict"] == "MISMATCH"


def test_missing_record_is_a_fail(monkeypatch):
    _stub(monkeypatch, None)
    r = vc.verify({"arxiv_id": "x", "reported_title": REAL})
    assert r["verdict"] == "FAIL" and "no such record" in r["reason"]


def test_lookup_error_is_a_fail_not_an_uncertain(monkeypatch):
    """IRON RULE #4: the gray zone is a FAIL."""
    def boom(_id, retries=3):
        raise RuntimeError("HTTP Error 403: Forbidden")
    monkeypatch.setattr(vc, "fetch", boom)
    r = vc.verify({"arxiv_id": "x", "reported_title": REAL})
    assert r["verdict"] == "FAIL"


def test_cosmetic_differences_do_not_trip_mismatch(monkeypatch):
    """Whitespace, case and punctuation are not evidence of fabrication."""
    _stub(monkeypatch, _record("  neural network-aided BCJR algorithm for joint\n"
                               "symbol detection and channel decoding.  "))
    assert vc.verify({"arxiv_id": "x", "reported_title": REAL})["verdict"] == "VERIFIED"


# --- main() dispatch -------------------------------------------------------
# The tests above call verify() directly with a stubbed lookup, so none of them
# executed main(). That is how a hard c['arxiv_id'] survived in main()'s
# progress line while verify() had already moved to .get(): the first DOI-only
# candidate raised KeyError on CI. These cover the dispatch itself.

import json
import pytest


def _run_main(monkeypatch, tmp_path, candidates, record=None):
    src = tmp_path / "cands.json"
    src.write_text(json.dumps({"topic": "t", "candidates": candidates}))
    out = tmp_path / "out.json"
    monkeypatch.setattr(vc, "fetch", lambda _i, retries=3: b"<xml/>")
    monkeypatch.setattr(vc, "parse", lambda _b: record)
    monkeypatch.setattr(vc, "fetch_json", lambda _u, retries=3: {
        "message": {"title": [record["title"]], "author": [], "issued": {}}
        if record else {}})
    monkeypatch.setattr(vc.time, "sleep", lambda _s: None)
    monkeypatch.setattr(vc.sys, "argv",
                        ["verify_citations.py", "--candidates", str(src),
                         "--out", str(out)])
    code = vc.main()
    return code, json.loads(out.read_text())


def test_main_handles_a_doi_only_candidate(monkeypatch, tmp_path):
    """The exact CI failure: no arxiv_id key at all."""
    code, out = _run_main(monkeypatch, tmp_path,
                          [{"doi": "10.1109/TSP.2019.2899805",
                            "reported_title": REAL}], _record(REAL))
    assert out["results"][0]["verdict"] == "VERIFIED"
    assert out["results"][0]["route"] == "crossref-doi"


def test_main_handles_a_title_only_candidate(monkeypatch, tmp_path):
    code, out = _run_main(monkeypatch, tmp_path,
                          [{"reported_title": REAL}], _record(REAL))
    assert out["results"][0]["route"] == "crossref-title"


def test_main_handles_a_mixed_corpus(monkeypatch, tmp_path):
    code, out = _run_main(monkeypatch, tmp_path, [
        {"arxiv_id": "2006.01125", "reported_title": REAL},
        {"doi": "10.1109/TSP.2019.2899805", "reported_title": REAL},
        {"reported_title": REAL},
    ], _record(REAL))
    assert [r["route"] for r in out["results"]] == \
        ["arxiv", "crossref-doi", "crossref-title"]


def test_main_rejects_a_candidate_with_nothing_to_look_up(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, [{"note": "no id, no title"}], None)
