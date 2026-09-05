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
