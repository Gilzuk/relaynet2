"""Protect scientific evidence comparisons against accidental false matches."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reproduce_round2 import differences, compare, jobs, write_json


def test_censor_flags_are_not_numeric_booleans():
    assert differences({"lower_bound": True}, {"lower_bound": False})
    assert differences({"lower_bound": True}, {"lower_bound": 1})


def test_missing_target_and_shorter_trial_list_never_match():
    a = {"targets": {"0.001": {"reached": False}}, "trials": [0.1, 0.2]}
    b = {"targets": {}, "trials": [0.1]}
    assert {d["kind"] for d in differences(a, b)} == {"missing_key", "length"}


def test_ignored_paths_are_exact_and_ber_drift_is_retained():
    a = {"git_rev": "old", "result": {"git_rev": "old", "ber": 1e-4}}
    b = {"git_rev": "new", "result": {"git_rev": "new", "ber": 2e-4}}
    d = differences(a, b, ["/git_rev"])
    assert {r["path"] for r in d} == {"/result/git_rev", "/result/ber"}


def test_roundoff_nan_and_missing_are_distinct():
    assert not differences([0.1, float("nan")], [0.1 + 1e-16, float("nan")])
    assert differences([float("nan")], [0.0])[0]["kind"] == "nonfinite"
    d = differences([float("nan")], [None])
    assert d
    json.dumps(d, allow_nan=False)


def canonical_payload(learned=0.2, df=0.1):
    return {"snr_range": [0], "results": {
        "DF": {"ber_mean": [df], "ber_trials": [[df, df]]},
        "MLP": {"ber_mean": [learned], "ber_trials": [[learned, learned]]}}}


def test_training_changes_flagged_classical_changes_fail(tmp_path):
    job = next(j for j in jobs() if j["id"] == "canonical_native")
    path = job["outputs"][0]
    write_json(tmp_path / "references" / path, canonical_payload())
    write_json(tmp_path / "outputs" / path, canonical_payload(learned=0.3))
    r = compare(job, tmp_path)
    assert r["status"] == "RETRAINING_DIFFERENCE_REQUIRES_REVIEW"
    assert r["matching_reference"] == []
    write_json(tmp_path / "outputs" / path, canonical_payload(df=0.11))
    assert compare(job, tmp_path)["status"] == "MISMATCH"
    write_json(tmp_path / "outputs" / path, canonical_payload())
    assert compare(job, tmp_path)["matching_reference"] == [path]


def test_atomic_writer_rejects_invalid_numbers(tmp_path):
    import pytest
    p = tmp_path / "status.json"
    write_json(p, {"status": "RUNNING"})
    with pytest.raises(ValueError):
        write_json(p, {"bad": float("nan")})
    assert json.loads(p.read_text()) == {"status": "RUNNING"}
