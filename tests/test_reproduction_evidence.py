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


def test_isolated_worktree_executes_fresh_output_and_preserves_reference(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from scripts import reproduce_round2 as suite
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "scripts").mkdir()
    (repo / "results").mkdir()
    write_json(repo / "results/value.json", {"value": 2})
    (repo / "scripts/reproduction_worker.py").write_text(
        'from pathlib import Path\n'
        'p=Path("results/value.json")\n'
        'assert not p.exists(), "stale output was not removed"\n'
        'p.write_text(\'{"value": 2}\\n\')\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.setattr(suite, "ROOT", repo)
    job = dict(id="fixture", kind="analysis", worker="script", outputs=["results/value.json"], scope="fixture")
    run_dir = repo / "results/reproduction/test"
    run_dir.mkdir(parents=True)
    state = dict(run_id="test", source_commit=source, status="RUNNING", jobs=[dict(job, status="RUNNING")])
    rec = suite.run_job(job, run_dir, state, SimpleNamespace(threads=1, timeout_hours=1))
    assert rec["status"] == "MATCH"
    assert read_json_fixture(repo / "results/value.json") == {"value": 2}
    assert (run_dir / "fixture/outputs/results/value.json").exists()
    assert not (repo / "_reproduction_work/test/fixture").exists()


def read_json_fixture(path):
    return json.loads(path.read_text())


def test_archived_provenance_uses_pinned_producer_and_rejects_wrong_source(tmp_path, monkeypatch):
    import provenance_audit as audit
    path = "manifest.json"
    revision = "a" * 40
    monkeypatch.setattr(audit, "ROOT", str(tmp_path))
    monkeypatch.setattr(audit, "REGISTRY", {"archived": ("runner.py", [path], ["evidence"])} )
    monkeypatch.setattr(audit, "FROZEN_RUN_SOURCES", {path: revision})

    def fake_commit(path, ref="HEAD"):
        # The current runner is newer than the archived data, its frozen
        # producer is older. Only the latter actually generated this run.
        ts = 10 if ref == revision else (20 if path == "manifest.json" else 30)
        return [ref, "2026-10-17", "Test", "fixture", ts]

    monkeypatch.setattr(audit, "last_commit", fake_commit)
    write_json(tmp_path / path, {"source_commit": revision})
    rows, warnings = audit.audit()
    assert not warnings
    assert rows[0]["producer_revision"] == revision
    write_json(tmp_path / path, {"source_commit": "b" * 40})
    assert audit.audit()[1][0][1] == "FROZEN SOURCE MISMATCH"


def test_rng_bootstrap_repeats_all_generators_and_fails_closed_for_nondeterminism():
    import random
    import numpy as np
    import pytest
    from scripts.reproduction_worker import configure_rng
    torch = pytest.importorskip("torch")
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        job = next(j for j in jobs() if j["id"] == "canonical_native")
        configure_rng(job, torch)
        first = (random.random(), np.random.random(8), torch.rand(8))
        configure_rng(job, torch)
        second = (random.random(), np.random.random(8), torch.rand(8))
        assert first[0] == second[0]
        assert np.array_equal(first[1], second[1])
        assert torch.equal(first[2], second[2])
        assert torch.initial_seed() == job["seed"] == 20261017
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def test_rng_evidence_records_state_without_advancing_it(tmp_path, monkeypatch):
    import random
    import numpy as np
    import pytest
    from scripts import reproduction_worker as worker
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "git", lambda *a: "frozen-source")
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    job = next(j for j in jobs() if j["id"] == "coded_matched")
    worker.snapshot_rng(job, "start", torch)
    assert random.getstate() == py_state
    current = np.random.get_state()
    assert current[0] == np_state[0] and np.array_equal(current[1], np_state[1])
    assert current[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    saved = read_json_fixture(tmp_path / "reproduction_checkpoints/rng/coded_matched_start.json")
    assert saved["torch"]["cpu_state_hex"] == bytes(torch_state.tolist()).hex()
    assert saved["policy"]["numpy_training_shuffle_seed"] == 42
