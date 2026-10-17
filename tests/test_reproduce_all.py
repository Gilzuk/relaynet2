"""The full-reproduction plan (scripts/reproduce_all.py) stays complete and ordered.

Running the plan takes hours, so these tests check the plan itself: it covers
every registered experiment, every script exists, a step runs after the steps
whose result files it reads, and the default invocation runs nothing.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import reproduce_all as RA  # noqa: E402
from provenance_audit import REGISTRY  # noqa: E402


def test_plan_covers_every_registered_experiment():
    plan = RA.build_plan()
    planned = {o for s in plan for o in s["outputs"]}
    for name, (_script, outputs, _art) in REGISTRY.items():
        for o in outputs:
            assert o in planned, f"{name}: {o} is not produced by any planned step"


def test_every_step_script_exists():
    for s in RA.build_plan():
        assert os.path.exists(os.path.join(ROOT, s["script"])), s["script"]


def test_steps_run_after_what_they_read():
    plan = RA.build_plan()
    pos = {s["name"]: k for k, s in enumerate(plan)}
    for k, s in enumerate(plan):
        for dep in s["after"]:
            assert pos[dep] < k, f"{s['name']} is planned before {dep}, whose output it reads"


def test_matched_reruns_get_the_matched_metric():
    for s in RA.build_plan():
        if not any(o.endswith("_matched.json") for o in s["outputs"]):
            continue
        if "matched" not in s["script"]:
            assert s["env"] == RA.MATCHED_ENV, s["name"]
        elif "CODED_METRIC" in RA.source(s["script"]):
            assert s["env"].get("CODED_METRIC") == "matched", s["name"]


def test_default_invocation_runs_nothing():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "reproduce_all.py"), "--only", "MAC accounting"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    assert "Nothing was run" in r.stdout


def test_run_uses_a_worktree_and_leaves_published_results_alone(tmp_path):
    """One cheap, deterministic step through --run: the output is regenerated in a
    separate worktree, copied to the evidence folder and matches the committed file,
    and nothing in the main checkout changes."""
    before = subprocess.run(["git", "status", "--porcelain", "--", "results"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    ev, wt = tmp_path / "evidence", tmp_path / "wt"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "reproduce_all.py"), "--run",
                        "--only", "ISI required-SNR", "--no-verify",
                        "--evidence-dir", str(ev), "--worktree-dir", str(wt)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    import json
    cmp = json.loads((ev / "comparison.json").read_text())
    assert [c["status"] for c in cmp] == ["MATCH"]
    assert (ev / "outputs" / "results" / "e6_isi_penalty.json").exists()
    assert not wt.exists() or not any(wt.iterdir()), "worktree was not removed"
    after = subprocess.run(["git", "status", "--porcelain", "--", "results"], cwd=ROOT,
                           capture_output=True, text=True).stdout
    assert after == before
