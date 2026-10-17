#!/usr/bin/env python3
"""Run the round-2 reproduction suite without overwriting published evidence.

Each job executes in its own detached worktree at one frozen source commit.
Fresh outputs, checkpoints, logs, reference hashes and comparisons live under
results/reproduction/<run-id>. A match is a link to the original JSON, not a
replacement for it. Training differences always require review, never an
automatic statistical-equivalence claim. See docs/round2_reproduction.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def differences(expected, actual, ignored=()):
    """Full recursive comparison; tolerances cover float roundoff, not BER drift.

    Ignore only declared exact JSON paths. Booleans, dimensions, missing keys,
    nonfinite values and strings are checked separately from numeric tolerance.
    Every mismatch is retained, including unsuccessful BER targets.
    """
    out = []

    def atom(x):
        return str(x) if isinstance(x, float) and not math.isfinite(x) else x

    def visit(a, b, path):
        if path in ignored:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                p = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if p in ignored:
                    continue
                if key not in a or key not in b:
                    out.append({"path": p, "kind": "missing_key", "expected_present": key in a,
                                "actual_present": key in b})
                else:
                    visit(a[key], b[key], p)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append({"path": path, "kind": "length", "expected": len(a), "actual": len(b)})
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(i))
        elif isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b:
                out.append({"path": path, "kind": "value", "expected": a, "actual": b})
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Historical JSONs contain NaN for unbracketed targets. Preserve it
            # as a named status in the comparison, not an invalid JSON number.
            if not math.isfinite(a) or not math.isfinite(b):
                if str(a) != str(b):
                    out.append({"path": path, "kind": "nonfinite", "expected": str(a), "actual": str(b)})
            elif not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14):
                out.append({"path": path, "kind": "numeric", "expected": a, "actual": b,
                            "abs_diff": abs(a - b)})
        elif type(a) is not type(b) or a != b:
            out.append({"path": path, "kind": "value", "expected": atom(a), "actual": atom(b)})

    visit(expected, actual, "")
    return out


def jobs():
    """Principal round-2 claims; no reduced-budget smoke runs in this manifest."""
    return [
        dict(id="common_target_analysis", kind="analysis", worker="script",
             script="penalty_common_targets.py", outputs=["results/penalty_common_targets.json"],
             ignored=["/git_rev"], scope="Table 6.7, Appendix 9.6 and minimum-size conclusions; saved-curve reanalysis"),
        dict(id="canonical_paired_analysis", kind="analysis", worker="script",
             script="canonical_paired_tests.py", outputs=["results/canonical_paired_tests.json"],
             ignored=[], scope="Tables 5.2–5.3 paired comparisons; saved-curve reanalysis"),
        dict(id="canonical_af_df", kind="deterministic_simulation", worker="baselines",
             outputs=["results/reproduction_canonical_af_df.json"],
             references=["results/modulation/qpsk_rayleigh.json", "results/normalized_3k/3k_rayleigh.json"],
             scope="Fresh AF/DF evaluation: 11 SNRs, 10 trials × 10,000 bits, trial seeds 0–9"),
        dict(id="factorized_mlse", kind="deterministic_simulation", worker="script",
             script="factorized_mlse_equivalence.py", outputs=["results/factorized_mlse_equivalence.json"],
             ignored=["/git_rev"], scope="Full joint/factorized control: L=3,5,7; six SNRs; 2,000 symbols per block; original depths"),
        dict(id="sequence_memory", kind="seeded_retraining", worker="sequence",
             outputs=["results/reproduction_seq_memory.json"],
             references=["results/seq_models_on_memory.json", "results/penalty_common_targets.json"],
             scope="All 3 channels × 4 architectures × seeds 0,1,2; original 40 epochs and 20×20,000-bit budget"),
        dict(id="coded_matched", kind="seeded_retraining", worker="script",
             script="coded_matched_metrics.py", outputs=["results/coded_matched_metrics.json"],
             env={"CMM_TRIALS": "30", "CMM_SNRS": "0,4,8,12,16,20"}, ignored=["/seconds"],
             scope="Full matched coded comparison: 30 trials × 500 frames at six SNRs; hard/soft MLP, no Mamba"),
        dict(id="canonical_native", kind="new_training_replication", worker="canonical", stage="7.10",
             outputs=["results/modulation/qpsk_rayleigh.json"], seed=20261017,
             scope="Original native protocol, 100 epochs; new declared PyTorch seed, absent historical checkpoints/RNG"),
        dict(id="canonical_3k", kind="new_training_replication", worker="canonical", stage="7.8",
             outputs=["results/normalized_3k/3k_rayleigh.json"], seed=20261017,
             scope="Original 3K protocol, 100 epochs; new declared PyTorch seed, absent historical checkpoints/RNG"),
    ]


def canonical_projection(data, classical_only=False):
    return {"snr_range": data["snr_range"], "results": {
        k: {field: v[field] for field in ("ber_mean", "ber_trials")}
        for k, v in data["results"].items() if not classical_only or k in ("AF", "DF")}}


def sequence_projection(data):
    return {"snr_db": data["snr_db"], "train_seeds": data["train_seeds"],
            "epochs": data["epochs"], "samples": data["samples"], "channels": {
                name: {"baseline": ch["baseline"], "baseline_ber": ch["baseline_ber"], "archs": {
                    arch: [{"seed": r["seed"], "ber": r["ber"]} for r in rec["runs"]]
                    for arch, rec in ch["archs"].items()}}
                for name, ch in data["channels"].items()}}


def compare(job, evidence):
    refs = job.get("references", job["outputs"])
    actual = read_json(evidence / "outputs" / job["outputs"][0])
    expected = read_json(evidence / "references" / refs[0])
    classical = []
    if job["worker"] == "baselines":
        changes = []
        for ref in refs:
            d = differences(canonical_projection(read_json(evidence / "references" / ref), True), actual)
            changes += [{**x, "reference": ref} for x in d]
    elif job["worker"] == "canonical":
        changes = differences(canonical_projection(expected), canonical_projection(actual))
        classical = differences(canonical_projection(expected, True), canonical_projection(actual, True))
    elif job["worker"] == "sequence":
        changes = differences(sequence_projection(expected), sequence_projection(actual))
        classical = differences({k: v["baseline_ber"] for k, v in expected["channels"].items()},
                                {k: v["baseline_ber"] for k, v in actual["channels"].items()})
    else:
        changes = differences(expected, actual, job.get("ignored", []))
        if job["id"] == "coded_matched":
            def controls(d):
                return {s: {k: v for k, v in rows.items() if not k.startswith("mlp_")}
                        for s, rows in d["per_trial"].items()}
            classical = differences(controls(expected), controls(actual))
    strict = job["kind"] in ("analysis", "deterministic_simulation")
    status = ("MATCH" if not changes else "MISMATCH" if strict or classical
              else "RETRAINING_DIFFERENCE_REQUIRES_REVIEW")
    result = {"status": status, "difference_count": len(changes), "differences": changes,
            "classical_control_differences": classical,
            "reference_files": refs, "matching_reference": refs if not changes else [],
            "ignored_paths": job.get("ignored", []), "numeric_tolerance": {"rtol": 1e-12, "atol": 1e-14},
            "scope": job["scope"],
            "interpretation": "MATCH means the selected scientific fields agree within roundoff; not byte identity, statistical equivalence or independent validation of every thesis claim."}
    if job["worker"] == "sequence":
        result["common_target_findings"] = {name: {
            "ranking": ch["ranking_check"],
            "summaries": {a: r["common_summary"] for a, r in ch["archs"].items()}}
            for name, ch in actual["channels"].items()}
        published = read_json(evidence / "references" / refs[1])["table_6_7"]
        result["published_common_target_summaries"] = published
    return result


def render_report(run_dir, state):
    lines = ["# Round-2 reproduction evidence", "", f"Run: `{state['run_id']}`; source: `{state['source_commit']}`.",
             f"Status: **{state['status']}**; updated {state['updated_at']}.", "",
             "Original published files are preserved. Each job has reference snapshots/hashes, output hashes, commands, logs and a comparison. Training differences require review; an unchanged ranking is not equivalence.", "",
             "| Job | Status | Scope | Evidence |", "|---|---|---|---|"]
    for j in state["jobs"]:
        p = j["id"]
        lines.append(f"| {p} | {j['status']} | {j['scope']} | [record]({p}/job.json), [log]({p}/run.log) |")
    lines += ["", "## Limits", "",
              "Canonical learned checkpoints and historical PyTorch RNG state are absent. New seeded canonical training is protocol replication, not recovery of the original weights. The suite reanalyses the complete saved minimum-size grid but does not retrain that entire grid. Unknown-channel adaptive rare-event tails, pilot studies, full-window/bidirectional Mamba variants and other secondary studies are outside this initial suite. Matched Mamba remains unevaluated.",
              "", "The frozen source, manifest and environment define this run. No numerical result is replaced automatically. Failures, partial outputs and timeouts remain evidence.", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def checkpoint(run_dir, state):
    state["updated_at"] = utc()
    write_json(run_dir / "status.json", state)
    render_report(run_dir, state)


def environment():
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "torch", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "versions": versions,
            "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}


def harvest(work, evidence, job):
    paths = set(job["outputs"])
    paths.update(git("diff", "--name-only", cwd=work).splitlines())
    paths.update(git("ls-files", "--others", "--exclude-standard", cwd=work).splitlines())
    # Existing repo rules ignore weights/; these newly generated checkpoints
    # are required evidence and must be harvested despite that ignore rule.
    for folder in ("weights", "reproduction_checkpoints"):
        paths.update(str(p.relative_to(work)) for p in (work / folder).rglob("*") if p.is_file())
    files = []
    for rel in sorted(paths):
        p = work / rel
        if not p.is_file() or not rel.startswith(("results/", "weights/", "reproduction_checkpoints/")):
            continue
        dest = evidence / "outputs" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        files.append({"path": str(dest.relative_to(evidence)), "sha256": digest(dest), "bytes": dest.stat().st_size})
    return files


def run_job(job, run_dir, state, args):
    evidence = run_dir / job["id"]
    evidence.mkdir()
    work = ROOT / "_reproduction_work" / state["run_id"] / job["id"]
    rec = {**job, "status": "RUNNING", "started_at": utc(), "reference_files": [],
           "source_commit": state["source_commit"], "timeout_seconds": args.timeout_hours * 3600}
    for ref in job.get("references", job["outputs"]):
        dest = evidence / "references" / ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / ref, dest)
        rec["reference_files"].append({"path": ref, "sha256": digest(dest),
            "url": f"https://github.com/Gilzuk/relaynet2/blob/{state['source_commit']}/{ref}"})
    write_json(evidence / "job.json", rec)
    worker = [sys.executable, "-u", "scripts/reproduction_worker.py", "--job", job["id"], "--threads", str(args.threads)]
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
           "MKL_NUM_THREADS": str(args.threads), "NUMEXPR_NUM_THREADS": str(args.threads),
           "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", **job.get("env", {})}
    rec["command"] = worker
    rec["environment_overrides"] = {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "MPLBACKEND", "CUDA_VISIBLE_DEVICES", *job.get("env", {}))}
    proc = None
    created_worktree = False
    start = time.monotonic()
    try:
        with (evidence / "run.log").open("w") as log:
            if work.exists():
                raise FileExistsError(f"Refusing to reuse or remove an existing worktree: {work}")
            subprocess.run(["git", "worktree", "add", "--detach", "--no-checkout", str(work), state["source_commit"]],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            created_worktree = True
            subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "/*.py", "/.gitignore",
                            "/requirements*", "/relaynet/", "/checkpoints/", "/scripts/", "/results/",
                            "/e6_unknown_channel_results/", "/e6_unknown_channel_qpsk_results/",
                            "!*.png", "!*.pdf"], cwd=work, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=300)
            # --no-checkout leaves an empty index. Populate it only after the
            # sparse paths are configured; otherwise no worker exists to run.
            subprocess.run(["git", "reset", "--hard", state["source_commit"]], cwd=work,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            # Remove the exact target before execution: a stale output cannot
            # pass a failed/no-op run. Caches in the isolated checkout are removed.
            for rel in job["outputs"]:
                (work / rel).unlink(missing_ok=True)
            if (work / "weights").exists():
                shutil.rmtree(work / "weights")
            proc = subprocess.Popen(worker, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rec["pid"] = proc.pid
            write_json(evidence / "job.json", rec)
            while proc.poll() is None:
                rec["elapsed_seconds"] = round(time.monotonic() - start, 1)
                if rec["elapsed_seconds"] > args.timeout_hours * 3600:
                    rec["status"] = "TIMEOUT"
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                write_json(evidence / "job.json", rec)
                checkpoint(run_dir, state)
                time.sleep(10)
            rec["returncode"] = proc.wait()
        rec["artifacts"] = harvest(work, evidence, job)
        if rec["status"] != "TIMEOUT":
            if rec["returncode"] != 0:
                rec["status"] = "FAILED"
            elif not all((evidence / "outputs" / p).is_file() for p in job["outputs"]):
                rec["status"] = "MISSING_OUTPUT"
            else:
                comparison = compare(job, evidence)
                write_json(evidence / "comparison.json", comparison)
                rec["status"] = comparison["status"]
                rec["difference_count"] = comparison["difference_count"]
        changed = [r["path"] for r in rec["reference_files"] if digest(ROOT / r["path"]) != r["sha256"]]
        if changed:
            rec["status"], rec["changed_published_files"] = "REFERENCE_CHANGED", changed
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        rec["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        rec["exception"] = f"{type(exc).__name__}: {exc}"
        if created_worktree and work.exists():
            rec["artifacts"] = harvest(work, evidence, job)
    finally:
        rec["finished_at"] = utc()
        rec["elapsed_seconds"] = round(time.monotonic() - start, 1)
        rec["log_sha256"] = digest(evidence / "run.log") if (evidence / "run.log").exists() else None
        write_json(evidence / "job.json", rec)
        # Only remove this runner's disposable checkout after evidence is copied.
        if created_worktree and work.exists() and rec.get("artifacts"):
            subprocess.run(["git", "worktree", "remove", "--force", str(work)], cwd=ROOT, check=False)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--jobs", nargs="+", choices=[j["id"] for j in jobs()])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--timeout-hours", type=float, default=8)
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(json.dumps(jobs(), indent=2)); return 0
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        ap.error("run-id may contain only letters, digits, underscores and hyphens")
    if args.threads < 1 or args.timeout_hours <= 0:
        ap.error("threads and timeout-hours must be positive")
    run_dir = ROOT / "results" / "reproduction" / args.run_id
    if run_dir.exists():
        ap.error(f"refusing to overwrite existing evidence: {run_dir}")
    if args.background:
        logdir = ROOT / "results" / "reproduction"
        logdir.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                "--threads", str(args.threads), "--timeout-hours", str(args.timeout_hours)]
        if args.jobs:
            argv += ["--jobs", *args.jobs]
        with (logdir / (args.run_id + ".launcher.log")).open("x") as log:
            p = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": p.pid, "run_id": args.run_id, "status_file": str(run_dir / "status.json")}))
        return 0
    source = git("rev-parse", "HEAD")
    # The executable driver must be committed; evidence may be uncommitted.
    if git("diff", "HEAD", "--", "scripts/reproduce_round2.py", "scripts/reproduction_worker.py"):
        ap.error("commit the reproduction code before freezing the run")
    run_dir.mkdir(parents=True)
    selected = [j for j in jobs() if not args.jobs or j["id"] in args.jobs]
    state = {"run_id": args.run_id, "source_commit": source, "branch": git("branch", "--show-current"),
             "pid": os.getpid(), "started_at": utc(), "status": "RUNNING", "jobs": [dict(j, status="PENDING") for j in selected]}
    write_json(run_dir / "environment.json", environment())
    write_json(run_dir / "manifest.json", {"source_commit": source, "jobs": selected,
               "training_difference_policy": "flag and investigate; never tune to match or infer equivalence",
               "timeout_hours_per_job": args.timeout_hours, "threads_per_worker": args.threads})
    checkpoint(run_dir, state)
    for i, job in enumerate(selected):
        state["jobs"][i]["status"] = "RUNNING"
        checkpoint(run_dir, state)
        rec = run_job(job, run_dir, state, args)
        state["jobs"][i].update({k: rec[k] for k in ("status", "started_at", "finished_at", "elapsed_seconds")})
        checkpoint(run_dir, state)
        print(f"{utc()} {job['id']}: {rec['status']}", flush=True)
        if rec["status"] in ("INTERRUPTED", "REFERENCE_CHANGED"):
            break
    state["status"] = "COMPLETED" if all(j["status"] == "MATCH" for j in state["jobs"]) else "REVIEW_REQUIRED"
    state["finished_at"] = utc()
    checkpoint(run_dir, state)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
