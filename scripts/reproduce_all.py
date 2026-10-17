"""Re-run every experiment behind the thesis, in dependency order, then verify.

The plan is built from two sources:

* ``provenance_audit.REGISTRY`` -- every registered experiment, its script and
  the result files it writes;
* ``EXTRA`` below -- producers the registry does not list: the two canonical
  ``run_experiments.py`` stages (Tables 5.2 and 5.3) and the original,
  unweighted coded studies and three matched reruns of Chapter 5.

Matched coded reruns run with ``CODED_METRIC=matched CODED_TRIALS=30``, the
setting the thesis reports (30 trials on the published seeds). Order is derived,
not hand-kept: a step that reads another step's result file (its name appears in
the reading script's source) runs after it; otherwise the declared order holds.

Nothing runs unless ``--run`` is given. Without it the script prints the plan:
each step's command, environment, outputs, whether those outputs are committed,
and whether the step needs PyTorch.

``--run`` never touches the published results. It checks out the current commit
into a separate, detached git worktree and runs every step there, in order, so a
later step reads what an earlier step regenerated. Evidence goes to
``results/reproduce_all/<run-id>/``: the plan and environment, one log per step,
a copy of every declared output, and ``comparison.json``, which compares each
regenerated file with the committed one by SHA-256 (MATCH, DIFFERS, MISSING or
NEW). A difference is recorded for review, never accepted under a tolerance.
``verify_thesis_tables.py`` then runs inside the worktree, against the
regenerated data, and the worktree is removed unless ``--keep-worktree``.
Uncommitted changes in the main checkout are not part of a run.

For the round-2 experiments, ``scripts/reproduce_round2.py`` (PR #184) is the
more detailed runner: frozen-source jobs, reference hashes and per-field
comparisons, run in GitHub Actions. This script covers every registered
experiment, including those outside that suite.

What a full re-run can and cannot show. Seeded experiments should reproduce
their committed files. The learned columns of Tables 5.2 and 5.3 cannot: the
canonical runs never seeded PyTorch's weight initialization (Sec. 4.3.3), so
those cells will move by training variance and the verifier will flag them.
A full run takes many hours on a CPU; ``--only``/``--skip`` select steps.

Usage:
    python scripts/reproduce_all.py                 # print the plan (dry run)
    python scripts/reproduce_all.py --only coded    # plan for matching steps
    python scripts/reproduce_all.py --run --no-torch
    python scripts/reproduce_all.py --run           # everything, in a worktree
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CANON_BUDGET = ["--num-trials", "10", "--bits-per-trial", "10000", "--snr-min", "0",
                "--snr-max", "20", "--snr-step", "2", "--skip-relays", "cgan", "--retrain"]
MATCHED_ENV = {"CODED_METRIC": "matched", "CODED_TRIALS": "30"}

# name -> (script, args, env, outputs); declared order is the fallback order.
EXTRA = [
    ("Canonical comparison (Table 5.2)", "run_experiments.py", ["--exp", "7.10"] + CANON_BUDGET, {},
     ["results/modulation/qpsk_rayleigh.json"]),
    ("Equal-budget comparison (Table 5.3)", "run_experiments.py", ["--exp", "7.8"] + CANON_BUDGET, {},
     ["results/normalized_3k/3k_rayleigh.json"]),
    ("Coded block DF, K sweep (unweighted)", "coded_df_experiment.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded K sweep, QPSK (unweighted)", "coded_k_sweep_qpsk.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded K sweep, 16-QAM (unweighted)", "coded_k_sweep_qam16.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded learned MLP relay (unweighted)", "coded_learned_relay.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded Mamba relay (unweighted)", "coded_mamba_relay.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded high-SNR paired re-test", "coded_high_budget_test.py", [], {}, ["results/coded_high_budget_test.json"]),
    ("Coded error location (unweighted)", "coded_error_mechanism.py", [], {}, ["results/coded_error_mechanism.json"]),
    ("Coded reliable regime (unweighted)", "coded_reliable_regime.py", [], {}, ["results/coded_reliable_regime.json"]),
    ("Coded soft-decision relaying", "coded_soft_decision.py", [], {}, ["results/coded_soft_decision.json"]),
    ("Coded equal throughput (unweighted)", "coded_latency_throughput.py", [], {}, ["results/coded_latency_throughput.json"]),
    ("Coded rate adaptation (unweighted)", "coded_rate_adaptation.py", [], {}, ["results/coded_rate_adaptation.json"]),
    ("Coded readiness budget (unweighted)", "coded_latency_capacity.py", [], {}, ["results/coded_latency_capacity.json"]),
    ("Coded K sweep, QPSK, matched metrics", "coded_k_sweep_qpsk.py", [], MATCHED_ENV,
     ["results/coded_df_experiment_matched.json"]),
    ("Coded equal throughput, matched metrics", "coded_latency_throughput.py", [], MATCHED_ENV,
     ["results/coded_latency_throughput_matched.json"]),
    ("Coded readiness budget, matched metrics", "coded_latency_capacity.py", [], MATCHED_ENV,
     ["results/coded_latency_capacity_matched.json"]),
]
TORCH_HINTS = ("torch", "run_experiments", "checkpoint_", "relaynet.relays.genai", "build_all_3k")
EVIDENCE_ROOT = os.path.join(ROOT, "results", "reproduce_all")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def source(script):
    with open(os.path.join(ROOT, script), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def build_plan():
    """Steps in run order: dicts with name, script, args, env, outputs, needs_torch, after."""
    from provenance_audit import REGISTRY
    steps = [dict(name=n, script=s, args=a, env=e, outputs=o) for n, s, a, e, o in EXTRA]
    for name, (script, outputs, _art) in REGISTRY.items():
        env = {}
        if any(o.endswith("_matched.json") for o in outputs):
            if "matched" not in script:      # a published script rerun with the matched metric
                env = dict(MATCHED_ENV)
            elif "CODED_METRIC" in source(script):   # a matched-only script that insists on the flag
                env = {"CODED_METRIC": "matched"}
        dup = [s for s in steps if s["script"] == script and s["args"] == [] and s["env"] == env]
        if dup:  # same command already planned: merge outputs instead of running twice
            dup[0]["outputs"] = sorted(set(dup[0]["outputs"]) | set(outputs))
            dup[0]["name"] += f"; {name}"
            continue
        steps.append(dict(name=name, script=script, args=[], env=env, outputs=list(outputs)))
    for s in steps:
        src = source(s["script"])
        s["needs_torch"] = any(h in src for h in TORCH_HINTS)
        s["_src"] = src
    # A step depends on every other step that writes a file whose name its source
    # mentions, unless it writes that file itself (the merge-into-one-file pattern).
    for i, s in enumerate(steps):
        own = set(s["outputs"])
        s["after"] = sorted({j for j, t in enumerate(steps) if j != i
                             for o in t["outputs"] if o not in own and os.path.basename(o) in s["_src"]})
    order, done, pending = [], set(), list(range(len(steps)))
    while pending:
        ready = [i for i in pending if all(j in done for j in steps[i]["after"])]
        if not ready:
            raise RuntimeError("dependency cycle among: " + ", ".join(steps[i]["name"] for i in pending))
        i = ready[0]
        order.append(i); done.add(i); pending.remove(i)
    plan = []
    for i in order:
        s = dict(steps[i]); s.pop("_src")
        s["after"] = [steps[j]["name"] for j in steps[i]["after"]]
        plan.append(s)
    return plan


def committed(path):
    r = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short", "--", path],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "not committed"


def command(s):
    env = " ".join(f"{k}={v}" for k, v in s["env"].items())
    return (env + " " if env else "") + " ".join(["python3", s["script"]] + s["args"])


def select(plan, only, skip, no_torch):
    keep = []
    for s in plan:
        text = f"{s['name']} {s['script']}"
        if only and not re.search(only, text, re.I):
            continue
        if skip and re.search(skip, text, re.I):
            continue
        if no_torch and s["needs_torch"]:
            continue
        keep.append(s)
    return keep


def print_plan(plan):
    for k, s in enumerate(plan, 1):
        print(f"[{k:2d}] {s['name']}{'   (needs PyTorch)' if s['needs_torch'] else ''}")
        print(f"     $ {command(s)}")
        for o in s["outputs"]:
            print(f"       -> {o}   [{committed(o)}]")
        if s["after"]:
            print(f"       after: {'; '.join(s['after'])}")
    print(f"\n{len(plan)} steps, {sum(s['needs_torch'] for s in plan)} need PyTorch. "
          "Nothing was run; add --run to execute.")


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def committed_sha256(source, path):
    """SHA-256 of `path` as committed at `source`, or None if it is not tracked there."""
    r = subprocess.run(["git", "show", f"{source}:{path}"], cwd=ROOT, capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def compare(source, worktree, path, outdir):
    """Copy a regenerated output into the evidence folder and compare it with the committed file."""
    regen = os.path.join(worktree, path)
    ref = committed_sha256(source, path)
    if not os.path.exists(regen):
        return {"path": path, "status": "MISSING", "committed_sha256": ref}
    dst = os.path.join(outdir, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(regen, dst)
    new = sha256_file(regen)
    status = "NEW" if ref is None else ("MATCH" if new == ref else "DIFFERS")
    return {"path": path, "status": status, "committed_sha256": ref, "regenerated_sha256": new}


def run(plan, evidence_dir, worktree_dir=None, keep_worktree=False, verify=True):
    """Run the plan in a detached worktree; write evidence under `evidence_dir`. Returns the summary."""
    source = git("rev-parse", "HEAD")
    os.makedirs(evidence_dir, exist_ok=True)
    logs, outdir = os.path.join(evidence_dir, "logs"), os.path.join(evidence_dir, "outputs")
    os.makedirs(logs, exist_ok=True)
    wt = worktree_dir or tempfile.mkdtemp(prefix="reproduce_all_")
    if os.path.exists(wt) and os.listdir(wt):
        raise SystemExit(f"worktree directory {wt} is not empty")
    git("worktree", "add", "--detach", wt, source)
    meta = {"source_commit": source, "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "platform": platform.platform(), "worktree": wt,
            "uncommitted_changes_excluded": bool(git("status", "--porcelain")), "steps": plan}
    with open(os.path.join(evidence_dir, "plan.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    summary, comparison = [], []
    try:
        for k, s in enumerate(plan, 1):
            log = os.path.join(logs, slug(s["name"])[:80] + ".log")
            print(f"[{k}/{len(plan)}] {s['name']} ... ", end="", flush=True)
            t0 = time.time()
            with open(log, "w") as fh:
                fh.write(f"$ {command(s)}\n")
                fh.flush()
                r = subprocess.run([sys.executable, s["script"]] + s["args"], cwd=wt,
                                   env={**os.environ, **s["env"]}, stdout=fh, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            cmp = [compare(source, wt, o, outdir) for o in s["outputs"]]
            comparison += [dict(c, step=s["name"]) for c in cmp]
            tally = ", ".join(f"{c['status']} {os.path.basename(c['path'])}" for c in cmp)
            print(f"{'ok' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode}  {dt:.0f} s  [{tally}]")
            summary.append({"step": s["name"], "command": command(s), "returncode": r.returncode,
                            "seconds": round(dt, 1), "log": os.path.relpath(log, evidence_dir)})
        if verify:
            print("\nVerifying the thesis against the regenerated results, inside the worktree "
                  "(the unseeded canonical learned columns are expected to move):")
            with open(os.path.join(evidence_dir, "verify.log"), "w") as fh:
                v = subprocess.run([sys.executable, "verify_thesis_tables.py"], cwd=wt,
                                   stdout=fh, stderr=subprocess.STDOUT)
            tail = open(os.path.join(evidence_dir, "verify.log")).read().strip().splitlines()
            print("\n".join(l for l in tail if "cells checked" in l or "MISMATCH" in l) or "(see verify.log)")
            meta["verify_returncode"] = v.returncode
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(evidence_dir, "plan.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        with open(os.path.join(evidence_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        with open(os.path.join(evidence_dir, "comparison.json"), "w") as fh:
            json.dump(comparison, fh, indent=1)
        if keep_worktree:
            print(f"\nWorktree kept at {wt}")
        else:
            subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT, capture_output=True)
    return summary, comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="execute the plan in a separate worktree (default: print it)")
    ap.add_argument("--only", help="regex: keep steps whose name or script matches")
    ap.add_argument("--skip", help="regex: drop steps whose name or script matches")
    ap.add_argument("--no-torch", action="store_true", help="drop steps that need PyTorch")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    ap.add_argument("--run-id", help="evidence folder name (default: UTC time and short commit)")
    ap.add_argument("--evidence-dir", help="where to write evidence (default: results/reproduce_all/<run-id>)")
    ap.add_argument("--worktree-dir", help="empty directory for the worktree (default: a new temporary one)")
    ap.add_argument("--keep-worktree", action="store_true", help="leave the worktree in place afterwards")
    ap.add_argument("--no-verify", action="store_true", help="skip the final verify_thesis_tables.py run")
    a = ap.parse_args()
    plan = select(build_plan(), a.only, a.skip, a.no_torch)
    if a.json:
        print(json.dumps(plan, indent=1))
        return 0
    if not a.run:
        print_plan(plan)
        return 0
    run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + git("rev-parse", "--short", "HEAD")
    evidence = a.evidence_dir or os.path.join(EVIDENCE_ROOT, run_id)
    if git("status", "--porcelain"):
        print("Note: uncommitted changes are not part of this run; it uses the committed HEAD.")
    summary, comparison = run(plan, evidence, a.worktree_dir, a.keep_worktree, verify=not a.no_verify)
    failed = [s for s in summary if s["returncode"] != 0]
    counts = {}
    for c in comparison:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    print(f"\n{len(summary) - len(failed)} of {len(summary)} steps succeeded. Outputs: "
          + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none")
          + f".\nEvidence: {os.path.relpath(evidence, ROOT) if evidence.startswith(ROOT) else evidence}/ "
          "(published results untouched).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
