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
and whether the step needs PyTorch. With ``--run`` each step's console output
goes to ``results/reproduce_logs/<step>.log``; a failing step is reported and
the rest continue; ``verify_thesis_tables.py`` runs at the end and
``git diff --stat`` shows which committed results changed.

What a full re-run can and cannot show. Seeded experiments should reproduce
their committed files. The learned columns of Tables 5.2 and 5.3 cannot: the
canonical runs never seeded PyTorch's weight initialization (Sec. 4.3.3), so
those cells will move by training variance and the verifier will flag them.
A full run takes many hours on a CPU; ``--only``/``--skip`` select steps.

Usage:
    python scripts/reproduce_all.py                 # print the plan (dry run)
    python scripts/reproduce_all.py --only coded    # plan for matching steps
    python scripts/reproduce_all.py --run --no-torch
    python scripts/reproduce_all.py --run --yes     # everything, dirty tree allowed
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

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
LOG_DIR = os.path.join(ROOT, "results", "reproduce_logs")


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
        matched = any(o.endswith("_matched.json") for o in outputs) and "matched" not in script
        env = dict(MATCHED_ENV) if matched else {}
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


def run(plan):
    os.makedirs(LOG_DIR, exist_ok=True)
    summary = []
    for k, s in enumerate(plan, 1):
        log = os.path.join(LOG_DIR, slug(s["name"])[:80] + ".log")
        print(f"[{k}/{len(plan)}] {s['name']} ... ", end="", flush=True)
        t0 = time.time()
        with open(log, "w") as fh:
            fh.write(f"$ {command(s)}\n")
            fh.flush()
            r = subprocess.run([sys.executable, s["script"]] + s["args"], cwd=ROOT,
                               env={**os.environ, **s["env"]}, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        print(f"{'ok' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode}  {dt:.0f} s")
        summary.append({"step": s["name"], "command": command(s), "returncode": r.returncode,
                        "seconds": round(dt, 1), "log": os.path.relpath(log, ROOT)})
    with open(os.path.join(LOG_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="execute the plan (default: print it only)")
    ap.add_argument("--only", help="regex: keep steps whose name or script matches")
    ap.add_argument("--skip", help="regex: drop steps whose name or script matches")
    ap.add_argument("--no-torch", action="store_true", help="drop steps that need PyTorch")
    ap.add_argument("--yes", action="store_true", help="run even with uncommitted changes in the tree")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = ap.parse_args()
    plan = select(build_plan(), a.only, a.skip, a.no_torch)
    if a.json:
        print(json.dumps(plan, indent=1))
        return 0
    if not a.run:
        print_plan(plan)
        return 0
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty and not a.yes:
        print("The working tree has uncommitted changes, and these steps overwrite committed result files.\n"
              "Commit or stash first, or pass --yes.")
        return 2
    summary = run(plan)
    failed = [s for s in summary if s["returncode"] != 0]
    print(f"\n{len(summary) - len(failed)} of {len(summary)} steps succeeded; logs in "
          f"{os.path.relpath(LOG_DIR, ROOT)}/.")
    print("\nVerifying the thesis against the regenerated results "
          "(the unseeded canonical learned columns are expected to move):")
    subprocess.run([sys.executable, "verify_thesis_tables.py"], cwd=ROOT)
    diff = subprocess.run(["git", "diff", "--stat", "--", "results", "e6_unknown_channel_results"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("\nCommitted result files that changed:\n" + (diff or "none: every regenerated file is identical"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
