#!/usr/bin/env python3
"""Check the first MLP replay against the original driver's unchanged loop.

Run from a disposable checkout, never the main working tree. The control
selects the first architecture/channel only, retaining original call order,
training/evaluation budgets and all three seeds. It tests whether the new
checkpoint/evidence instrumentation changed the reproduced BER arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--replay-snapshot", required=True, type=Path)
    args = ap.parse_args()
    work = Path.cwd().resolve()
    if "_reproduction_work" not in work.parts or not (work / ".git").is_file():
        ap.error("control must run in a disposable worktree under _reproduction_work")
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, evidence / "control_driver.py")
    sys.path.insert(0, str(work))
    import torch
    import numpy as np
    from scripts.reproduce_round2 import differences, write_json, git
    import seq_models_on_memory as original
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    published = (work / "results/seq_models_on_memory.json").read_bytes()
    (evidence / "published.json").write_bytes(published)
    replay_bytes = args.replay_snapshot.read_bytes()
    (evidence / "replay_snapshot.json").write_bytes(replay_bytes)
    original.ARCHS = [original.ARCHS[0]]
    assert original.ARCHS[0][0] == "MLP-3K"
    sys.argv = ["seq_models_on_memory.py", "isi"]
    original.main()
    raw = (work / "results/seq_models_on_memory.json").read_bytes()
    (evidence / "original_driver_rerun.json").write_bytes(raw)

    def projection(d):
        ch = d["channels"]["isi"]
        return {"baseline_ber": ch["baseline_ber"], "runs": [
            {"seed": r["seed"], "ber": r["ber"]} for r in ch["archs"]["MLP-3K"]["runs"]]}

    a = projection(json.loads(raw))
    changes = differences(projection(json.loads(replay_bytes)), a)
    pub_changes = differences(projection(json.loads(published)), a)
    out = {"status": "INSTRUMENTATION_CONTROL_MATCH" if not changes else "INSTRUMENTATION_CONTROL_MISMATCH",
           "completed_at": datetime.now(timezone.utc).isoformat(), "source_commit": git("rev-parse", "HEAD"),
           "control_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "original_driver_git_blob": git("rev-parse", "HEAD:seq_models_on_memory.py"),
           "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
           "threads": torch.get_num_threads(), "threads_env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
           "protocol": "Original driver, only ARCHS reduced to its first MLP entry and channel restricted to isi; full three-seed budgets unchanged.",
           "differences_vs_instrumented_replay": changes, "differences_vs_published": pub_changes,
           "interpretation": "Agreement rules out the new instrumentation as the source of this MLP BER difference in this environment. It does not identify the historical software, thread setting or other cause, nor prove equivalence of training populations.",
           "artifacts": [{"path": n, "sha256": hashlib.sha256((evidence / n).read_bytes()).hexdigest()}
                         for n in ("control_driver.py", "published.json", "replay_snapshot.json", "original_driver_rerun.json")]}
    write_json(evidence / "comparison.json", out)
    print(out["status"], len(changes), "differences vs replay;", len(pub_changes), "vs published", flush=True)
    return 0 if not changes else 1


if __name__ == "__main__":
    raise SystemExit(main())
