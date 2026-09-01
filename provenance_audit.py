#!/usr/bin/env python3
"""Link every experiment to its script, its data, and the commit that made it.

Nothing in this repository recorded which script produced which result file,
when, or which published table rests on it. That gap is what allowed a thesis
table to be edited away from its data and stay wrong for four days, and what
later let a correction be made *towards* data that was itself older than the
script that generated it.

The ledger is generated rather than written down, because a hand-maintained
one would go stale the same way. REGISTRY below is the only hand-kept part:
it declares experiment -> script -> outputs -> published artefacts. Everything
else -- commit, author, date, staleness -- is read from git at run time.

The check that matters is STALE: an output whose producing commit is older
than the last change to the script that produces it. That is the condition
that was silently true for e6_sim_ported_results.npy for ten days.

    python provenance_audit.py            # ledger + warnings
    python provenance_audit.py --markdown # ledger for memory-bank/
"""

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# experiment -> (script, [output files], [published tables/figures])
REGISTRY = {
    "QPSK error decomposition": (
        "qpsk_error_decomposition.py", ["results/qpsk_error_decomposition.json"],
        ["prose: QPSK SER/BER and bits-per-symbol-error"]),
    "ISI slicer floor, closed form": (
        "isi_slicer_floor.py", ["results/isi_slicer_floor.json"],
        ["eq:slicer-floor", "prose: closed-form slicer BER table"]),
    "E6 unknown ISI (S1-S4)": (
        "e6_sim_ported.py", ["e6_unknown_channel_results/e6_sim_ported_results.npy"],
        ["tbl:tableE6"]),
    "E6 flat control": (
        "e6_flat_ported.py", ["e6_unknown_channel_results/e6_flat_ported_results.npy"],
        ["tbl:tableE6flat"]),
    "E6 composite cascade": (
        "e6_composite_ported.py",
        ["e6_unknown_channel_results/e6_composite_ported_results.npy"],
        ["fig:figE6composite", "prose:E6composite"]),
    "E6 pilot-budget sweep": (
        "e6_partial_ported.py",
        ["e6_unknown_channel_results/e6_partial_ported_results.npy"],
        ["fig:e6-partial", "prose:E6partial"]),
    "E6 blind / posterior-free": (
        "e6_blind_ported.py", ["e6_unknown_channel_results/e6_blind_ported_results.npy"],
        ["fig:figE6blind", "prose:E6blind"]),
    "E6 QPSK unknown channel": (
        "e6_qpsk_unknown_channel.py",
        ["e6_unknown_channel_results/e6_qpsk_unknown_channel_results.npy"],
        ["tbl:tableE6qpsk"]),
    "Minimum relay size, 9 channels": (
        "mlp_min_size_all_channels.py", ["results/mlp_min_size_all_channels.json"],
        ["tbl:table-minsize", "fig:minsize-crossover", "fig:minsize-budget"]),
    "Minimum size, window x depth": (
        "mlp_min_size_bisect.py", ["results/mlp_min_size_bisect.json"],
        ["prose: depth 1-3, window 1-7"]),
    "Coded minimum size": (
        "coded_min_size.py", ["results/coded_min_size.json"], ["prose: coded row"]),
    "Seed spread, equal budget": (
        "seed_spread_architectures.py", ["results/seed_spread_architectures.json"],
        ["tbl:seed-spread", "tbl:seed-spread-3k"]),
    "Transformer instability": (
        "transformer_instability.py", ["results/transformer_instability.json"],
        ["fig:transformer-seed-curves", "fig:transformer-loss-penalty"]),
    "MMSE complexity-matched baseline": (
        "mmse_equalizer.py", ["results/mmse_equalizer.json"], ["tbl:mmse-baseline"]),
    "Sequence models on memory": (
        "seq_models_on_memory.py", ["results/seq_models_on_memory.json"],
        ["tbl:seq-on-memory"]),
    "Joint latency/memory": (
        "joint_latency_memory.py", ["results/joint_latency_memory.json"],
        ["tbl:joint-latency"]),
    "Memory sweep, precision re-run": (
        "joint_memory_precision.py", ["results/joint_memory_precision.json"],
        ["tbl:joint-memory"]),
    "MAC accounting": (
        "unified_latency_axis.py", ["results/unified_latency_axis.json"],
        ["eq:mac-crossover"]),
}


PARAM_KEYS = ("SEED", "SEEDS", "N_TRAIN", "N_TRIALS", "N_BITS", "N_FRAMES",
              "FRAME_INFO_BITS", "SNRS", "TRAIN_SNRS", "MLP_HIDDEN", "N_TRAIN_BITS",
              "MODULATION", "W", "H_ISI", "MC_TRIALS", "EPOCHS")


def params_of(script):
    """Module-level constants a rerun would need, read from the script itself.

    Extracted rather than transcribed: a hand-copied seed is one more thing
    that can silently stop matching the code it claims to describe.
    """
    try:
        src = io.open(os.path.join(ROOT, script), encoding="utf-8").read()
    except IOError:
        return {}
    out = {}
    for k in PARAM_KEYS:
        m = re.search(r"^%s\s*=\s*([^\n#]+)" % re.escape(k), src, re.M)
        if m:
            out[k] = m.group(1).strip().rstrip(",")
    return out


def last_commit(path):
    """(sha, iso-date, author, subject, iso-timestamp) for path's last commit.

    Both a short date (for display) and a full timestamp (for the staleness
    comparison). Comparing on %cs alone made the check blind to a script edited
    later the same day as the run it describes -- which is precisely how a
    metadata fix to e6_sim_ported.py slipped past this tool while it printed
    "ok".
    """
    if not os.path.exists(os.path.join(ROOT, path)):
        return None
    out = subprocess.run(
        ["git", "log", "-1", "--format=%h\t%cs\t%an\t%s\t%cI", "--", path],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    parts = out.split("\t")
    # subject may itself contain tabs; timestamp is the last field
    sha, date, author, ts = parts[0], parts[1], parts[2], parts[-1]
    subject = "\t".join(parts[3:-1])
    return [sha, date, author, subject, ts]


# (script, output) pairs where the script is newer than its data on purpose,
# with the reason. Only for changes that cannot alter a simulated value --
# persisted metadata, comments, logging. A change to the simulation itself is
# never allowlisted; it is re-run.
REVIEWED_STALE = {
    ("e6_sim_ported.py", "e6_unknown_channel_results/e6_sim_ported_results.npy"):
        "metadata-only fix (3fc7f91): single writer for the .npy plus persisted "
        "rare_event_meta; no simulated value depends on it. That run's error "
        "counts are in results/e6_sim_rerun_progress.txt.",
    ("mlp_min_size_all_channels.py", "results/mlp_min_size_all_channels.json"):
        "comment correction plus a display-name change to the isi_rayleigh "
        "comparator ('MLSE' -> 'MLSE (taps only)'). Same relay object, same "
        "numbers; only the JSON's `baseline` label would differ on a re-run.",
    ("seq_models_on_memory.py", "results/seq_models_on_memory.json"):
        "6048c95 touched only main()'s console reporting -- a NaN guard around "
        "min() over architectures that reached no target. Every value written "
        "to the JSON is computed before that code runs.",
}


def audit():
    rows, warn = [], []
    for name, (script, outputs, artefacts) in sorted(REGISTRY.items()):
        s = last_commit(script)
        for out in outputs:
            o = last_commit(out)
            status = "ok"
            if s is None:
                status = "SCRIPT MISSING"
            elif o is None:
                status = "DATA UNCOMMITTED"
            elif o[4] < s[4]:
                reason = REVIEWED_STALE.get((script, out))
                status = (f"stale, reviewed: {reason}" if reason
                          else "STALE (data older than script)")
            if status != "ok" and not status.startswith("stale, reviewed"):
                warn.append((name, status, script, out,
                             s[1] if s else "-", o[1] if o else "-"))
            rows.append({"experiment": name, "script": script, "output": out,
                         "script_commit": s, "data_commit": o,
                         "artefacts": artefacts, "status": status})
    return rows, warn


def main():
    rows, warn = audit()
    md = "--markdown" in sys.argv
    tables = "--tables" in sys.argv

    if tables:
        pass
    elif md:
        print("| Experiment | Script | Data | Produced by | Backs | Status |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            d = r["data_commit"]
            print("| %s | `%s` | `%s` | %s | %s | %s |" % (
                r["experiment"], r["script"], os.path.basename(r["output"]),
                ("`%s` %s" % (d[0], d[1])) if d else "**uncommitted**",
                ", ".join("`%s`" % a for a in r["artefacts"]),
                "ok" if r["status"] == "ok" else "**%s**" % r["status"]))
    elif not tables:
        for r in rows:
            d, s = r["data_commit"], r["script_commit"]
            print("%-34s %-30s %s" % (r["experiment"], os.path.basename(r["output"]),
                                      r["status"]))
            print("   script %-28s %s" % (r["script"], ("%s %s" % (s[0], s[1])) if s else "-"))
            print("   data   %-28s %s" % (os.path.basename(r["output"]),
                                          ("%s %s  [%s]" % (d[0], d[1], d[2])) if d else "UNCOMMITTED"))

    if tables:
        print("# Table and figure provenance\n")
        print("Generated by `provenance_audit.py --tables`. Every published table and")
        print("figure, the experiment behind it, the exact command that reproduces it,")
        print("the parameters that run used, and the commit that produced the data it")
        print("rests on. Do not hand-edit; re-run the script.\n")
        byart = {}
        for r in rows:
            for a in r["artefacts"]:
                byart.setdefault(a, r)
        for a in sorted(byart):
            r = byart[a]
            d = r["data_commit"]
            pr = params_of(r["script"])
            print("## `%s`\n" % a)
            print("| field | value |")
            print("|---|---|")
            print("| experiment | %s |" % r["experiment"])
            print("| reproduce | `python %s` |" % r["script"])
            print("| data file | `%s` |" % r["output"])
            print("| data commit | %s |" % (("`%s` %s (%s)" % (d[0], d[1], d[2]))
                                            if d else "**UNCOMMITTED**"))
            print("| provenance | %s |" % ("ok" if r["status"] == "ok"
                                           else "**%s**" % r["status"]))
            for k in sorted(pr):
                print("| %s | `%s` |" % (k, pr[k]))
            print()
        return 0

    if warn:
        print("\n%d PROVENANCE WARNING(S):" % len(warn))
        for name, status, script, out, sd, od in warn:
            print("  [%s] %s" % (status, name))
            print("      script %s last changed %s; data %s last committed %s"
                  % (script, sd, os.path.basename(out), od))
        return 1
    print("\nAll declared outputs are committed and no data predates its script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
