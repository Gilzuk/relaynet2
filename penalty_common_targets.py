"""SNR penalties on comparable target sets (thesis Tables 6.6, 6.7, 9.4-9.6).

Every single-number penalty in the thesis was the worst penalty over the BER
targets (1e-1, 1e-2, 1e-3) that *both* curves reach. A relay that never reaches
a target is then scored on the easier targets alone, so it can look better than
one that reaches them all: the Mamba-2 initialization reported at +4.69 dB on
the QPSK ISI channel never reaches 1e-3, while the other two, held to it, give
+5.83 and +6.80.

This script recomputes every such number from the committed per-seed BER curves
with the target set fixed by the reference curve (ber_metrics.penalty_common):
a target the reference reaches is kept for every relay, and a relay that misses
it is reported as a lower bound (">") rather than silently scored on fewer
targets. For each published quantity it stores the published-rule value (which
must reproduce the thesis), the comparable-set value, and which seeds miss
which targets. No experiment is rerun; only the stored curves are reread.

Output: results/penalty_common_targets.json
Usage:  python penalty_common_targets.py
"""
import json
import os
import subprocess

import numpy as np

from ber_metrics import penalty_table, penalty_common, best_of_seeds, spread_of_seeds

OUT = "results/penalty_common_targets.json"
BUDGETS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa
        return None


def load(name):
    return json.load(open(os.path.join("results", name)))


def old_worst(snr, ber, ref):
    p = penalty_table(snr, np.asarray(ber, float), np.asarray(ref, float))
    return p["worst_db_penalty"] if p["targets_reached"] else float("nan")


def new(snr, ber, ref):
    return penalty_common(snr, np.asarray(ber, float), np.asarray(ref, float))


def seeds_summary(snr, runs, ref):
    """Per-seed old and new penalties plus best and spread under both rules."""
    old = [old_worst(snr, r, ref) for r in runs]
    nw = [new(snr, r, ref) for r in runs]
    o = [v for v in old if v == v]
    best_n, best_f = best_of_seeds(nw)
    spr_n, spr_f = spread_of_seeds(nw)
    return {
        "old_per_seed": old,
        "new_per_seed": [r["worst_db"] for r in nw],
        "new_lower_bound": [r["lower_bound"] for r in nw],
        "unreached": [[str(t) for t in r["unreached"]] for r in nw],
        "common_targets": [str(t) for t in nw[0]["common_targets"]] if nw else [],
        "old_best": min(o) if o else float("nan"),
        "old_spread": (max(o) - min(o)) if len(o) > 1 else float("nan"),
        "new_best": best_n, "new_best_flag": best_f,
        "new_spread": spr_n, "new_spread_flag": spr_f,
        "new_median": float(np.median([r["worst_db"] for r in nw])) if nw else float("nan"),
        "new_median_flag": (">" if nw and sorted(nw, key=lambda r: r["worst_db"])[len(nw) // 2]["lower_bound"] else ""),
    }


def table_6_7():
    d = load("seq_models_on_memory.json")
    snr = d["snr_db"]
    out = {}
    for ch, c in d["channels"].items():
        out[ch] = {}
        for a, v in c["archs"].items():
            s = seeds_summary(snr, [r["ber"] for r in v["runs"]], c["baseline_ber"])
            s["published_best"], s["published_spread"] = v["best_db"], v["spread"]
            out[ch][a] = s
    return out


def table_6_6():
    ms = load("mlp_min_size_all_channels.json")
    det = load("mmse_equalizer_detail.json")
    snr = ms["snr_db"]
    out = {}
    for ch in ("isi", "isi_complex", "composite"):
        base = ms["channels"][ch]["baseline_ber"]
        row = {"equalizer": {}}
        for n, rec in det[ch].items():
            p = new(snr, rec["ber"], base)
            row["equalizer"][n] = {"published": rec["worst_db"], "old": old_worst(snr, rec["ber"], base),
                                   "new": p["worst_db"], "new_lower_bound": p["lower_bound"],
                                   "unreached": [str(t) for t in p["unreached"]]}
        best_old, best_new, cfg_new = None, None, None
        for cfg in ms["channels"][ch]["sweep"]:
            s = seeds_summary(snr, [r["ber"] for r in cfg["seed_runs"]], base)
            if s["old_best"] == s["old_best"] and (best_old is None or s["old_best"] < best_old[0]):
                best_old = (s["old_best"], cfg["params"], cfg["window"], float(np.median([v for v in s["old_per_seed"] if v == v])))
            if s["new_best_flag"] == "" and (best_new is None or s["new_best"] < best_new[0]):
                best_new = (s["new_best"], cfg["params"], cfg["window"], s["new_median"], s["new_median_flag"])
        row["learned_old"] = {"best": best_old[0], "params": best_old[1], "window": best_old[2], "median": best_old[3]}
        row["learned_new"] = {"best": best_new[0], "params": best_new[1], "window": best_new[2],
                              "median": best_new[3], "median_flag": best_new[4]}
        out[ch] = row
    return out


def minsize_rows(sweep, snr, ref_w, ref_h, base, rule):
    ref_row = [x for x in sweep if x["window"] == ref_w and x["hidden"] == ref_h][0]
    ref = np.mean([s["ber"] for s in ref_row["seed_runs"]], axis=0)
    rows = []
    for x in sweep:
        if x["params"] >= ref_row["params"]:
            continue
        worst = np.max([s["ber"] for s in x["seed_runs"]], axis=0)
        if rule == "old":
            p = penalty_table(snr, worst, ref)
            if not p["targets_reached"]:
                continue
            rows.append({"params": x["params"], "window": x["window"], "db": p["worst_db_penalty"], "lb": False})
        else:
            p = penalty_common(snr, worst, ref)
            if p["worst_db"] != p["worst_db"]:
                continue
            rows.append({"params": x["params"], "window": x["window"], "db": p["worst_db"],
                         "lb": p["lower_bound"], "unreached": [str(t) for t in p["unreached"]]})
    return rows, ref_row["params"]


def budget_cells(rows):
    cells = []
    for b in BUDGETS:
        ok = [z for z in rows if not z["lb"] and z["db"] <= b]
        cells.append(f"{min(ok, key=lambda q: q['params'])['params']}p w{min(ok, key=lambda q: q['params'])['window']}" if ok else "-")
    return cells


def table_9_4():
    ms = load("mlp_min_size_all_channels.json")
    out = {}
    srcs = [(n, r["sweep"], ms["snr_db"], 5, 24) for n, r in ms["channels"].items()]
    c = load("coded_min_size.json")
    srcs.append(("coded", c["sweep"], c["snr_db"], 21, 16))
    for name, sweep, snr, w, h in srcs:
        ro, refp = minsize_rows(sweep, snr, w, h, None, "old")
        rn, _ = minsize_rows(sweep, snr, w, h, None, "new")
        out[name] = {"reference_params": refp, "old_cells": budget_cells(ro), "new_cells": budget_cells(rn),
                     "configs_missing_targets": [{"params": z["params"], "window": z["window"], "unreached": z["unreached"]}
                                                 for z in rn if z["lb"]]}
    return out


def table_9_5():
    """Across-seed spread per configuration, summarized per channel.

    Published rule: per-seed worst penalty over the targets both curves reach,
    and only configurations whose three seeds all give a value (that is the
    selection that reproduces the published table). Comparable rule: every
    configuration is kept; a seed that misses a target the comparator reaches
    makes that configuration's spread a lower bound, and a channel statistic
    computed over any bounded spread is itself a lower bound.
    """
    ms = load("mlp_min_size_all_channels.json")
    snr = ms["snr_db"]
    out, all_old, all_new, all_flags = {}, [], [], []
    summ = lambda v: {"configs": len(v), "median": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)), "max": float(max(v))}

    def bounded_stats(v, f):
        """Which statistics a lower-bounded spread could still raise.

        Raising a value above an order statistic cannot move it, so a statistic
        is exact when every bounded spread lies strictly above the values it is
        interpolated from; otherwise it is itself a lower bound.
        """
        v, b = np.asarray(v), [x for x, g in zip(v, f) if g == ">"]
        srt = np.sort(v)
        def exact(q):
            pos = q / 100 * (len(v) - 1)
            lo_val = srt[int(np.ceil(pos))]
            return all(x > lo_val for x in b)
        return {"median_exact": exact(50), "p90_exact": exact(90), "max_exact": not b}
    for ch, c in ms["channels"].items():
        base = c.get("baseline_ber", c["df_ber"])
        old_sp, new_sp, flags, undefined = [], [], [], 0
        for cfg in c["sweep"]:
            s = seeds_summary(snr, [r["ber"] for r in cfg["seed_runs"]], base)
            if all(v == v for v in s["old_per_seed"]):
                old_sp.append(s["old_spread"])
            if s["new_spread"] == s["new_spread"]:
                new_sp.append(s["new_spread"]); flags.append(s["new_spread_flag"])
            else:
                undefined += 1
        out[ch] = {"old": summ(old_sp), "new": summ(new_sp), "new_exact": bounded_stats(new_sp, flags),
                   "new_bounded_configs": sum(f == ">" for f in flags), "new_undefined_configs": undefined}
        all_old += old_sp; all_new += new_sp; all_flags += flags
    out["All"] = {"old": summ(all_old), "new": summ(all_new), "new_exact": bounded_stats(all_new, all_flags),
                  "new_bounded_configs": sum(f == ">" for f in all_flags)}
    return out


def table_9_6():
    d = load("seed_spread_architectures.json")
    snr = d["snr_db"]
    out = {}
    for a, v in d["architectures"].items():
        s = seeds_summary(snr, [r["ber"] for r in v["runs"]], d["df_ber"])
        s["published_spread"] = v["spread_db"]
        out[a] = s
    return out


def native_spreads():
    d = load("seed_spread_native_architectures.json")
    snr = d["snr_db"]
    return {a: dict(seeds_summary(snr, [r["ber"] for r in v["runs"]], d["df_ber"]), published_spread=v["spread_db"])
            for a, v in d["architectures"].items()}


def transformer_instability():
    d = load("transformer_instability.json")
    snr = d["snr_db"]
    out = {}
    for a, runs in d["runs"].items():
        s = seeds_summary(snr, [r["ber"] for r in runs], d["df_ber"])
        s["published_per_seed"] = [r.get("db_penalty") for r in runs]
        s["final_loss"] = [r["final_loss"] for r in runs]
        out[a] = s
    t = out["Transformer-3K"]
    lo, db = np.array(t["final_loss"]), np.array(t["new_per_seed"])
    keep = lo <= 5 * np.median(lo)
    out["correlation_new"] = {"all": float(np.corrcoef(lo, db)[0, 1]), "converged": float(np.corrcoef(lo[keep], db[keep])[0, 1])}
    return out


def coded_minsize():
    out = {}
    for f in ("coded_min_size.json", "coded_min_size_matched.json"):
        c = load(f)
        snr, df = c["snr_db"], c["df_ber"]
        rows = []
        for cfg in c["sweep"]:
            s = seeds_summary(snr, [r["ber"] for r in cfg["seed_runs"]], df)
            rows.append({"params": cfg["params"], "window": cfg["window"], "old_best": s["old_best"],
                         "new_best": s["new_best"], "new_best_flag": s["new_best_flag"],
                         "old_per_seed": s["old_per_seed"], "new_per_seed": s["new_per_seed"],
                         "unreached": s["unreached"]})
        out[f] = rows
    return out


def window_crossover():
    """Best penalty at each relay window (Fig. 9.1), channels with memory."""
    ms = load("mlp_min_size_all_channels.json")
    snr = ms["snr_db"]
    out = {}
    for ch, c in ms["channels"].items():
        base = np.asarray(c.get("baseline_ber", c["df_ber"]), float)
        out[ch] = {"memory": c["memory"], "windows": {}}
        for w in sorted({x["window"] for x in c["sweep"]}):
            old, nw = [], []
            for x in c["sweep"]:
                if x["window"] != w:
                    continue
                for r in x["seed_runs"]:
                    o = old_worst(snr, r["ber"], base)
                    if o == o:
                        old.append(o)
                    nw.append(new(snr, r["ber"], base))
            b, f = best_of_seeds(nw)
            out[ch]["windows"][str(w)] = {"old": min(old) if old else float("nan"), "new": b, "new_flag": f}
        ws = out[ch]["windows"]
        if c["memory"] > 1:
            out[ch]["gain_w1_to_w7"] = {"value": ws["1"]["new"] - ws["7"]["new"],
                                        "flag": ">" if ws["1"]["new_flag"] == ">" else ""}
    return out


def seq_rerun():
    """Sec. 6.4.2: the rerun's best values against Table 6.7, both rules."""
    pub = load("seq_models_on_memory.json")
    ro, ctl = load("seq_models_on_memory_readout.json"), load("seq_models_on_memory_rerun_controls.json")
    snr = pub["snr_db"]
    rerun = {"MLP-3K": (ctl, "MLP-3K"), "Transformer-3K": (ctl, "Transformer-3K"),
             "Mamba-S6-3K": (ro, "Mamba-S6-3K|center"), "Mamba2-3K": (ro, "Mamba2-3K|center")}
    cells = {}
    for ch in ("isi", "isi_complex", "composite"):
        for a, (src, key) in rerun.items():
            p = seeds_summary(snr, [r["ber"] for r in pub["channels"][ch]["archs"][a]["runs"]], pub["channels"][ch]["baseline_ber"])
            q = seeds_summary(snr, [r["ber"] for r in src["channels"][ch]["archs"][key]["runs"]], src["channels"][ch]["baseline_ber"])
            cells[f"{ch}/{a}"] = {"published_new": p["new_best"], "rerun_new": q["new_best"],
                                  "diff_new": abs(p["new_best"] - q["new_best"]),
                                  "diff_old": abs(pub["channels"][ch]["archs"][a]["best_db"] - src["channels"][ch]["archs"][key]["best_db"]),
                                  "rerun_spread_new": q["new_spread"], "rerun_spread_flag": q["new_spread_flag"]}
    lim = 0.25
    return {"cells": cells, "limit_db": lim,
            "within_new": [k for k, v in cells.items() if v["diff_new"] <= lim + 0.005],
            "outside_new": [k for k, v in cells.items() if v["diff_new"] > lim + 0.005],
            "outside_old": [k for k, v in cells.items() if v["diff_old"] > lim + 0.005]}


def report(res):
    print("Table 6.7 (best of three, spread):")
    for ch, archs in res["table_6_7"].items():
        for a, s in archs.items():
            flag = "  <-- CHANGES" if abs(s["new_best"] - s["published_best"]) > 0.005 or s["new_best_flag"] or s["new_spread_flag"] else ""
            print(f"  {ch:12s} {a:15s} published {s['published_best']:+.2f} ({s['published_spread']:.2f})  "
                  f"common {s['new_best_flag']}{s['new_best']:+.2f} ({s['new_spread_flag']}{s['new_spread']:.2f})  "
                  f"unreached {s['unreached']}{flag}")
    print("Table 6.6:")
    for ch, r in res["table_6_6"].items():
        eq = {n: (f"{'>' if v['new_lower_bound'] else ''}{v['new']:+.2f}", f"{v['published']:+.2f}") for n, v in r["equalizer"].items()}
        print(f"  {ch:12s} eq {eq}  learned old {r['learned_old']['best']:+.2f} (median {r['learned_old']['median']:+.2f}) "
              f"new {r['learned_new']['best']:+.2f} (median {r['learned_new']['median_flag']}{r['learned_new']['median']:+.2f}) "
              f"cfg {r['learned_new']['params']}p w{r['learned_new']['window']}")
    print("Table 9.4 (budget cells):")
    for ch, r in res["table_9_4"].items():
        mark = "  <-- CHANGES" if r["old_cells"] != r["new_cells"] else ""
        print(f"  {ch:14s} old {r['old_cells']}\n  {'':14s} new {r['new_cells']}{mark}  missing-target configs: {len(r['configs_missing_targets'])}")
    print("Table 9.5 (seed spread):")
    for ch, r in res["table_9_5"].items():
        o, n = r["old"], r["new"]
        print(f"  {ch:14s} old {o['configs']:3d} {o['median']:.3f} {o['p90']:.3f} {o['max']:.3f}   new {n['configs']:3d} {n['median']:.3f} {n['p90']:.3f} {n['max']:.3f}  bounded {r.get('new_bounded_configs', '')} undefined {r.get('new_undefined_configs', '')}")
    print("Table 9.6:")
    for a, s in res["table_9_6"].items():
        print(f"  {a:15s} published spread {s['published_spread']:.3f}  old {s['old_per_seed']}  new {s['new_per_seed']} {s['new_lower_bound']} spread {s['new_spread_flag']}{s['new_spread']:.3f}")
    print("Native spreads:", {a: (round(s['published_spread'], 3), s['new_spread_flag'] + f"{s['new_spread']:.3f}") for a, s in res["native_spreads"].items()})
    ti = res["transformer_instability"]
    print("Transformer instability: per-seed new", [round(x, 3) for x in ti["Transformer-3K"]["new_per_seed"]],
          "bounds", ti["Transformer-3K"]["new_lower_bound"], "r", ti["correlation_new"])
    for f, rows in res["coded_minsize"].items():
        ch = [(r["params"], round(r["old_best"], 2), r["new_best_flag"] + f"{r['new_best']:.2f}") for r in rows if abs(r["old_best"] - r["new_best"]) > 0.005 or r["new_best_flag"]]
        print(f"Coded {f}: range old {min(r['old_best'] for r in rows):+.2f}..{max(r['old_best'] for r in rows):+.2f}  "
              f"new {min(r['new_best'] for r in rows):+.2f}..{max(r['new_best'] for r in rows):+.2f}  changed {ch}")


def main():
    res = {"description": __doc__.split("\n\n")[0], "git_rev": git_rev(),
           "rule": "worst penalty over the targets the reference curve reaches; a relay that misses one is a lower bound",
           "table_6_7": table_6_7(), "table_6_6": table_6_6(), "table_9_4": table_9_4(),
           "table_9_5": table_9_5(), "table_9_6": table_9_6(), "native_spreads": native_spreads(),
           "transformer_instability": transformer_instability(), "coded_minsize": coded_minsize(),
           "window_crossover": window_crossover(), "seq_rerun": seq_rerun()}
    report(res)
    wc = res["window_crossover"]
    print("Window crossover (memory channels):", {ch: {w: f"{v['new_flag']}{v['new']:.2f}" for w, v in c["windows"].items()} for ch, c in wc.items() if c["memory"] > 1})
    print("  w1->w7 gain:", {ch: f"{c['gain_w1_to_w7']['flag']}{c['gain_w1_to_w7']['value']:.2f}" for ch, c in wc.items() if c["memory"] > 1})
    sr = res["seq_rerun"]
    print(f"Seq rerun: outside {sr['limit_db']} dB, comparable rule {sr['outside_new']}; published rule {sr['outside_old']}")
    print("Table 9.5 exactness:", {ch: r["new_exact"] for ch, r in res["table_9_5"].items()})
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
