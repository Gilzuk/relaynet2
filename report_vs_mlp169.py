"""Smallest relay matching or beating MLP-169, per channel.

A different question from the rest of this study, and a cleaner one. The
other scripts ask how small a relay can be and still match the *classical*
comparator, which drags in whether that comparator is any good on the channel
-- the reason baseline_diagnostics exists. This asks how small a relay can be
and still match *the thesis's own 169-parameter relay*, w=5 h1-24p, which is
the number Chapter 5 actually reports.

That comparison is immune to baseline quality. Both sides are the same kind
of object, trained the same way, evaluated on identical channel draws and
identical payload bits, so whatever the channel does to one it does to the
other.

REFERENCE. MLP-169's mean BER over its three initializations, per SNR.
Averaging is the right choice for the reference specifically: a single
initialization of the incumbent is a draw, and the question "can something
smaller do what MLP-169 does" is about the architecture, not about which
weights it happened to land on.

CANDIDATE. Every configuration is held to its *worst* initialization, the
same all-seeds discipline used elsewhere: a size counts only if it works
whichever way it initializes, not on its best day.

That pairing is deliberately unfair to the candidate -- worst-of-three
against a mean -- so a size that clears it has cleared a conservative bar.

VERDICTS. Two, reported separately:
  matches  worst-case relative BER penalty vs MLP-169 <= TOL at every SNR
  beats    candidate BER <= MLP-169 BER at every SNR, i.e. never worse

High-SNR SNRs where MLP-169 has driven BER to ~0 are excluded from the
relative test rather than allowed to divide by nothing; they are still
included in the "beats" test, which is an absolute comparison.
"""

import json
import sys

import numpy as np

from deep_mlp_relay import arch_label

TOL = 0.02
REF_W, REF_H = 5, 24
FLOOR = 1e-5          # below this, a relative penalty is not meaningful


def analyse(path="results/mlp_min_size_all_channels.json", only_memory=None):
    d = json.load(open(path))
    snrs = d["snr_db"]

    for name, r in d["channels"].items():
        if only_memory is not None and r["memory"] != only_memory:
            continue
        ref_row = [x for x in r["sweep"]
                   if x["window"] == REF_W and x["hidden"] == REF_H]
        if not ref_row:
            print(f"{name}: no MLP-169 row, skipping")
            continue
        ref = np.mean([s["ber"] for s in ref_row[0]["seed_runs"]], axis=0)

        print(f"\n{'=' * 88}")
        print(f"  {name}   memory {r['memory']} tap(s), {r['modulation']}, "
              f"vs MLP-169 (w=5 {arch_label(24, 1)})")
        print(f"{'=' * 88}")
        print(f"  MLP-169 mean BER   " + "  ".join(f"{x:.5f}" for x in ref))
        usable = [i for i, x in enumerate(ref) if x > FLOOR]
        skipped = [snrs[i] for i in range(len(ref)) if i not in usable]
        if skipped:
            print(f"  (relative test skips {skipped} dB: MLP-169 BER ~ 0 there)")

        rows = []
        for x in r["sweep"]:
            if x["window"] == REF_W and x["hidden"] == REF_H:
                continue
            worst = np.max([s["ber"] for s in x["seed_runs"]], axis=0)
            rel = [(worst[i] - ref[i]) / ref[i] for i in usable]
            rows.append({
                "params": x["params"], "window": x["window"],
                "arch": arch_label(x["hidden"], 1),
                "worst_rel": max(rel) if rel else float("nan"),
                "matches": bool(rel and max(rel) <= TOL),
                "beats": bool(np.all(worst <= ref + 1e-12)),
            })

        rows.sort(key=lambda z: z["params"])
        print(f"\n  {'params':>6} {'w':>2}  {'architecture':<16} "
              f"{'worst rel vs 169':>17}  {'matches':>8} {'beats':>6}")
        for z in rows:
            print(f"  {z['params']:>6} {z['window']:>2}  {z['arch']:<16} "
                  f"{100 * z['worst_rel']:>+16.1f}%  "
                  f"{'yes' if z['matches'] else '-':>8} "
                  f"{'yes' if z['beats'] else '-':>6}")

        m = [z for z in rows if z["matches"]]
        b = [z for z in rows if z["beats"]]
        print()
        if m:
            z = min(m, key=lambda q: q["params"])
            print(f"  -> smallest MATCHING MLP-169: {z['params']} params "
                  f"(w={z['window']} {z['arch']}) -- "
                  f"{169 / z['params']:.1f}x smaller")
        else:
            print("  -> no configuration matches MLP-169 within tolerance")
        if b:
            z = min(b, key=lambda q: q["params"])
            print(f"  -> smallest BEATING MLP-169 at every SNR: {z['params']} "
                  f"params (w={z['window']} {z['arch']}) -- "
                  f"{169 / z['params']:.1f}x smaller")
        else:
            print("  -> no configuration beats MLP-169 at every SNR")


if __name__ == "__main__":
    mem = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyse(only_memory=mem)
