#!/usr/bin/env python3
"""Link-adaptation envelopes with matched block DF (Tables tbl:table42, tbl:table43).

Derived only; nothing is simulated. Block DF uses the matched-metric run
(results/coded_rate_adaptation_matched.json, |h|^2-weighted relay and destination
decoders). The denoise relay forwards hard bit decisions, so the |h2|^2 destination
weight does not model its relay errors and lowers its goodput; it is therefore
taken with the destination metric that serves it better, the unweighted one, which
is the published run (results/coded_rate_adaptation.json). Its weighted-destination
envelope is also recorded. Envelopes use coded_latency_capacity's own latency
accounting and best_at_budget.
"""

import json
import math

from coded_latency_capacity import best_at_budget

BUDGETS = {"unconstrained": math.inf, "150": 150, "100": 100}


def entries_of(d, relay_filter):
    out = {}
    for key, v in d["mcs"].items():
        mod, rate, relay = key.split("|")
        if relay_filter(relay, rate):
            out[(mod, rate, relay)] = v
    return out


def main():
    pub = json.load(open("results/coded_rate_adaptation.json"))
    mat = json.load(open("results/coded_rate_adaptation_matched.json"))
    assert pub["snr_db"] == mat["snr_db"]
    snrs = pub["snr_db"]
    # Uncoded MCS rows carry no code to decode, so they are stored once (labelled
    # denoise) and are open to both relays, as in coded_latency_capacity.
    matched_blockdf = entries_of(mat, lambda r, rate: r == "blockdf" or rate == "uncoded")
    denoise_plain = entries_of(pub, lambda r, rate: r == "denoise")
    denoise_wdest = entries_of(mat, lambda r, rate: r == "denoise")

    out = {"snr_db": snrs, "source": {
        "blockdf": "results/coded_rate_adaptation_matched.json",
        "denoise": "results/coded_rate_adaptation.json (unweighted destination)",
        "denoise_wdest": "results/coded_rate_adaptation_matched.json"},
        "n_trials": {"blockdf": mat["n_trials"], "denoise": pub["n_trials"]},
        "envelopes": {}}
    for name, lmax in BUDGETS.items():
        rows = []
        for i, snr in enumerate(snrs):
            row = {"snr_db": snr}
            for label, ent, relay in (("blockdf", matched_blockdf, "blockdf"),
                                      ("denoise", denoise_plain, "denoise"),
                                      ("denoise_wdest", denoise_wdest, "denoise")):
                b = best_at_budget(ent, relay, i, lmax)
                row[label] = b
            rows.append(row)
        out["envelopes"][name] = rows
        print(f"\n== envelope, budget {name}")
        for r in rows:
            f = lambda b: f"{b['mod']} {b['rate']:7s} G={b['goodput']:.4f}" if b else "none"
            print(f"  {r['snr_db']:>3} dB  blockdf(matched) {f(r['blockdf'])}   "
                  f"denoise {f(r['denoise'])}   denoise|wdest {f(r['denoise_wdest'])}")
    with open("results/coded_matched_envelopes.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/coded_matched_envelopes.json")


if __name__ == "__main__":
    main()
