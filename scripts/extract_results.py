"""Extract key results from experiment JSON files for thesis."""
import json
import os
import numpy as np

results_dir = "results/csi"

for constellation in ["qam16", "psk16"]:
    fn = os.path.join(results_dir, f"csi_experiment_{constellation}_rayleigh.json")
    if not os.path.exists(fn):
        print(f"--- {constellation.upper()}: FILE NOT FOUND ---")
        continue
    with open(fn) as f:
        data = json.load(f)
    
    snr = data["snr_range"]
    results = data["results"]
    print(f"=== {constellation.upper()} ===")
    print(f"SNR range: {snr}")
    
    # Rank all neural variants by avg BER in upper half of SNR range
    half = len(snr) // 2
    ranked = []
    for name, vdata in results.items():
        if name in ["AF", "DF"]:
            continue
        avg_upper = np.mean(vdata["ber_mean"][half:])
        ranked.append((name, avg_upper, vdata["ber_mean"]))
    ranked.sort(key=lambda x: x[1])
    
    top3_names = [r[0] for r in ranked[:3]]
    print(f"Top-3: {top3_names}")
    print()
    
    # Print BER table for AF, DF, and top-3
    key_variants = ["AF", "DF"] + top3_names
    
    print(f"{'Variant':45s}", end="")
    for s in snr:
        print(f"  {s:6.0f}", end="")
    print()
    print("-" * 135)
    
    for name in key_variants:
        if name not in results:
            continue
        m = results[name]["ber_mean"]
        print(f"{name:45s}", end="")
        for b in m:
            print(f"  {b:.4f}", end="")
        print()
    
    print()
    
    # Full ranking at SNR=20
    idx_20 = snr.index(20) if 20 in snr else -1
    print(f"--- Full ranking at SNR=20 dB ---")
    ranked_20 = []
    for name, vdata in results.items():
        if name in ["AF", "DF"]:
            continue
        ranked_20.append((name, vdata["ber_mean"][idx_20]))
    ranked_20.sort(key=lambda x: x[1])
    
    for i, (name, ber) in enumerate(ranked_20):
        marker = " <-- TOP-3" if name in top3_names else ""
        print(f"  #{i+1:2d} {name:45s} BER={ber:.6f}{marker}")
    
    af_ber = results["AF"]["ber_mean"][idx_20]
    df_ber = results["DF"]["ber_mean"][idx_20]
    print(f"  --- AF: {af_ber:.6f}, DF: {df_ber:.6f}")
    print()
