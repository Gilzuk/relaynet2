import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

files = [
    "results/bpsk_comparison/awgn.json",
    "results/normalized_3k/3k_awgn.json",
    "results/all_relays_16class/all_relays_16class.json",
    "results/activation_comparison/activation_qam16_awgn.json",
    "results/qam16_activation/qam16_activation_awgn.json",
    "results/csi/csi_experiment_qam16_rayleigh.json",
    "results/classify_16class/classify_16class_qam16.json",
]
for p in files:
    if not os.path.exists(p): print("MISSING:", p); continue
    with open(p) as f: d = json.load(f)
    keys = list(d.keys())
    results = d.get("results", {})
    n = len(results)
    labels = list(results.keys())[:6]
    print(f"{p}")
    print(f"  top keys: {keys[:6]}")
    print(f"  n_curves: {n}  labels[:6]: {labels}")
    if n > 0:
        first = list(results.values())[0]
        print(f"  entry keys: {list(first.keys())[:5]}")
    print()
