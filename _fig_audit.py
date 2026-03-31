import sys, io, json, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Map every thesis.md figure to its JSON source
FIGURES = [
    ("Fig 9",  "results/awgn_comparison_ci.png",           "results/modulation/bpsk_awgn.json"),
    ("Fig 10", "results/fading_comparison.png",            "results/modulation/bpsk_rayleigh.json"),
    ("Fig 11", "results/rician_comparison_ci.png",         None),
    ("Fig 12", "results/mimo_2x2_comparison_ci.png",       None),
    ("Fig 13", "results/mimo_2x2_mmse_comparison_ci.png",  None),
    ("Fig 14", "results/mimo_2x2_sic_comparison_ci.png",   None),
    ("Fig 15", "results/normalized_3k_all_channels.png",   None),
    ("Fig 16", "results/normalized_3k_awgn.png",           None),
    ("Fig 17", "results/normalized_3k_rayleigh.png",       None),
    ("Fig 18", "results/normalized_3k_rician_k3.png",      None),
    ("Fig 19", "results/complexity_comparison_all_relays.png", None),
    ("Fig 20", "results/master_ber_comparison.png",        None),
    ("Fig 21", "results/modulation/bpsk_awgn_ci.png",      "results/modulation/bpsk_awgn.json"),
    ("Fig 22", "results/modulation/bpsk_rayleigh_ci.png",  "results/modulation/bpsk_rayleigh.json"),
    ("Fig 23", "results/modulation/qpsk_awgn_ci.png",      "results/modulation/qpsk_awgn.json"),
    ("Fig 24", "results/modulation/qpsk_rayleigh_ci.png",  "results/modulation/qpsk_rayleigh.json"),
    ("Fig 25", "results/modulation/qam16__awgn_ci.png",    "results/modulation/qam16_awgn.json"),
    ("Fig 26", "results/modulation/qam16__rayleigh_ci.png","results/modulation/qam16_rayleigh.json"),
    ("Fig 27", "results/modulation/combined_modulation_awgn.png", None),
    ("Fig 28", "results/qam16_activation/qam16_activation_awgn.png",     "results/qam16_activation/qam16_activation_awgn.json"),
    ("Fig 29", "results/qam16_activation/qam16_activation_rayleigh.png", "results/qam16_activation/qam16_activation_rayleigh.json"),
    ("Fig 30", "results/activation_comparison/bpsk_activation_awgn.png",     "results/activation_comparison/activation_qam16_awgn.json"),
    ("Fig 31", "results/activation_comparison/bpsk_activation_rayleigh.png", None),
    ("Fig 32", "results/activation_comparison/qpsk_activation_awgn.png",     None),
    ("Fig 33", "results/activation_comparison/qpsk_activation_rayleigh.png", None),
    ("Fig 34", "results/activation_comparison/qam16_activation_awgn.png",    "results/activation_comparison/activation_qam16_awgn.json"),
    ("Fig 35", "results/activation_comparison/qam16_activation_rayleigh.png","results/activation_comparison/activation_qam16_rayleigh.json"),
    ("Fig 39", "results/csi/csi_experiment_qam16_rayleigh.png", "results/csi/csi_experiment_qam16_rayleigh.json"),
    ("Fig 40", "results/csi/top3_qam16_rayleigh.png",           "results/csi/csi_experiment_qam16_rayleigh.json"),
    ("Fig 41", "results/csi/csi_experiment_psk16_rayleigh.png", "results/csi/csi_experiment_psk16_rayleigh.json"),
    ("Fig 42", "results/csi/top3_psk16_rayleigh.png",           "results/csi/csi_experiment_psk16_rayleigh.json"),
    ("Fig 50", "results/all_relays_16class/ber_all_relays_16class.png", "results/classify_16class/classify_16class_qam16.json"),
    ("Fig 53", "results/all_relays_16class/top3_16class.png",           "results/classify_16class/classify_16class_qam16.json"),
]

for fig, png, jpath in FIGURES:
    exists = os.path.exists(png)
    n_curves = "?"
    if jpath and os.path.exists(jpath):
        with open(jpath) as f:
            d = json.load(f)
        n_curves = len(d.get("results", {}))
    import PIL.Image as Image
    if exists:
        img = Image.open(png)
        w, h = img.size
        dims = "%dx%d" % (w, h)
    else:
        dims = "MISSING"
    print("%s | %3s curves | %s | %s" % (fig, n_curves, dims, png))
