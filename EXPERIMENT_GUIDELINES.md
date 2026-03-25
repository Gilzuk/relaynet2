# Experiment Guidelines

Standards and conventions for all thesis experiments in `run_experiments.py`.

---

## 1. Unified Entry Point

All experiments run through a single script:

```bash
python run_experiments.py --all          # Run everything
python run_experiments.py --exp 7.17     # Run one experiment
python run_experiments.py --list         # List available experiments
```

**Do not** add standalone experiment scripts. New experiments go into
`run_experiments.py` as a function `exp_<section>(args)` registered in the
`EXPERIMENTS` dict.

---

## 2. Quick Mode (`--quick`)

Every experiment **must** support `--quick` for fast CI / smoke testing.

| Parameter | Full | Quick |
|-----------|------|-------|
| Training samples | 20 000–50 000 | 1 000–5 000 |
| Epochs | 80–10 000 | 5–500 |
| MC trials | 10–20 | 3–5 |
| Bits per trial | 10 000 | 2 000 |

Use `args.quick` to branch:

```python
samples = 5_000 if args.quick else 50_000
epochs  = 10    if args.quick else 200
```

Quick mode is applied automatically to `args.bits_per_trial` and
`args.num_trials` by the main runner.

---

## 3. Checkpoint `.pt` Files

Every trained model **must** be saved as a `.pt` checkpoint via
`WeightManager`:

```python
wm = WeightManager(args.weights_dir, args.seed)
# After training:
wm.save("MLP 16-cls", relay, subdir="16class")
# Before training (cache check):
if not args.retrain and wm.load(name, relay, subdir=subdir):
    print("Using cached weights")
```

Checkpoints live under `weights/seed_<N>/<subdir>/<name>.pt`.

---

## 4. Retraining (`--retrain`)

When `--retrain` is passed, **skip** cache loading and retrain from scratch.
Without it, cached checkpoints are loaded if they exist.

```python
if not args.retrain and wm.load(name, relay, subdir=subdir):
    # use cached
else:
    relay.train(...)
    wm.save(name, relay, subdir=subdir)
```

Use `--inference-only` to load weights and skip training entirely (no
fallback to train).

---

## 5. JSON Output

Every experiment **must** save a `.json` file via `save_results_json()` with:

- `snr_range`: list of SNR dB values
- `results`: dict of `{name: {"ber_mean": [...], "ber_trials": [[...]], "ci_lower": [...], "ci_upper": [...]}}`
- `meta`: experiment section, modulation, channel, timestamp

JSON files go under `results/<experiment_dir>/<descriptive_name>.json`.

Charts can be regenerated from JSON without re-running via:

```bash
python run_experiments.py --regen-charts
```

---

## 6. Chart Guidelines Compliance

All charts **must** follow `CHART_GUIDELINES.md` (22 rules). Use the provided
helpers:

| Helper | Rules |
|--------|-------|
| `plot_ber_chart()` | 1–6, 10–16 (colors, markers, jitter, inset, annotations) |
| `plot_top3_chart()` | 22 (top-3 neural relays + baselines) |
| `plot_achievement_chart()` | 20 (winner highlighted, others faded) |
| `plot_summary_heatmap()` | 19 (aggregation across channels) |
| `print_ber_summary_table()` | 18 (BER table at key SNR points) |

Every experiment should produce at minimum:
1. A main BER vs SNR chart with CI bands (`*_ci.png`)
2. A Top-3 chart (`*_top3.png`)
3. A BER summary table (printed to console / log)
4. JSON results file

---

## 7. Failure Logging

Every experiment is wrapped in `run_experiment_safe()` which:

1. Catches unhandled exceptions
2. Logs the full traceback to `results/logs/experiments_<timestamp>.log`
3. Saves a JSON failure report to `results/logs/fail_<name>_<timestamp>.json`
4. Continues with remaining experiments (does not abort)

At the end, a summary prints succeeded/failed counts.

---

## 8. Adding a New Experiment

1. Write `exp_<section>(args)` following the pattern of existing experiments.
2. Register it in the `EXPERIMENTS` dict:
   ```python
   "7.18": ("Short Description", exp_7_18_my_experiment),
   ```
3. Add chart regeneration logic in `regenerate_all_charts()`.
4. Ensure `--quick`, `--retrain`, `--inference-only` all work.
5. Save JSON via `save_results_json()`.
6. Save checkpoints via `WeightManager.save()`.
7. Run tests: `python run_experiments.py --exp 7.18 --quick`

---

## 9. Directory Structure

```
relaynet2/
├── run_experiments.py           # Unified runner
├── CHART_GUIDELINES.md          # 22 publication rules
├── EXPERIMENT_GUIDELINES.md     # This file
├── Makefile                     # Build targets
├── weights/
│   └── seed_42/
│       ├── mlp_169p.pt          # §7.2-7.7 checkpoints
│       ├── 3k/                  # §7.8 normalized checkpoints
│       ├── 16class/             # §7.17 checkpoints
│       ├── e2e/                 # §7.16 checkpoints
│       └── metadata.json
├── results/
│   ├── logs/                    # Experiment logs + failure reports
│   ├── bpsk_comparison/         # §7.2-7.7
│   ├── normalized_3k/           # §7.8
│   ├── modulation/              # §7.10 + constellation diagrams
│   ├── qam16_activation/        # §7.11
│   ├── layernorm/               # §7.12
│   ├── activation_comparison/   # §7.13
│   ├── csi/                     # §7.14-7.15
│   ├── e2e/                     # §7.16
│   ├── all_relays_16class/      # §7.17
│   └── channel_analysis/        # §7.1
└── tests/                       # pytest suite
```

---

## 10. Command Quick Reference

```bash
# List experiments
python run_experiments.py --list

# Run all (full quality)
python run_experiments.py --all --gpu

# Run all (fast smoke test)
python run_experiments.py --all --quick

# Run specific experiments
python run_experiments.py --exp 7.2 7.10 7.17

# Force retrain
python run_experiments.py --exp 7.17 --retrain --gpu

# Inference only (load weights, skip training)
python run_experiments.py --all --inference-only

# Regenerate charts from JSON (no training)
python run_experiments.py --regen-charts

# Custom parameters
python run_experiments.py --exp 7.2 --seed 123 --snr-max 30 --num-trials 20

# Using build script (PowerShell)
.\make.ps1 quick              # Quick smoke test
.\make.ps1 full               # Full run with GPU
.\make.ps1 charts             # Regenerate charts
.\make.ps1 clean-results      # Clean all results
.\make.ps1 clean-weights      # Clean all weights
.\make.ps1 clean              # Clean everything
```
