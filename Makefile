# Makefile for relaynet2 thesis experiments
# ==========================================
# Reproduction suite (start here after cloning -- see REPRODUCE.md):
#   make setup          - Install pinned dependencies
#   make check          - Report what this machine can reproduce
#   make verify         - Check all thesis values against their data sources
#   make repro-unknown  - Re-run the unknown-channel study, then verify
#   make repro-full     - Re-run everything (hours, needs torch)
#   make thesis         - Build the thesis PDF
#
# Experiment runner:
#   make quick          - Run all experiments in quick mode
#   make full           - Run all experiments (full quality, GPU)
#   make charts         - Regenerate charts from existing JSON
#   make test           - Run pytest suite
#   make clean          - Clean results, weights, and logs
#
# Individual experiments:
#   make exp SECTION=7.17
#   make exp SECTION="7.2 7.10 7.17"

# python3 by default: `python` is often absent on Linux/macOS.
PYTHON   ?= python3
VENV     ?= .venv
ACTIVATE  = $(VENV)/Scripts/activate
RUNNER    = run_experiments.py
SEED     ?= 42

# Detect OS for activation
ifeq ($(OS),Windows_NT)
    ACTIVATE_CMD = $(VENV)\Scripts\activate &&
else
    ACTIVATE_CMD = . $(VENV)/bin/activate &&
endif

# ──────────────────────────────────────────
# Main targets
# ──────────────────────────────────────────

.PHONY: help quick full charts test exp list clean clean-results clean-weights clean-logs \
        setup check verify repro-unknown repro-qpsk repro-full thesis bundles

help: ## Show this help
	@echo ""
	@echo "  relaynet2 -- thesis reproduction and experiment suite"
	@echo "  ====================================================="
	@echo ""
	@echo "  REPRODUCTION (see REPRODUCE.md)"
	@echo "    make setup           Install pinned dependencies"
	@echo "    make check           Report what this machine can reproduce"
	@echo "    make verify          Tier 0: check every thesis value vs its data source (~1 min)"
	@echo "    make repro-unknown   Tier 1: re-run unknown-channel study + verify (~40 min)"
	@echo "    make repro-qpsk      Tier 1b: re-run the QPSK unknown-channel study (~20 min)"
	@echo "    make repro-full      Tier 2: re-run every experiment (hours, needs torch)"
	@echo "    make bundles         Rebuild the two Overleaf zips from thesis/"
	@echo "    make thesis          Build the thesis PDF (needs XeLaTeX + latexmk)"
	@echo ""
	@echo "  EXPERIMENT RUNNER"
	@echo "    make quick           Run all experiments (quick mode)"
	@echo "    make full            Run all experiments (full, GPU)"
	@echo "    make charts          Regenerate charts from JSON"
	@echo "    make test            Run pytest suite"
	@echo "    make exp SECTION=7.17   Run specific experiment(s)"
	@echo "    make list            List available experiments"
	@echo ""
	@echo "  CLEANING"
	@echo "    make clean           Clean all outputs"
	@echo "    make clean-results   Clean results only"
	@echo "    make clean-weights   Clean weights only"
	@echo "    make clean-logs      Clean logs only"
	@echo ""

# ──────────────────────────────────────────
# Reproduction suite
# ──────────────────────────────────────────

setup: ## Install the pinned dependency set
	$(PYTHON) -m pip install -r requirements-repro.txt

check: ## Report which reproduction tiers this machine can run
	@$(PYTHON) scripts/check_env.py

verify: ## Tier 0: verify every thesis number against its data source
	@echo ">> Verifying every thesis table cell and quantitative claim..."
	$(PYTHON) verify_thesis_tables.py

repro-unknown: ## Tier 1: recompute the unknown-channel study, then verify
	@echo ">> Re-running the unknown-channel experiments, then verifying..."
	$(PYTHON) verify_thesis_tables.py --rerun

repro-qpsk: ## Tier 1b: recompute the QPSK unknown-channel study
	@echo ">> QPSK unknown channel: ISI hop 1, AWGN and Rayleigh hop 2..."
	$(PYTHON) e6_qpsk_unknown_channel.py

repro-full: ## Tier 2: recompute every experiment (hours; requires torch)
	@$(PYTHON) -c "import torch" 2>/dev/null || \
	  { echo "ERROR: torch required for this target -- see requirements-repro.txt"; exit 1; }
	@echo ">> Re-running all experiments. Hours; a GPU is strongly advised."
	$(PYTHON) $(RUNNER) --all --seed $(SEED)
	$(MAKE) verify

bundles: ## Rebuild the two Overleaf zips from thesis/
	$(PYTHON) scripts/build_bundles.py

thesis: ## Build the thesis PDF
	@command -v latexmk >/dev/null || \
	  { echo "ERROR: latexmk not found -- install TeX Live with XeLaTeX"; exit 1; }
	cd thesis && latexmk -xelatex -interaction=nonstopmode main.tex
	@echo ">> thesis/main.pdf"

list: ## List all available experiments
	$(PYTHON) $(RUNNER) --list

quick: ## Run all experiments in quick mode (fast smoke test)
	$(PYTHON) $(RUNNER) --all --quick --seed $(SEED)

full: ## Run all experiments with full quality and GPU
	$(PYTHON) $(RUNNER) --all --gpu --seed $(SEED)

full-cpu: ## Run all experiments with full quality, CPU only
	$(PYTHON) $(RUNNER) --all --seed $(SEED)

charts: ## Regenerate all charts from existing JSON files
	$(PYTHON) $(RUNNER) --regen-charts

exp: ## Run specific experiment(s): make exp SECTION=7.17
ifndef SECTION
	@echo "Usage: make exp SECTION=7.17"
	@echo "       make exp SECTION=\"7.2 7.10 7.17\""
	@exit 1
endif
	$(PYTHON) $(RUNNER) --exp $(SECTION) --gpu --seed $(SEED)

exp-quick: ## Run specific experiment(s) in quick mode
ifndef SECTION
	@echo "Usage: make exp-quick SECTION=7.17"
	@exit 1
endif
	$(PYTHON) $(RUNNER) --exp $(SECTION) --quick --seed $(SEED)

retrain: ## Force retrain specific experiment(s)
ifndef SECTION
	@echo "Usage: make retrain SECTION=7.17"
	@exit 1
endif
	$(PYTHON) $(RUNNER) --exp $(SECTION) --retrain --gpu --seed $(SEED)

inference: ## Run inference only (load weights, no training)
	$(PYTHON) $(RUNNER) --all --inference-only --seed $(SEED)

# ──────────────────────────────────────────
# Testing
# ──────────────────────────────────────────

test: ## Run full pytest suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick: ## Run pytest with reduced scope
	$(PYTHON) -m pytest tests/ -v --tb=short -x -q

# ──────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────

clean: clean-results clean-weights clean-logs ## Clean all generated outputs
	@echo All outputs cleaned.

clean-results: ## Remove all experiment results (JSON, charts)
ifeq ($(OS),Windows_NT)
	@if exist results\bpsk_comparison    rmdir /s /q results\bpsk_comparison
	@if exist results\normalized_3k      rmdir /s /q results\normalized_3k
	@if exist results\modulation         rmdir /s /q results\modulation
	@if exist results\qam16_activation   rmdir /s /q results\qam16_activation
	@if exist results\layernorm          rmdir /s /q results\layernorm
	@if exist results\activation_comparison rmdir /s /q results\activation_comparison
	@if exist results\csi                rmdir /s /q results\csi
	@if exist results\e2e                rmdir /s /q results\e2e
	@if exist results\all_relays_16class rmdir /s /q results\all_relays_16class
	@if exist results\channel_analysis   rmdir /s /q results\channel_analysis
	@if exist results\master_ber_comparison.png del results\master_ber_comparison.png
else
	rm -rf results/bpsk_comparison results/normalized_3k results/modulation
	rm -rf results/qam16_activation results/layernorm results/activation_comparison
	rm -rf results/csi results/e2e results/all_relays_16class results/channel_analysis
	rm -f results/master_ber_comparison.png
endif
	@echo Results cleaned.

clean-weights: ## Remove all saved model weights
ifeq ($(OS),Windows_NT)
	@if exist weights rmdir /s /q weights
else
	rm -rf weights/
endif
	@echo Weights cleaned.

clean-logs: ## Remove experiment logs
ifeq ($(OS),Windows_NT)
	@if exist results\logs rmdir /s /q results\logs
else
	rm -rf results/logs/
endif
	@echo Logs cleaned.

clean-pycache: ## Remove Python cache files
ifeq ($(OS),Windows_NT)
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
endif
	@echo Cache cleaned.
