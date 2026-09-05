#!/usr/bin/env bash
#
# Full re-run of every simulation feeding a thesis table, on the corrected
# Eb/N0 channel convention (commit 97ca397).
#
# Why everything and not just the AWGN/BPSK tables: the 3 dB fix changed
# relaynet/channels/awgn.py, and every learned relay generates its *training*
# data through awgn_channel (relaynet/utils/activations.py:214). So the fix
# moved the trained weights too, and results on Rayleigh and on QPSK/16-QAM
# are stale for that reason even though their own evaluation channel was
# never touched.
#
# Relay-set policy, two tiers, deliberately not uniform. cGAN is dropped
# everywhere per the lean-configuration instruction (Table 5.3 records it at
# 7,293 s on a GPU, and this box has no CUDA); beyond that:
#
#   BREADTH (all eight remaining relays) for the tables whose entire purpose
#   is to compare architectures against each other: 7.2/7.3 (tbl:table2, the
#   canonical headline comparison), 7.10 (tbl:table14ray / tbl:table14),
#   7.8 (tbl:table8, hypothesis H4 "architectures converge at equal parameter
#   count") and 7.17 (tbl:table24, 4-class vs 16-class head per architecture).
#   Cutting these to three learned models would not shrink the finding, it
#   would delete it.
#
#   LEAN (AF, DF, MLP, Transformer, Mamba-2) for the activation ablations,
#   7.11 and 7.13. There the variable under study is the output activation,
#   not the architecture, and 7.13 alone is twelve modulation x channel x
#   activation combinations — the single most expensive stage in the file.
#
# Dropping cGAN costs tbl:table24 its cGAN row and tbl:table2 its cGAN
# column. The thesis text has to stop citing those rather than carry a
# pre-correction number forward.
#
# Budget is the documented one (ch04_methods.tex:197): M=10 trials x
# N=10,000 bits = 100,000 bits per SNR point, SNR 0:2:20. Defaults in
# run_experiments.py already match, so they are passed explicitly here only
# to keep the record in the log.
#
# Runs strictly sequentially: this box has 4 cores, no CUDA, and torch takes
# all 4 threads, so running stages in parallel would only thrash.

set -u  # deliberately NOT -e: one failing stage must not abandon the rest

cd "$(dirname "$0")/.." || exit 1

LOG_DIR="results/rerun_logs"
mkdir -p "$LOG_DIR"

LEAN="hybrid vae cgan mamba_s6"
BREADTH="cgan"
BUDGET="--num-trials 10 --bits-per-trial 10000 --snr-min 0 --snr-max 20 --snr-step 2"

run_stage () {
    local tag="$1"; shift
    local skip="$1"; shift
    local log="$LOG_DIR/${tag}.log"
    echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ')  START $tag  (skip: ${skip:-none})"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    python3 run_experiments.py --exp "$tag" $BUDGET \
        ${skip:+--skip-relays $skip} --retrain >"$log" 2>&1
    local rc=$?
    echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ')  END   $tag  rc=$rc  $((SECONDS - t0))s"
    if [ $rc -ne 0 ]; then
        echo "--- last 20 lines of $log:"
        tail -20 "$log"
    fi
    return $rc
}

echo "############ re-run started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "############ HEAD $(git rev-parse --short HEAD)"

# Ordered by how much of the thesis depends on the output, so that an
# interrupted run still leaves the load-bearing tables refreshed. 7.2 leads
# because results/bpsk_comparison/rayleigh.json — the source for tbl:table2,
# the canonical headline table — still carries created=2026-03-23: it is the
# copy restored after an earlier lean run overwrote it, so it has never been
# regenerated on the corrected axis at all. 7.13 is last because it is the
# most expensive stage and feeds figures only, no verifier-checked table.
run_stage 7.1  ""           # channel-model validation figures (minutes)
run_stage 7.2  "$BREADTH"   # tbl:table2 + tbl:table2awgn — canonical, both channels
run_stage 7.10 "$BREADTH"   # tbl:table14ray (Table 5.6, canonical QPSK), tbl:table14
run_stage 7.17 "$BREADTH"   # tbl:table24
run_stage 7.8  "$BREADTH"   # tbl:table8 (H4)
run_stage 7.11 "$LEAN"      # tbl:table15
run_stage 7.13 "$LEAN"      # activation study figures

echo "############ re-run finished $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
