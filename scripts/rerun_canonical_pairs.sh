#!/usr/bin/env bash
#
# Re-run the thesis's three (modulation, channel) configurations:
#
#   BPSK  x AWGN      calibration reference (closed forms exist here)
#   QPSK  x Rayleigh  canonical operating point
#   16-QAM x Rayleigh extension (per-axis relay formulation breaks)
#
# The omitted pairings are deliberate, not gaps. QPSK and 16-QAM on AWGN are
# dropped because AWGN is a measuring stick and a measuring stick needs only
# the constellation whose closed form is being checked; BPSK on Rayleigh is
# dropped per instruction. See CANONICAL_PAIRS in run_experiments.py.
#
# Budget is the documented one (ch04_methods.tex): M=10 trials x N=10,000
# bits = 100,000 bits per SNR point, SNR 0:2:20.
#
# cGAN is excluded throughout (lean configuration; ~2/3 of a full pass, and
# this box has no CUDA). Runs sequentially: 4 cores, torch takes all of them.
#
# setsid detaches the run from the launching shell's process group. An
# earlier attempt with plain nohup was killed mid-stage when its parent
# went away, losing 80 minutes of stage 7.2 with no END line written.

set -u

cd "$(dirname "$0")/.." || exit 1

LOG_DIR="results/rerun_logs"
mkdir -p "$LOG_DIR"

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
    [ $rc -ne 0 ] && { echo "--- last 20 lines of $log:"; tail -20 "$log"; }
    return $rc
}

echo "############ canonical-pair re-run started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "############ HEAD $(git rev-parse --short HEAD)"

# 7.10 is the experiment that produces all three configurations in one pass,
# from one relay set, so it goes first and is the one that matters.
run_stage 7.10 "cgan"   # bpsk_awgn, qpsk_rayleigh, qam16_rayleigh
run_stage 7.1  ""       # channel-model validation figures (minutes)
run_stage 7.17 "cgan"   # 16-class 2D study, tbl:table24
run_stage 7.8  "cgan"   # normalized-3K, tbl:table8 (hypothesis H4)

echo "############ canonical-pair re-run finished $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
