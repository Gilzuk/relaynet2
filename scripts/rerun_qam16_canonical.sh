#!/usr/bin/env bash
# Re-run the two 16-QAM studies on 16-QAM's canonical channel (Rayleigh).
#
# Both previously ran on AWGN, and neither said so: 7.17 omitted channel_fn
# and so took run_monte_carlo's awgn_channel default, while 7.11 swept both
# channels. Under the pairing rule a complex constellation belongs on the
# complex channel, and AWGN is reserved for the BPSK calibration baseline.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/rerun_logs; mkdir -p "$LOG"
BUDGET="--num-trials 10 --bits-per-trial 10000 --snr-min 0 --snr-max 20 --snr-step 2"
run () {
  echo "=== $(date -u '+%H:%M:%SZ')  START $1"; t0=$SECONDS
  python3 run_experiments.py --exp "$1" $BUDGET --skip-relays cgan --retrain \
      >"$LOG/$1_rayleigh.log" 2>&1
  echo "=== $(date -u '+%H:%M:%SZ')  END   $1  rc=$?  $((SECONDS-t0))s"
}
echo "###### 16-QAM canonical re-run $(date -u '+%H:%M:%SZ'), HEAD $(git rev-parse --short HEAD)"
run 7.17    # tbl:table24
run 7.11    # tbl:table15
echo "###### finished $(date -u '+%H:%M:%SZ')"
