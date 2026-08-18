#!/usr/bin/env bash
# Re-run the three cGAN-bearing studies with the cGAN included.
#
# Why not reuse the previous cGAN numbers: relaynet/relays/cgan.py was created
# on 2026-04-01, and the surviving cGAN results are dated 2026-03-23, so they
# come from a superseded implementation. The VAE showed what that costs -- its
# March numbers (0.3928 at 0 dB) differ from the current implementation's
# (0.3359) by far more than the SNR-convention change accounts for, which the
# A/B in results/vae_convention_ab.json demonstrates directly. Splicing March
# cGAN values into tables measured with today's code would repeat that error.
#
# Why re-run each study whole rather than the cGAN alone: training the cGAN
# consumes the shared RNG stream, so its presence perturbs every other relay's
# numbers. Running it separately and pasting in one column would produce a
# table whose rows came from two different draws. Each affected study is
# therefore re-run end to end and re-transcribed from the single new file.
#
# 7.8 is not included: the normalized-3K study excludes the cGAN by design
# (build_all_3k defaults include_cgan=False) and tbl:table8 has no cGAN column.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/rerun_logs; mkdir -p "$LOG"
BUDGET="--num-trials 10 --bits-per-trial 10000 --snr-min 0 --snr-max 20 --snr-step 2"
run () {
  echo "=== $(date -u '+%H:%M:%SZ')  START $1"; t0=$SECONDS
  python3 run_experiments.py --exp "$1" $BUDGET --retrain >"$LOG/$1_cgan.log" 2>&1
  echo "=== $(date -u '+%H:%M:%SZ')  END   $1  rc=$?  $((SECONDS-t0))s"
}
echo "###### cGAN-inclusive re-run $(date -u '+%H:%M:%SZ'), HEAD $(git rev-parse --short HEAD)"
run 7.10   # tbl:table2 (canonical), tbl:table14
run 7.17   # tbl:table24
run 7.11   # tbl:table15
echo "###### finished $(date -u '+%H:%M:%SZ')"
