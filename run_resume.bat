@echo off
cd /d C:\thesis\relaynet2
"C:\Program Files\Python311\python.exe" -u scripts\run_full_comparison.py --resume --include-sequence-models --include-normalized --gpu --log-timings > logs\resume_run_output.log 2>&1
