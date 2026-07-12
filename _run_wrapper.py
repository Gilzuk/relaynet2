"""Wrapper to run the experiment and signal completion."""
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-u", r"c:\thesis\relaynet2\_all_relays_16class.py"],
    cwd=r"c:\thesis\relaynet2",
    capture_output=False,
)
# Write a sentinel file when done
with open(r"c:\thesis\relaynet2\_experiment_done.txt", "w") as f:
    f.write(f"exit_code={result.returncode}\n")
